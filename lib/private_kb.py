import pandas as pd
import hashlib
import requests
from typing import List, Optional
from datetime import datetime
from langchain.schema.embeddings import Embeddings
from streamlit.runtime.uploaded_file_manager import UploadedFile
from clickhouse_connect import get_client
from multiprocessing.pool import ThreadPool
from langchain.vectorstores.myscale import MyScaleWithoutJSON, MyScaleSettings
from .helper import create_retriever_tool

parser_url = "https://api.unstructured.io/general/v0/general"


def parse_files(api_key, user_id, files: List[UploadedFile]):
    def parse_file(file: UploadedFile):
        headers = {
            "accept": "application/json",
            "unstructured-api-key": api_key,
        }
        data = {"strategy": "auto", "ocr_languages": ["eng"]}
        file_hash = hashlib.sha256(file.read()).hexdigest()
        file_data = {"files": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            parser_url, headers=headers, data=data, files=file_data
        )
        json_response = response.json()
        if response.status_code != 200:
            raise ValueError(str(json_response))
        texts = [
            {
                "text": t["text"],
                "file_name": t["metadata"]["filename"],
                "entity_id": hashlib.sha256(
                    (file_hash + t["text"]).encode()
                ).hexdigest(),
                "user_id": user_id,
                "created_by": datetime.now(),
            }
            for t in json_response
            if t["type"] == "NarrativeText" and len(t["text"].split(" ")) > 10
        ]
        return texts

    with ThreadPool(8) as p:
        rows = []
        for r in p.imap_unordered(parse_file, files):
            rows.extend(r)
        return rows


def extract_embedding(embeddings: Embeddings, texts):
    if len(texts) > 0:
        embs = embeddings.embed_documents([t["text"] for _, t in enumerate(texts)])
        for i, _ in enumerate(texts):
            texts[i]["vector"] = embs[i]
        return texts
    raise ValueError("No texts extracted!")


class PrivateKnowledgeBase:
    def __init__(
        self,
        host,
        port,
        username,
        password,
        embedding: Embeddings,
        parser_api_key,
        db="chat",
        kb_table="private_kb",
        tool_table="private_tool",
    ) -> None:
        super().__init__()
        kb_schema_ = f"""
            CREATE TABLE IF NOT EXISTS {db}.{kb_table}(
                entity_id String,
                file_name String,
                text String,
                user_id String,
                created_by DateTime,
                vector Array(Float32),
                CONSTRAINT cons_vec_len CHECK length(vector) = 768,
                VECTOR INDEX vidx vector TYPE MSTG('metric_type=Cosine')
            ) ENGINE = ReplacingMergeTree ORDER BY entity_id
        """
        tool_schema_ = f"""
            CREATE TABLE IF NOT EXISTS {db}.{tool_table}(
                tool_id String,
                tool_name String,
                file_names Array(String),
                user_id String,
                created_by DateTime,
                tool_description String
            ) ENGINE = ReplacingMergeTree ORDER BY tool_id
        """
        self.kb_table = kb_table
        self.tool_table = tool_table
        config = MyScaleSettings(
            host=host,
            port=port,
            username=username,
            password=password,
            database=db,
            table=kb_table,
        )
        client = get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
        )
        client.command("SET allow_experimental_object_type=1")
        client.command(kb_schema_)
        client.command(tool_schema_)
        self.parser_api_key = parser_api_key
        self.vstore = MyScaleWithoutJSON(
            embedding=embedding,
            config=config,
            must_have_cols=["file_name", "text", "created_by"],
        )

    def list_files(self, user_id, tool_name=None):
        query = f"""
        SELECT DISTINCT file_name, COUNT(entity_id) AS num_paragraph, 
            arrayMax(arrayMap(x->length(x), groupArray(text))) AS max_chars
        FROM {self.vstore.config.database}.{self.kb_table}
        WHERE user_id = '{user_id}' GROUP BY file_name
        """
        return [r for r in self.vstore.client.query(query).named_results()]

    def add_by_file(
        self, user_id, files: List[UploadedFile], **kwargs
    ):
        data = parse_files(self.parser_api_key, user_id, files)
        data = extract_embedding(self.vstore.embeddings, data)
        self.vstore.client.insert_df(
            self.kb_table,
            pd.DataFrame(data),
            database=self.vstore.config.database,
        )

    def clear(self, user_id):
        self.vstore.client.command(
            f"DELETE FROM {self.vstore.config.database}.{self.kb_table} "
            f"WHERE user_id='{user_id}'"
        )
        query = f"""DELETE FROM {self.vstore.config.database}.{self.tool_table} 
                    WHERE user_id  = '{user_id}'"""
        self.vstore.client.command(query)

    def create_tool(
        self, user_id, tool_name, tool_description, files: Optional[List[str]] = None
    ):
        self.vstore.client.insert_df(
            self.tool_table,
            pd.DataFrame(
                [
                    {
                        "tool_id": hashlib.sha256(
                            (user_id + tool_name).encode("utf-8")
                        ).hexdigest(),
                        "tool_name": tool_name,
                        "file_names": files,
                        "user_id": user_id,
                        "created_by": datetime.now(),
                        "tool_description": tool_description,
                    }
                ]
            ),
            database=self.vstore.config.database,
        )

    def list_tools(self, user_id, tool_name=None):
        extended_where = f"AND tool_name = '{tool_name}'" if tool_name else ""
        query = f"""
        SELECT tool_name, tool_description, length(file_names) 
        FROM {self.vstore.config.database}.{self.tool_table}
        WHERE user_id = '{user_id}' {extended_where}
        """
        return [r for r in self.vstore.client.query(query).named_results()]

    def remove_tools(self, user_id, tool_names):
        tool_names = ",".join([f"'{t}'" for t in tool_names])
        query = f"""DELETE FROM {self.vstore.config.database}.{self.tool_table}
                    WHERE user_id  = '{user_id}' AND tool_name IN [{tool_names}]"""
        self.vstore.client.command(query)

    def as_tools(self, user_id, tool_name=None):
        tools = self.list_tools(user_id=user_id, tool_name=tool_name)
        retrievers = {
            t["tool_name"]: create_retriever_tool(
                self.vstore.as_retriever(
                    search_kwargs={
                        "where_str": (
                            f"user_id='{user_id}' "
                            f"""AND file_name IN (
                                SELECT arrayJoin(file_names) FROM (
                                    SELECT file_names 
                                    FROM {self.vstore.config.database}.{self.tool_table}
                                    WHERE user_id = '{user_id}' AND tool_name = '{t['tool_name']}')
                        )"""
                        )
                    },
                ),
                name=t["tool_name"],
                description=t["tool_description"],
            )
            for t in tools
        }
        return retrievers
