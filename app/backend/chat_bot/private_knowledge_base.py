import hashlib
from datetime import datetime
from typing import List, Optional

import pandas as pd
from clickhouse_connect import get_client
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.myscale import MyScaleWithoutJSON, MyScaleSettings
from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.chat_bot.tools import parse_files, extract_embedding
from backend.construct.build_retriever_tool import create_retriever_tool


class ChatBotKnowledgeTable:
    def __init__(self, host, port, username, password,
                 embedding: Embeddings, parser_api_key: str, db="chatdata",
                 kb_table="private_kb", tool_table="private_tool") -> None:
        super().__init__()
        personal_files_schema_ = f"""
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

        # `tool_name` represent private knowledge database name.
        private_knowledge_base_schema_ = f"""
            CREATE TABLE IF NOT EXISTS {db}.{tool_table}(
                tool_id String,
                tool_name String,
                file_names Array(String),
                user_id String,
                created_by DateTime,
                tool_description String
            ) ENGINE = ReplacingMergeTree ORDER BY tool_id
        """
        self.personal_files_table = kb_table
        self.private_knowledge_base_table = tool_table
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
        client.command(personal_files_schema_)
        client.command(private_knowledge_base_schema_)
        self.parser_api_key = parser_api_key
        self.vector_store = MyScaleWithoutJSON(
            embedding=embedding,
            config=config,
            must_have_cols=["file_name", "text", "created_by"],
        )

    # List all files with given `user_id`
    def list_files(self, user_id: str):
        query = f"""
        SELECT DISTINCT file_name, COUNT(entity_id) AS num_paragraph, 
            arrayMax(arrayMap(x->length(x), groupArray(text))) AS max_chars
        FROM {self.vector_store.config.database}.{self.personal_files_table}
        WHERE user_id = '{user_id}' GROUP BY file_name
        """
        return [r for r in self.vector_store.client.query(query).named_results()]

    # Parse and embedding files
    def add_by_file(self, user_id, files: List[UploadedFile]):
        data = parse_files(self.parser_api_key, user_id, files)
        data = extract_embedding(self.vector_store.embeddings, data)
        self.vector_store.client.insert_df(
            table=self.personal_files_table,
            df=pd.DataFrame(data),
            database=self.vector_store.config.database,
        )

    # Remove all files and private_knowledge_bases with given `user_id`
    def clear(self, user_id: str):
        self.vector_store.client.command(
            f"DELETE FROM {self.vector_store.config.database}.{self.personal_files_table} "
            f"WHERE user_id='{user_id}'"
        )
        query = f"""DELETE FROM {self.vector_store.config.database}.{self.private_knowledge_base_table} 
                    WHERE user_id  = '{user_id}'"""
        self.vector_store.client.command(query)

    def create_private_knowledge_base(
            self, user_id: str, tool_name: str, tool_description: str, files: Optional[List[str]] = None
    ):
        self.vector_store.client.insert_df(
            self.private_knowledge_base_table,
            pd.DataFrame(
                [
                    {
                        "tool_id": hashlib.sha256(
                            (user_id + tool_name).encode("utf-8")
                        ).hexdigest(),
                        "tool_name": tool_name,  # tool_name represent user's private knowledge base.
                        "file_names": files,
                        "user_id": user_id,
                        "created_by": datetime.now(),
                        "tool_description": tool_description,
                    }
                ]
            ),
            database=self.vector_store.config.database,
        )

    # Show all private knowledge bases with given `user_id`
    def list_private_knowledge_bases(self, user_id: str, private_knowledge_base=None):
        extended_where = f"AND tool_name = '{private_knowledge_base}'" if private_knowledge_base else ""
        query = f"""
        SELECT tool_name, tool_description, length(file_names) 
        FROM {self.vector_store.config.database}.{self.private_knowledge_base_table}
        WHERE user_id = '{user_id}' {extended_where}
        """
        return [r for r in self.vector_store.client.query(query).named_results()]

    def remove_private_knowledge_bases(self, user_id: str, private_knowledge_bases: List[str]):
        private_knowledge_bases = ",".join([f"'{t}'" for t in private_knowledge_bases])
        query = f"""DELETE FROM {self.vector_store.config.database}.{self.private_knowledge_base_table}
                    WHERE user_id  = '{user_id}' AND tool_name IN [{private_knowledge_bases}]"""
        self.vector_store.client.command(query)

    def as_retrieval_tools(self, user_id, tool_name=None):
        private_knowledge_bases = self.list_private_knowledge_bases(user_id=user_id, private_knowledge_base=tool_name)
        retrievers = {
            t["tool_name"]: create_retriever_tool(
                self.vector_store.as_retriever(
                    search_kwargs={
                        "where_str": (
                            f"user_id='{user_id}' "
                            f"""AND file_name IN (
                                SELECT arrayJoin(file_names) FROM (
                                    SELECT file_names 
                                    FROM {self.vector_store.config.database}.{self.private_knowledge_base_table}
                                    WHERE user_id = '{user_id}' AND tool_name = '{t['tool_name']}')
                        )"""
                        )
                    },
                ),
                tool_name=t["tool_name"],
                description=t["tool_description"],
            )
            for t in private_knowledge_bases
        }
        return retrievers
