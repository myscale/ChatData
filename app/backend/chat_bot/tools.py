import hashlib
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import List

import requests
from clickhouse_sqlalchemy import types, engines
from langchain.schema.embeddings import Embeddings
from sqlalchemy import Column, Text
from streamlit.runtime.uploaded_file_manager import UploadedFile


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
            url="https://api.unstructured.io/general/v0/general",
            headers=headers,
            data=data,
            files=file_data
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
        embeddings = embeddings.embed_documents(
            [t["text"] for _, t in enumerate(texts)])
        for i, _ in enumerate(texts):
            texts[i]["vector"] = embeddings[i]
        return texts
    raise ValueError("No texts extracted!")


def create_message_history_table(table_name: str, base_class):
    class Message(base_class):
        __tablename__ = table_name
        id = Column(types.Float64)
        session_id = Column(Text)
        user_id = Column(Text)
        msg_id = Column(Text, primary_key=True)
        type = Column(Text)
        # should be additions, formal developer mistake spell it.
        addtionals = Column(Text)
        message = Column(Text)
        __table_args__ = (
            engines.MergeTree(
                partition_by='session_id',
                order_by=('id', 'msg_id')
            ),
            {'comment': 'Store Chat History'}
        )

    return Message


def create_session_table(table_name: str, DynamicBase):
    class Session(DynamicBase):
        __tablename__ = table_name
        user_id = Column(Text)
        session_id = Column(Text, primary_key=True)
        system_prompt = Column(Text)
        # represent create time.
        create_by = Column(types.DateTime)
        # should be additions, formal developer mistake spell it.
        additionals = Column(Text)
        __table_args__ = (
            engines.MergeTree(order_by=session_id),
            {'comment': 'Store Session and Prompts'}
        )

    return Session
