import json
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from langchain.schema import BaseChatMessageHistory
from datetime import datetime
from sqlalchemy import Column, Text, orm, create_engine
from clickhouse_sqlalchemy import types, engines
from .schemas import create_message_model, create_session_table

def get_sessions(engine, model_class, user_id):
    with orm.sessionmaker(engine)() as session:
        result = (
            session.query(model_class)
            .where(
                model_class.session_id == user_id
            )
            .order_by(model_class.create_by.desc())
        )
    return json.loads(result)

class SessionManager:
    def __init__(self, session_state, host, port, username, password,
                 db='chat', sess_table='sessions', msg_table='chat_memory') -> None:
        conn_str = f'clickhouse://{username}:{password}@{host}:{port}/{db}?protocol=https'
        self.engine = create_engine(conn_str, echo=False)
        self.sess_model_class = create_session_table(sess_table, declarative_base())
        self.sess_model_class.metadata.create_all(self.engine)
        self.msg_model_class = create_message_model(msg_table, declarative_base())
        self.msg_model_class.metadata.create_all(self.engine)
        self.Session = orm.sessionmaker(self.engine)
        self.session_state = session_state

    def list_sessions(self, user_id):
        with self.Session() as session:
            result = (
                session.query(self.sess_model_class)
                .where(
                    self.sess_model_class.user_id == user_id
                )
                .order_by(self.sess_model_class.create_by.desc())
            )
            sessions = []
            for r in result:
                sessions.append({
                    "session_id": r.session_id.split("?")[-1],
                    "system_prompt": r.system_prompt,
                    })
            return sessions
    
    def modify_system_prompt(self, session_id, sys_prompt):
        with self.Session() as session:
            session.update(self.sess_model_class).where(self.sess_model_class==session_id).value(system_prompt=sys_prompt)
            session.commit()
    
    def add_session(self, user_id, session_id, system_prompt, **kwargs):
        with self.Session() as session:
            elem = self.sess_model_class(
                user_id=user_id, session_id=session_id, system_prompt=system_prompt,
                create_by=datetime.now(), additionals=json.dumps(kwargs)
            )
            session.add(elem)
            session.commit()
    
    def remove_session(self, session_id):
        with self.Session() as session:
            session.query(self.sess_model_class).where(self.sess_model_class.session_id==session_id).delete()
            # session.query(self.msg_model_class).where(self.msg_model_class.session_id==session_id).delete()
        if "agent" in self.session_state:
            self.session_state.agent.memory.chat_memory.clear()
        if "file_analyzer" in self.session_state:
            self.session_state.file_analyzer.clear_files()

            