import json

from backend.chat_bot.tools import create_session_table, create_message_history_table
from backend.constants.variables import GLOBAL_CONFIG

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy import orm, create_engine
from logger import logger


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
    def __init__(
            self,
            session_state,
            host,
            port,
            username,
            password,
            db='chat',
            session_table='sessions',
            msg_table='chat_memory'
    ) -> None:
        if GLOBAL_CONFIG.mode == "dev":
            conn_str = f'clickhouse://{username}:{password}@{host}:{port}/{db}?protocol=http'
        else:
            conn_str = f'clickhouse://{username}:{password}@{host}:{port}/{db}?protocol=https'
        self.engine = create_engine(conn_str, echo=False)
        self.session_model_class = create_session_table(
            session_table, declarative_base())
        self.session_model_class.metadata.create_all(self.engine)
        self.msg_model_class = create_message_history_table(msg_table, declarative_base())
        self.msg_model_class.metadata.create_all(self.engine)
        self.session_orm = orm.sessionmaker(self.engine)
        self.session_state = session_state

    def list_sessions(self, user_id: str):
        with self.session_orm() as session:
            result = (
                session.query(self.session_model_class)
                .where(
                    self.session_model_class.user_id == user_id
                )
                .order_by(self.session_model_class.create_by.desc())
            )
            sessions = []
            for r in result:
                sessions.append({
                    "session_id": r.session_id.split("?")[-1],
                    "system_prompt": r.system_prompt,
                })
            return sessions

    # Update sys_prompt with given session_id
    def modify_system_prompt(self, session_id, sys_prompt):
        with self.session_orm() as session:
            obj = session.query(self.session_model_class).where(
                self.session_model_class.session_id == session_id).first()
            if obj:
                obj.system_prompt = sys_prompt
                session.commit()
            else:
                logger.warning(f"Session {session_id} not found")

    # Add a session(session_id, sys_prompt)
    def add_session(self, user_id: str, session_id: str, system_prompt: str, **kwargs):
        with self.session_orm() as session:
            elem = self.session_model_class(
                user_id=user_id, session_id=session_id, system_prompt=system_prompt,
                create_by=datetime.now(), additionals=json.dumps(kwargs)
            )
            session.add(elem)
            session.commit()

    # Remove a session and related chat history.
    def remove_session(self, session_id: str):
        with self.session_orm() as session:
            # remove session
            session.query(self.session_model_class).where(self.session_model_class.session_id == session_id).delete()
            # remove related chat history.
            session.query(self.msg_model_class).where(self.msg_model_class.session_id == session_id).delete()
