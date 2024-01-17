from sqlalchemy import Column, Text
from clickhouse_sqlalchemy import types, engines


def create_message_model(table_name, DynamicBase):  # type: ignore
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model decleared inside a function to have a dynamic table name
    class Message(DynamicBase):
        __tablename__ = table_name
        id = Column(types.Float64)
        session_id = Column(Text)
        user_id = Column(Text)
        msg_id = Column(Text, primary_key=True)
        type = Column(Text)
        addtionals = Column(Text)
        message = Column(Text)
        __table_args__ = (
            engines.ReplacingMergeTree(
                partition_by='session_id',
                order_by=('id', 'msg_id')),
            {'comment': 'Store Chat History'}
        )

    return Message


def create_session_table(table_name, DynamicBase):  # type: ignore
    # Model decleared inside a function to have a dynamic table name
    class Session(DynamicBase):
        __tablename__ = table_name
        user_id = Column(Text)
        session_id = Column(Text, primary_key=True)
        system_prompt = Column(Text)
        create_by = Column(types.DateTime)
        additionals = Column(Text)
        __table_args__ = (
            engines.ReplacingMergeTree(
                order_by=('session_id')),
            {'comment': 'Store Session and Prompts'}
        )
    return Session
