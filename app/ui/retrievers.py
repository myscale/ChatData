import streamlit as st

from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.variables import CHAINS_RETRIEVERS_MAPPING, RetrieverButtons
from backend.retrievers.self_query import process_self_query
from backend.retrievers.vector_sql_query import process_sql_query
from logger import logger
from login import back_to_main


def render_retrievers():
    logger.info("render retrievers UI")
    selected_table = st.selectbox(
        label='Choose the table you want to query with:',
        options=MYSCALE_TABLES.keys()
    )
    MYSCALE_TABLES[selected_table].hint()
    tab_sql, tab_self_query = st.tabs(
        tabs=['Vector SQL', 'Self-Query Retrievers']
    )

    with tab_sql:
        render_tab_sql(selected_table)

    with tab_self_query:
        render_tab_self_query(selected_table)


def render_tab_sql(selected_table: str):
    MYSCALE_TABLES[selected_table].hint_sql()
    st.text_input("Ask a question:", key='query_sql')
    cols = st.columns([2, 2, 1, 4])
    cols[0].button("Search SQL", key=RetrieverButtons.vector_sql_query_from_db)
    cols[1].button("Ask SQL", key=RetrieverButtons.vector_sql_query_with_llm)
    cols[2].button("Back", key='back_sql', on_click=back_to_main)

    if st.session_state[RetrieverButtons.vector_sql_query_from_db]:
        process_sql_query(selected_table, RetrieverButtons.vector_sql_query_from_db)

    if st.session_state[RetrieverButtons.vector_sql_query_with_llm]:
        process_sql_query(selected_table, RetrieverButtons.vector_sql_query_with_llm)


def render_tab_self_query(selected_table):
    st.info(
        "You can retrieve papers with button `Query` or ask questions based on retrieved papers with button `Ask`.",
        icon='💡')
    st.dataframe(st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["metadata_columns"])
    st.text_input("Ask a question:", key='query_self')
    cols = st.columns([2, 2, 1, 4])
    cols[0].button("Query Self", key='search_self')
    cols[1].button("Ask Self", key='ask_self')
    cols[2].button("Back", key='back_self', on_click=back_to_main)

    if st.session_state.search_self:
        process_self_query(selected_table, RetrieverButtons.self_query_from_db)

    if st.session_state.ask_self:
        process_self_query(selected_table, RetrieverButtons.self_query_with_llm)

