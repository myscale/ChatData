import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from backend.constants.myscale_tables import MYSCALE_TABLES
from backend.constants.variables import CHAINS_RETRIEVERS_MAPPING, RetrieverButtons
from backend.retrievers.self_query import process_self_query
from backend.retrievers.vector_sql_query import process_sql_query
from backend.constants.variables import JUMP_QUERY_ASK, USER_NAME, USER_INFO


def back_to_main():
    if USER_INFO in st.session_state:
        del st.session_state[USER_INFO]
    if USER_NAME in st.session_state:
        del st.session_state[USER_NAME]
    if JUMP_QUERY_ASK in st.session_state:
        del st.session_state[JUMP_QUERY_ASK]


def _render_table_selector() -> str:
    col1, col2 = st.columns(2)
    with col1:
        selected_table = st.selectbox(
            label='Each public knowledge base is stored in a MyScaleDB table, which is read-only.',
            options=MYSCALE_TABLES.keys(),
        )
        MYSCALE_TABLES[selected_table].hint()
    with col2:
        add_vertical_space(1)
        st.info(f"Here is your selected public knowledge base schema in MyScaleDB",
                icon='📚')
        MYSCALE_TABLES[selected_table].hint_sql()

    return selected_table


def render_retrievers():
    st.button("⬅️ Back", key="back_sql", on_click=back_to_main)
    st.subheader('Please choose a public knowledge base to search.')
    selected_table = _render_table_selector()

    tab_sql, tab_self_query = st.tabs(
        tabs=['Vector Search', 'SelfQuery Retrievers']
    )

    with tab_sql:
        render_tab_sql(selected_table)

    with tab_self_query:
        render_tab_self_query(selected_table)


def render_tab_sql(selected_table: str):
    st.warning(
        "When you input a query with filtering conditions, you need to ensure that your filters are applied only to "
        "the metadata we provide. This table allows filters to be established on the following metadata fields:",
        icon="⚠️")
    st.dataframe(st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["metadata_columns"])

    cols = st.columns([8, 3, 3, 2])
    cols[0].text_input("Input your question:", key='query_sql')
    with cols[1].container():
        add_vertical_space(2)
        st.button("VectorSearch from MyScaleDB ➡️", key=RetrieverButtons.vector_sql_query_from_db)
    with cols[2].container():
        add_vertical_space(2)
        st.button("VectorSearch with Sources(LLM) ➡️", key=RetrieverButtons.vector_sql_query_with_llm)

    if st.session_state[RetrieverButtons.vector_sql_query_from_db]:
        process_sql_query(selected_table, RetrieverButtons.vector_sql_query_from_db)

    if st.session_state[RetrieverButtons.vector_sql_query_with_llm]:
        process_sql_query(selected_table, RetrieverButtons.vector_sql_query_with_llm)


def render_tab_self_query(selected_table):
    st.warning(
        "When you input a query with filtering conditions, you need to ensure that your filters are applied only to "
        "the metadata we provide. This table allows filters to be established on the following metadata fields:",
        icon="⚠️")
    st.dataframe(st.session_state[CHAINS_RETRIEVERS_MAPPING][selected_table]["metadata_columns"])

    cols = st.columns([8, 3, 3, 2])
    cols[0].text_input("Input your question:", key='query_self')

    with cols[1].container():
        add_vertical_space(2)
        st.button("SelfQuery from MyScaleDB ➡️", key='search_self')
    with cols[2].container():
        add_vertical_space(2)
        st.button("SelfQuery with Sources(LLM) ➡️", key='ask_self')

    if st.session_state.search_self:
        process_self_query(selected_table, RetrieverButtons.self_query_from_db)

    if st.session_state.ask_self:
        process_self_query(selected_table, RetrieverButtons.self_query_with_llm)
