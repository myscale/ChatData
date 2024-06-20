import streamlit as st


def display(dataframe, columns_=None, index=None):
    if len(dataframe) > 0:
        if index:
            dataframe.set_index(index)
        if columns_:
            st.dataframe(dataframe[columns_])
        else:
            st.dataframe(dataframe)
    else:
        st.write(
            "Sorry 😵 we didn't find any articles related to your query.\n\n"
            "Maybe the LLM is too naughty that does not follow our instruction... \n\n"
            "Please try again and use verbs that may match the datatype.",
            unsafe_allow_html=True
        )
