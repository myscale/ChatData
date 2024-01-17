import streamlit as st
from auth0_component import login_button

AUTH0_CLIENT_ID = st.secrets['AUTH0_CLIENT_ID']
AUTH0_DOMAIN = st.secrets['AUTH0_DOMAIN']


def login():
    if "user_name" in st.session_state or ("jump_query_ask" in st.session_state and st.session_state.jump_query_ask):
        return True
    st.subheader(
        "🤗 Welcom to [MyScale](https://myscale.com)'s [ChatData](https://github.com/myscale/ChatData)! 🤗 ")
    st.write("You can now chat with ArXiv and Wikipedia! 🌟\n")
    st.write("Built purely with streamlit 👑 , LangChain 🦜🔗 and love ❤️ for AI!")
    st.write(
        "Follow us on [Twitter](https://x.com/myscaledb) and [Discord](https://discord.gg/D2qpkqc4Jq)!")
    st.write(
        "For more details, please refer to [our repository on GitHub](https://github.com/myscale/ChatData)!")
    st.divider()
    col1, col2 = st.columns(2, gap='large')
    with col1.container():
        st.write("Try out MyScale's Self-query and Vector SQL retrievers!")
        st.write("In this demo, you will be able to see how those retrievers "
                 "**digest** -> **translate** -> **retrieve** -> **answer** to your question!")
        st.session_state["jump_query_ask"] = st.button("Query / Ask")
    with col2.container():
        # st.warning("To use chat, please jump to [https://myscale-chatdata.hf.space](https://myscale-chatdata.hf.space)")
        st.write("Now with the power of LangChain's Conversantional Agents, we are able to build "
                 "an RAG-enabled chatbot within one MyScale instance! ")
        st.write("Log in to Chat with RAG!")
        login_button(AUTH0_CLIENT_ID, AUTH0_DOMAIN, "auth0")
    st.divider()
    st.write("- [Privacy Policy](https://myscale.com/privacy/)\n"
             "- [Terms of Sevice](https://myscale.com/terms/)")
    if st.session_state.auth0 is not None:
        st.session_state.user_info = dict(st.session_state.auth0)
        if 'email' in st.session_state.user_info:
            email = st.session_state.user_info["email"]
        else:
            email = f"{st.session_state.user_info['nickname']}@{st.session_state.user_info['sub']}"
        st.session_state["user_name"] = email
        del st.session_state.auth0
        st.experimental_rerun()
    if st.session_state.jump_query_ask:
        st.experimental_rerun()


def back_to_main():
    if "user_info" in st.session_state:
        del st.session_state.user_info
    if "user_name" in st.session_state:
        del st.session_state.user_name
    if "jump_query_ask" in st.session_state:
        del st.session_state.jump_query_ask
