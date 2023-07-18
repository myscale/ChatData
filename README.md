# ChatData 🔍 📖
***We are constantly improving LangChain's self-query retriever. Some of the features are not merged yet.***

[![](https://dcbadge.vercel.app/api/server/D2qpkqc4Jq?compact=true&style=flat)](https://discord.gg/D2qpkqc4Jq)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/myscaledb.svg?style=social&label=Follow%20%40MyScaleDB)](https://twitter.com/myscaledb)
<a href="https://huggingface.co/spaces/myscale/ChatData"  style="padding-left: 0.5rem;"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange"></a>

<br>
<div style="text-align: center">
<img src="assets/logo.png" width=60%>
</div>

Yet another chat-with-documents app, but supporting query over millions of files with [MyScale](https://myscale.com) and [LangChain](https://github.com/hwchase17/langchain/).

## News 🔥 (July-2023)

- 🤖 LLMs are now capable of writing **Vector SQL** - a extended SQL with vector search! Vector SQL allows you to **access MyScale faster and stronger**! This will **be added to LangChain** soon! ([PR 7454](https://github.com/hwchase17/langchain/pull/7454))
- 🌏 Customized Retrieval QA Chain that gives you **more information** on each PDF and **answer question in your native language**!
- 🔧 Our contribution to LangChain that helps self-query retrievers [**filter with more types and functions**](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/self_query/myscale_self_query)
- 🌟 **We just opened a FREE pod hosting data for ArXiv paper.** Anyone can try their own SQL with vector search!!! Feel the power when SQL meets vector search! See how to access the pod [here](#data-service).
- 📚 We collected about **2 million papers on arxiv**! We are collecting more and we need your advice!
- More coming...

## Quickstart

1. Create an virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

> This app is currently using [MyScale's fork of LangChain](https://github.com/myscale/langchain/tree/master). 
>> 
>> It contains improved SQLDatabaseChain in [this PR](https://github.com/hwchase17/langchain/pull/7454)
>> 
>> It contains [improved prompts](https://github.com/hwchase17/langchain/pull/6737#discussion_r1243527112) for comparators `LIKE` and `CONTAIN` in [MyScale self-query retriever](https://github.com/hwchase17/langchain/pull/6143).

```bash
python3 -m pip install -r requirements.txt
```

3. Run the app!

```python
# fill you OpenAI key in .streamlit/secrets.toml
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# start the app
python3 -m streamlit run app.py
```

## Quick Navigator 🧭

- [How can I run this app?](README.md#how-to-run)

- [How this app is built?](docs/self-query.md)

- [What is the overview pipeline?](docs/self-query.md#query-pipeline-design)

- [How did LangChain and MyScale convert natural language to structured filters?](docs/self-query.md#selfqueryretriever-defines-interaction-between-vectorstore-and-your-app)

- [How to make chain execution more responsive in LangChain?](docs/self-query.md#not-responsive-add-callbacks)

- Where can I get those arxiv data?
  - [From parquet files on S3](docs/self-query.md#insert-data)
  - <a name="data-service"></a>Or Directly use MyScale database as service... for **FREE** ✨
    ```python
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host='msc-1decbcc9.us-east-1.aws.staging.myscale.cloud',
        port=443,
        username='chatdata',
        password='myscale_rocks'
    )
    ```


## Introduction

Database credentials:

```toml
MYSCALE_HOST = "msc-1decbcc9.us-east-1.aws.staging.myscale.cloud"
MYSCALE_PORT = 443
MYSCALE_USER = "chatdata"
MYSCALE_PASSWORD = "myscale_rocks"
```

ChatData brings millions of papers into your knowledge base. We imported 2.2 million papers with metadata info, which contains:

1. `id`: paper's arxiv id
2. `abstract`: paper's abstracts used as ranking criterion (with InstructXL)
3. `vector`: column that contains the vector array in `Array(Float32)`
4. `metadata`: LangChain VectorStore Compatible Columns
    1. `metadata.authors`: paper's authors in *list of strings*
    2. `metadata.abstract`: paper's abstracts used as ranking criterion (with InstructXL)
    3. `metadata.titles`: papers's titles
    4. `metadata.categories`: paper's categories in *list of strings* like ["cs.CV"]
    5. `metadata.pubdate`: paper's date of publication in *ISO 8601 formated strings*
    6. `metadata.primary_category`: paper's primary category in *strings* defined by ArXiv
    7. `metadata.comment`: some additional comment to the paper
  
*Columns below are native columns in MyScale and can only be used as SQLDatabase*

5. `authors`: paper's authors in *list of strings*
6. `titles`: papers's titles
7. `categories`: paper's categories in *list of strings* like ["cs.CV"]
8. `pubdate`: paper's date of publication in *Date32 data type* (faster)
9. `primary_category`: paper's primary category in *strings* defined by ArXiv
10. `comment`: some additional comment to the paper

And for overall table schema, please refer to [table creation section in docs/self-query.md](docs/self-query.md#table-creation).

If you want to use this database with `langchain.chains.sql_database.base.SQLDatabaseChain` or `langchain.retrievers.SQLDatabaseRetriever`, please follow guides on [data preparation section](docs/vector-sql.md#prepare-the-database) and [chain creation section](docs/vector-sql.md#create-the-sqldatabasechain) in docs/vector-sql.md

## How to run 🏃
<a name="how-to-run"></a>

```bash
python3 -m pip install requirements.txt
python3 -m streamlit run app.py
```

## How to build from scratch? 🧱

- with LangChain SQLDatabaseRetrievers: See [docs/vector-sql.md](docs/vector-sql.md)
- with LangChain Self-Query Retrievers: See [docs/self-query.md](docs/self-query.md)

## Special Thanks 👏 (Ordered Alphabetically)

- [ArXiv API](https://info.arxiv.org/help/api/index.html) for its open access interoperability to pre-printed papers.
- [InstructorXL](https://huggingface.co/hkunlp/instructor-xl) for its promptable embeddings that improves retrieve performance.
- [LangChain🦜️🔗](https://github.com/hwchase17/langchain/) for its easy-to-use and composable API designs and prompts.
- [OpenChatPaper](https://github.com/liuyixin-louis/OpenChatPaper) for prompt design reference.
- [The Alexandria Index](https://alex.macrocosm.so/download) for providing arXiv data index to the public.
