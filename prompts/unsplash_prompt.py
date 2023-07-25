_DEFAULT_TEMPLATE = """Given an input question, create a syntactically correct MyScale query to run. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. 

MyScale queries has a distance function called `DISTANCE(column, array)` to compute relevance to an entity's array and sort an array column by the relevance. When the query is asking for {top_k} closest row, you have to use this distance function to calculate distance to entity's array on vector column and order by the distance to retrieve relevant rows.

NOTICE: `DISTANCE(column, array)` only accept an array column as its first argument and a `NeuralArray(entity)` as its second argument.

You also need a user defined function called `NeuralArray(entity)` to retrieve the entity's array. 


`ORDER BY` clause should always be after `WHERE` clause.

You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for the few relevant columns given the question.

Never query for all the tables from the database, only ask for a few relevant tables given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table. 

DO NOT add semicolon to the end of SQL.

You MUST specify the name of the table when selecting each column names.


Use the following format:

======== table info ========
<some table infos>
Question: "Question here"
SQLQuery: "SQL Query to run"


Here are some examples:


======== table info ========
CREATE TABLE IF NOT EXISTS flowers (
        id String,
        url String,
        vector Array(Float32)
        CONSTRAINT cons_vec_len CHECK length(vector) = 512
) ENGINE = MergeTree ORDER BY id

CREATE TABLE IF NOT EXISTS flowers_attribute (
        id String,
        color String
) ENGINE = MergeTree ORDER BY id

Question: what is the top 5 closet flower's photos to daisy whose color is yellow?
SQLQuery: SELECT flowers.url, flowers.id FROM flowers JOIN flowers_attribute ON flowers.id = flowers_attribute.id WHERE flowers_attribute.color = 'yellow' ORDER BY DISTANCE(flowers.vector, NeuralArray(daisy)) LIMIT 5


======== table info ========
CREATE TABLE painting_infos(
        p_id String, 
        artist_username String, 
        artist_location_country String, 
        artist_location_city String
) ENGINE = MergeTree ORDER BY p_id

CREATE TABLE paintings (
        p_id String, 
        p_url String, 
        is_featured Bool,
        p_feature Array(Float32)
) ENGINE = MergeTree() ORDER BY p_id

CREATE TABLE painting_conversions (
	converted_at DateTime, 
	conversion_type String, 
	keyword String, 
	paint_id String, 
	anonymous_user_id String, 
	conversion_country String
) ENGINE = MergeTree()
 ORDER BY painting_id
 PRIMARY KEY painting_id
Question: what is the featured painting that has the most downloads which was taken by mone?
SQLQuery: SELECT paintings.p_id, temp.downloads, paintings.p_url FROM paintings JOIN (SELECT painting_conversions.paint_id as paint_id, COUNT(*) as downloads FROM painting_conversions JOIN paintings ON painting_conversions.paint_id = paintings.p_id WHERE conversion_type = 'download' GROUP BY painting_conversions.paint_id) temp ON temp.paint_id = paintings.p_id JOIN painting_infos ON painting_infos.p_id = paintings.p_id WHERE artist_username = 'mone' and paintings.is_featured = True ORDER BY temp.downloads DESC LIMIT {top_k}


======== table info ========
CREATE TABLE IF NOT EXISTS movies (
        title String,
        country String,
        tiltle_embedding Array(Float32) 
        CONSTRAINT emb_vec_len CHECK length(vector) = 768
) ENGINE = MergeTree ORDER BY title
Question: which german movie's is the most similar to Star Wars?
SQLQuery: SELECT movies.title FROM movies WHERE country = 'german' ORDER BY DISTANCE(movies.title_embedding, NeuralArray(Star Wars)) LIMIT {top_k}


======== table info ========
CREATE TABLE painting_infos(
        id String, 
        artist_username String, 
        artist_location_country String, 
        artist_location_city String
) ENGINE = MergeTree ORDER BY painting_id

CREATE TABLE paintings (
        p_id String, 
        p_url String, 
        p_feature Array(Float32)
) ENGINE = MergeTree() ORDER BY p_id

CREATE TABLE painting_conversions (
	converted_at DateTime, 
	conversion_type String, 
	keyword String, 
	paint_id String, 
	anonymous_user_id String, 
	conversion_country String
) ENGINE = MergeTree()
 ORDER BY painting_id
 PRIMARY KEY painting_id
Question: what are the most popular paintings that have beautiful women?
SQLQuery: SELECT temp1.p_url, temp2.downloads FROM (SELECT paintings.p_url, paintings.p_id FROM paintings ORDER BY DISTANCE(paintings.p_feature, NeuralArray(beautiful women))  LIMIT {top_k}) temp1 JOIN (SELECT painting_conversions.p_id, COUNT(*) as downloads FROM painting_conversions GROUP BY painting_conversions.p_id) temp2 ON temp1.p_id = temp2.p_id  ORDER BY temp2.downloads DESC


======== table info ========
CREATE TABLE paintings (
        p_id String, 
        p_url String, 
        p_feature Array(Float32)
) ENGINE = MergeTree() ORDER BY p_id
Question: give me some paintings contains kite.
SQLQuery: SELECT paintings.p_url, paintings.p_id FROM photos ORDER BY DISTANCE(paintings.p_feature, NeuralArray(kite)) LIMIT {top_k}

Let's begin!


======== table info ========
{table_info}
Question: {input}"""
