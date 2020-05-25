from typing import List

import google.cloud.bigquery as bigquery
from google.cloud.exceptions import NotFound


_USER_TWEETS_VECTORS_TABLE_NAME_BASE = "recsys2020.user_tweets_vectors"
_USER_SURFACING_TWEETS_VECTORS_TABLE_NAME_BASE = (
    "recsys2020.user_surfacing_tweets_vectors"
)


def _get_or_initialize_table(project_id: str, table_name: str, query: str) -> str:
    client = bigquery.Client(project=project_id)
    try:
        client.get_table(table_name)
        return table_name
    except NotFound:
        pass
    job_config = bigquery.QueryJobConfig(destination=table_name)
    query_job = client.query(query, job_config=job_config)
    query_job.result()
    return table_name


def get_or_initialize_user_tweets_vectors_table(
    project_id: str, testing: bool, input_tables: List[str]
) -> str:
    table_name = f"{project_id}.{_USER_TWEETS_VECTORS_TABLE_NAME_BASE}"
    if testing:
        table_name += "_testing"
    query = "with unique_tweets as (\n"
    query += "  union distinct\n".join(
        (
            f"  select tweet_id, engaged_user_id from {input_table} group by tweet_id, engaged_user_id"
            for input_table in input_tables
        )
    )
    query += "\n)\n"
    query += "select engaged_user_id as user_id,\n"
    for i in range(768):
        query += f"avg(gap_{i}) as gap_{i},\n"
    query += "from unique_tweets\n"
    query += "inner join `recsys2020.pretrained_bert_gap` gap on unique_tweets.tweet_id = gap.tweet_id\n"
    query += "group by user_id"
    return _get_or_initialize_table(project_id, table_name, query)


def get_or_initialize_user_surfacing_tweets_vectors_table(
    project_id: str, testing: bool, input_tables: List[str]
) -> str:
    table_name = f"{project_id}.{_USER_SURFACING_TWEETS_VECTORS_TABLE_NAME_BASE}"
    if testing:
        table_name += "_testing"
    query = "with surfacing_tweets as (\n"
    query += "  union distinct\n".join(
        (
            f"  select tweet_id, engaging_user_id from {input_table} group by tweet_id, engaging_user_id"
            for input_table in input_tables
        )
    )
    query += "\n)\n"
    query += "select engaging_user_id as user_id,\n"
    for i in range(768):
        query += f"avg(gap_{i}) as gap_{i},\n"
    query += "from surfacing_tweets\n"
    query += "inner join `recsys2020.pretrained_bert_gap` gap on surfacing_tweets.tweet_id = gap.tweet_id\n"
    query += "group by user_id"
    return _get_or_initialize_table(project_id, table_name, query)


def get_similarity_between_tweet_and_engaging_user_using_tweets_vector(
    project_id: str, testing: bool, input_tables: List[str], target_table: str
) -> str:
    output_column_name = "dot_product_of_engaged_tweet_and_engaging_user"
    tweet_vector_table = "`recsys2020.pretrained_bert_gap`"
    user_feature_table = get_or_initialize_user_tweets_vectors_table(
        project_id, testing, input_tables
    )

    query = "select (\n"
    query += " + ".join((f"f1.gap_{i} * f2.gap_{i}" for i in range(768)))
    query += f") as {output_column_name}\n"
    query += f"from {target_table} as t\n"
    query += f"left join {tweet_vector_table} as f1 on t.tweet_id = f1.tweet_id\n"
    query += (
        f"left join {user_feature_table} as f2 on t.engaging_user_id = f2.user_id\n"
    )
    query += "order by t.tweet_id, t.engaging_user_id"

    return query


def get_similarity_between_engaged_and_engaging_user_using_tweets_vector(
    project_id: str, testing: bool, input_tables: List[str], target_table: str
) -> str:
    output_column_name = "dot_product_of_engaging_user_and_engaged_user"
    user_feature_table = get_or_initialize_user_tweets_vectors_table(
        project_id, testing, input_tables
    )

    query = "select (\n"
    query += " + ".join((f"f1.gap_{i} * f2.gap_{i}" for i in range(768)))
    query += f") as {output_column_name}\n"
    query += f"from {target_table} as t\n"
    query += f"left join {user_feature_table} as f1 on t.engaged_user_id = f1.user_id\n"
    query += (
        f"left join {user_feature_table} as f2 on t.engaging_user_id = f2.user_id\n"
    )
    query += "order by t.tweet_id, t.engaging_user_id"

    return query


def get_similarity_between_tweet_and_engaging_user_using_surfacing_tweets_vector(
    project_id: str, testing: bool, input_tables: List[str], target_table: str
) -> str:
    output_column_name = (
        "dot_product_of_engaged_tweet_and_engaging_user_surfacing_tweets"
    )
    tweet_vector_table = "`recsys2020.pretrained_bert_gap`"
    user_feature_table = get_or_initialize_user_surfacing_tweets_vectors_table(
        project_id, testing, input_tables
    )

    query = "select (\n"
    query += " + ".join((f"f1.gap_{i} * f2.gap_{i}" for i in range(768)))
    query += f") as {output_column_name}\n"
    query += f"from {target_table} as t\n"
    query += f"left join {tweet_vector_table} as f1 on t.tweet_id = f1.tweet_id\n"
    query += (
        f"left join {user_feature_table} as f2 on t.engaging_user_id = f2.user_id\n"
    )
    query += "order by t.tweet_id, t.engaging_user_id"

    return query


def get_similarity_between_engaged_and_engaging_user_using_surfacing_tweets_vector(
    project_id: str, testing: bool, input_tables: List[str], target_table: str
) -> str:
    output_column_name = (
        "dot_product_of_engaged_tweet_and_engaging_user_surfacing_tweets"
    )
    user_feature_table = get_or_initialize_user_surfacing_tweets_vectors_table(
        project_id, testing, input_tables
    )

    query = "select (\n"
    query += " + ".join((f"f1.gap_{i} * f2.gap_{i}" for i in range(768)))
    query += f") as {output_column_name}\n"
    query += f"from {target_table} as t\n"
    query += f"left join {user_feature_table} as f1 on t.engaged_user_id = f1.user_id\n"
    query += (
        f"left join {user_feature_table} as f2 on t.engaging_user_id = f2.user_id\n"
    )
    query += "order by t.tweet_id, t.engaging_user_id"

    return query
