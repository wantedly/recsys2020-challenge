import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class TweetTime(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id",
            "language",
            "EXTRACT(HOUR FROM TIMESTAMP_SECONDS(Timestamp)) AS hour"
        ]

    def _read_activity_time_from_bigquery(
            self, train_table_name: str, test_table_name) -> pd.DataFrame:
        self._logger.info(f"Reading from {train_table_name} and {test_table_name}")
        query = """
                WITH subset AS (
                  (
                    SELECT tweet_id, any_value(language) AS language, any_value(Timestamp) AS Timestamp
                    FROM {}
                    GROUP BY tweet_id
                  )
                UNION ALL
                  (
                    SELECT tweet_id, any_value(language) AS language, any_value(Timestamp) AS Timestamp
                    FROM {}
                    GROUP BY tweet_id
                  )
                ), agg AS (
                SELECT language, hour, count(*) AS cnt
                FROM (
                    SELECT
                    tweet_id,
                    language,
                    EXTRACT(HOUR FROM TIMESTAMP_SECONDS(Timestamp)) AS hour
                    FROM
                    subset
                )
                GROUP BY language, hour
                )
                SELECT
                A.language,
                A.hour,
                A.cnt / B.total_cnt AS activity_ratio
                FROM agg AS A
                INNER JOIN (
                SELECT language, sum(cnt) as total_cnt
                FROM agg
                GROUP BY language
                ) AS B
                ON A.language = B.language
        """.format(train_table_name, test_table_name)
        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def make_features(self, df_train_input, df_test_input):
        # read tweet time
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val`"
        tweet_time = self._read_activity_time_from_bigquery(train_table, test_table)
        feature_names = ["activity_ratio"]
        print(tweet_time.shape)
        print(tweet_time.isnull().sum())

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        df_train_input = pd.merge(df_train_input, tweet_time, on=["language", "hour"], how="left")
        df_test_input = pd.merge(df_test_input, tweet_time, on=["language", "hour"], how="left")

        for fe in feature_names:
            df_train_features[fe] = df_train_input[fe].values
            df_test_features[fe] = df_test_input[fe].values

        print(df_train_features.shape)
        print(df_train_features.isnull().sum())
        print(df_test_features.shape)
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TweetTime.main()
