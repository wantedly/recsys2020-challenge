import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class CountEncodingHashtags(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id"
        ]

    def _read_hashtags_count_from_bigquery(
            self, train_table_name: str, test_table_name) -> pd.DataFrame:
        self._logger.info(f"Reading from {train_table_name} and {test_table_name}")
        query = """
                WITH subset AS (
                  (
                    SELECT tweet_id, any_value(hashtags) AS hashtags
                    FROM {}
                    GROUP BY tweet_id
                  )
                UNION ALL
                  (
                    SELECT tweet_id, any_value(hashtags) AS hashtags
                    FROM {}
                    GROUP BY tweet_id
                  )
                )
                , unnest_subset AS (
                SELECT tweet_id, hashtag
                FROM subset,
                unnest(hashtags) AS hashtag
                )
                , count_hashtag AS (
                SELECT hashtag, COUNT(*) AS cnt
                FROM unnest_subset
                GROUP BY hashtag
                )

                SELECT
                tweet_id,
                AVG(cnt) AS mean_value,
                min(cnt) AS min_value,
                max(cnt) AS max_value,
                case when stddev(cnt) is null then 1 else stddev(cnt) end AS std_value
                FROM (
                    SELECT A.tweet_id, A.hashtag, B.cnt
                    FROM unnest_subset AS A
                    LEFT OUTER JOIN count_hashtag AS B
                    ON A.hashtag = B.hashtag
                )
                GROUP BY
                tweet_id
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
        # read hashtags unnested hashtags
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
        count_hashtags = self._read_hashtags_count_from_bigquery(train_table, test_table)
        feature_names = ["mean_value", "max_value", "min_value", "std_value"]
        print(count_hashtags.shape)
        print(count_hashtags.isnull().sum())

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        df_train_input = pd.merge(df_train_input, count_hashtags, on="tweet_id", how="left").fillna(0)
        df_test_input = pd.merge(df_test_input, count_hashtags, on="tweet_id", how="left").fillna(0)

        for fe in feature_names:
            df_train_features[fe] = df_train_input[fe].values
            df_test_features[fe] = df_test_input[fe].values

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingHashtags.main()
