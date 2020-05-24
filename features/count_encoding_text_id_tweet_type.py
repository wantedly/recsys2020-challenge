import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class CountEncodingTextIdTweetType(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_text: str, test_text: str,
            train_table: str, test_table: str,
            read_table_name: str, read_text_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH full_data AS (
                    (
                    SELECT A.tweet_id, engaged_user_id, engaging_user_id, text_id, tweet_type
                    FROM {} AS A
                    LEFT OUTER JOIN {} AS B
                    ON A.tweet_id = B.tweet_id
                    )
                    UNION ALL
                    (
                    SELECT A.tweet_id, engaged_user_id, engaging_user_id, text_id, tweet_type
                    FROM {} AS A
                    LEFT OUTER JOIN {} AS B
                    ON A.tweet_id = B.tweet_id
                    )
                ),
                agg as (
                    select
                      text_id, tweet_type,
                      sum(n0_text_id_tweet_type) as n0_text_id_tweet_type, -- 同じ tweet_id のものも数える
                      count(1) as n1_text_id_tweet_type,  -- 異なる tweet_id のもののみ数える
                    from
                      (
                      select
                        tweet_id, text_id, tweet_type, count(1) as n0_text_id_tweet_type
                      from
                        full_data
                      group by
                        tweet_id, text_id, tweet_type
                      )
                    group by
                      text_id, tweet_type
                )

                SELECT
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  n0_text_id_tweet_type,
                  n1_text_id_tweet_type,
                FROM {} AS A
                LEFT OUTER JOIN {} AS B
                ON A.tweet_id = B.tweet_id
                LEFT OUTER JOIN agg AS C
                ON B.text_id = C.text_id
                AND A.tweet_type = C.tweet_type
                ORDER BY
                  A.tweet_id, A.engaging_user_id
        """.format(
            train_table, train_text, test_table, test_text,
            read_table_name, read_text_name)
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
        # read features
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        train_text = f"`{PROJECT_ID}.recsys2020.texts_training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
            test_text = f"`{PROJECT_ID}.recsys2020.texts_test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
            test_text = f"`{PROJECT_ID}.recsys2020.texts_val_20200418`"

        df_train_features = self._read_features_from_bigquery(train_text, test_text, train_table, test_table, train_table, train_text)
        df_test_features = self._read_features_from_bigquery(train_text, test_text, train_table, test_table, test_table, test_text)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingTextIdTweetType.main()
