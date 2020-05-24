import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class TextTypeCountOfText(BaseFeature):
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
                    SELECT text_id, tweet_type
                    FROM {} AS A
                    LEFT OUTER JOIN {} AS B
                    ON A.tweet_id = B.tweet_id
                    )
                    UNION ALL
                    (
                    SELECT text_id, tweet_type
                    FROM {} AS A
                    LEFT OUTER JOIN {} AS B
                    ON A.tweet_id = B.tweet_id
                    )
                ),
                agg as (
                    select
                      text_id,
                      n_rt, n_tl, n_q,
                      n_rt / (n_rt + n_tl + n_q + 1e-2) n_rt_rate,
                      n_tl / (n_rt + n_tl + n_q + 1e-2) n_tl_rate,
                      n_q / (n_rt + n_tl + n_q + 1e-2) n_q_rate,
                    from
                    (
                    select
                      text_id,
                      countif(tweet_type = "Retweet") n_rt,
                      countif(tweet_type = "TopLevel") n_tl,
                      countif(tweet_type = "Quote") n_q,
                    from
                      full_data
                    group by
                      text_id
                    )
                )

                SELECT
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  C.* except(text_id)
                FROM {} AS A
                LEFT OUTER JOIN {} AS B
                ON A.tweet_id = B.tweet_id
                LEFT OUTER JOIN agg AS C
                ON B.text_id = C.text_id
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
    TextTypeCountOfText.main()
