import os
import pandas as pd
from base import BaseFeature
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class SentenceCount(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, read_table_name: str, read_text_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH sentence_count AS (
                    SELECT
                        tweet_id,
                        array_length(split(text, "。")) sen_cnt_0,
                        array_length(split(regexp_replace(text, r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+", ""), ".")) sen_cnt_1,
                    FROM {}
                )
                SELECT
                    B.sen_cnt_0,
                    B.sen_cnt_1,
                FROM {} AS A
                LEFT OUTER JOIN sentence_count AS B
                ON A.tweet_id = B.tweet_id
                ORDER BY
                A.tweet_id, A.engaging_user_id
        """.format(
            read_text_name, read_table_name)
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

        df_train_features = self._read_features_from_bigquery(train_table, train_text)
        df_test_features = self._read_features_from_bigquery(test_table, test_text)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    SentenceCount.main()
