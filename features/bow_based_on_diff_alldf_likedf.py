import os
import pandas as pd
from base import BaseFeature
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class BOWBasedOnDiffAlldfLikedf(BaseFeature):
    def import_columns(self):
        return [
            "1"
       ]

    def _read_features_from_bigquery(self, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                select
                    countif(text_token=119) as n_text_token_119,
                    countif(text_token=188) as n_text_token_188,
                    countif(text_token=11170) as n_text_token_11170,
                    countif(text_token=14120) as n_text_token_14120,
                    countif(text_token=120) as n_text_token_120,
                    countif(text_token=146) as n_text_token_146,
                    countif(text_token=18628) as n_text_token_18628,
                    countif(text_token=10111) as n_text_token_10111,
                    countif(text_token=15221) as n_text_token_15221,
                    countif(text_token=1881) as n_text_token_1881,
                    countif(text_token=11662) as n_text_token_11662,
                from {}
                cross join unnest(text_tokens) as text_token
                group by tweet_id, engaging_user_id
                order by tweet_id, engaging_user_id
        """.format(read_table_name)
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
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
        df_train_features = self._read_features_from_bigquery(train_table)
        df_test_features = self._read_features_from_bigquery(test_table)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    BOWBasedOnDiffAlldfLikedf.main()
