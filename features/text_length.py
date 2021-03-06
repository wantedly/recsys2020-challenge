import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class TextLength(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, read_table_name: str, read_text_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH text_length AS (
                    SELECT
                        tweet_id,
                        length(regexp_replace(regexp_replace(text, r"https?://[w/:%#$&?()~.=+-…]+", ""), r'(@[a-zA-Z0-9_]{{1,15}})', "")) as removed_html_at
                    FROM {}
                )

                SELECT
                    B.removed_html_at
                FROM {} AS A
                LEFT OUTER JOIN text_length AS B
                ON A.tweet_id = B.tweet_id
                ORDER BY
                A.tweet_id, A.engaging_user_id
        """.format(
            read_text_name, read_table_name)
        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=self.PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def make_features(self, df_train_input, df_test_input):
        # read features
        df_train_features = self._read_features_from_bigquery(self.train_table, self.train_text)
        df_test_features = self._read_features_from_bigquery(self.test_table, self.test_text)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TextLength.main()
