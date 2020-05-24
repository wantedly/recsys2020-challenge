import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEncodingTextId(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_text: str, test_text: str,
            read_table_name: str, read_text_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH agg_text AS (
                SELECT text_id, count(*) as n_text
                FROM (
                    (SELECT text_id FROM {})
                    UNION ALL
                    (SELECT text_id FROM {})
                )
                GROUP BY text_id
                )

                SELECT
                C.n_text 
                FROM {} AS A
                LEFT OUTER JOIN {} AS B
                ON A.tweet_id = B.tweet_id
                LEFT OUTER JOIN agg_text AS C
                ON B.text_id = C.text_id
                ORDER BY
                A.tweet_id, A.engaging_user_id
        """.format(
            train_text, test_text, read_table_name, read_text_name)
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
        df_train_features = self._read_features_from_bigquery(
            self.train_text, self.test_text, self.train_table, self.train_text
        )
        df_test_features = self._read_features_from_bigquery(
            self.train_text, self.test_text, self.test_table, self.test_text
        )
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingTextId.main()
