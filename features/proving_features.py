import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class Proving(BaseFeature):
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
                    SELECT A.tweet_id, any_value(B.text_id) as text_id, any_value(A.engaged_user_id) as engaged_user_id
                    FROM {} AS A
                    LEFT OUTER JOIN {} AS B
                    ON A.tweet_id = B.tweet_id
                    GROUP BY A.tweet_id
                    )
                    UNION ALL
                    (
                    SELECT A.tweet_id, any_value(B.text_id) as text_id, any_value(A.engaged_user_id) as engaged_user_id
                    FROM {} AS A
                    LEFT OUTER JOIN {} AS B
                    ON A.tweet_id = B.tweet_id
                    GROUP BY A.tweet_id
                    )
                )

                SELECT
                countif(C.engaged_user_id IS NOT NULL) AS proving
                FROM {} AS A
                LEFT OUTER JOIN {} AS B
                ON A.tweet_id = B.tweet_id
                LEFT OUTER JOIN full_data AS C
                ON B.text_id = C.text_id
                AND A.engaging_user_id = C.engaged_user_id
                GROUP BY A.tweet_id, A.engaging_user_id
                ORDER BY
                A.tweet_id, A.engaging_user_id
        """.format(
            train_table, train_text, test_table, test_text,
            read_table_name, read_text_name)
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
            self.train_text, self.test_text,
            self.train_table, self.test_table,
            self.train_table, self.train_text
        )
        df_test_features = self._read_features_from_bigquery(
            self.train_text, self.test_text,
            self.train_table, self.test_table,
            self.test_table, self.test_text
        )
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    Proving.main()
