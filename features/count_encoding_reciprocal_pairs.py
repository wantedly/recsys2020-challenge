import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEncodingReciprocalPair(BaseFeature):
    def import_columns(self):
        return [
            "1"
        ]

    def _read_features_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH subset AS (
                SELECT
                    engaged_user_id,
                    engaging_user_id,
                    count(1) as n_pair,
                FROM (
                    (
                        SELECT engaged_user_id, engaging_user_id
                        FROM {}
                    )
                    UNION ALL
                    (
                        SELECT engaged_user_id, engaging_user_id
                        FROM {}
                    )
                )
                GROUP BY
                    engaged_user_id, engaging_user_id
                ),
                agg AS (
                    select
                      org.engaged_user_id,
                      org.engaging_user_id,
                      IFNULL(org.n_pair + rev.n_pair, org.n_pair) AS n_reciprocal_pair,
                    FROM
                      subset AS org
                    LEFT JOIN
                      subset AS rev
                    ON
                      org.engaged_user_id = rev.engaging_user_id
                      AND org.engaging_user_id = rev.engaged_user_id
                )
                SELECT
                -- A.engaged_user_id,
                -- A.engaging_user_id,
                B.n_reciprocal_pair,
                FROM {} AS A
                LEFT OUTER JOIN agg AS B
                ON A.engaged_user_id = B.engaged_user_id AND A.engaging_user_id = B.engaging_user_id
                ORDER BY A.tweet_id, B.engaging_user_id
        """.format(train_table_name, test_table_name, read_table_name)

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
        df_train_features = self._read_features_from_bigquery(self.train_table, self.test_table, self.train_table)
        df_test_features = self._read_features_from_bigquery(self.train_table, self.test_table, self.test_table)

        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingReciprocalPair.main()
