import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class Connected2ndEngagingToEngaged(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id"
        ]

    def _read_2nd_connected_count_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH follow_edges AS (
                SELECT
                    engaged_user_id, engaging_user_id
                FROM (
                    (
                        SELECT engaged_user_id, engaging_user_id, MAX(CAST(engagee_follows_engager AS INT64)) AS engagee_follows_engager
                        FROM {}
                        GROUP BY engaged_user_id, engaging_user_id
                    )
                    UNION ALL
                    (
                        SELECT engaged_user_id, engaging_user_id, MAX(CAST(engagee_follows_engager AS INT64)) AS engagee_follows_engager
                        FROM {}
                        GROUP BY engaged_user_id, engaging_user_id
                    )
                )
                GROUP BY
                    engaged_user_id, engaging_user_id
                HAVING
                    MAX(CAST(engagee_follows_engager AS INT64)) = 1
                )

                SELECT
                -- t0.tweet_id,
                -- t0.engaging_user_id,
                cast(logical_or(f2.engaging_user_id is not null) as int64) AS engaging_to_engaged_2nd,
                cast(logical_or(f2.engaging_user_id is not null) or logical_or(f3.engaged_user_id is not null) as int64) AS engaging_to_engaged_1st_and_2nd,
                FROM {} as t0
                LEFT JOIN follow_edges f1 ON t0.engaged_user_id = f1.engaging_user_id
                LEFT JOIN follow_edges f2 ON f1.engaged_user_id = f2.engaging_user_id and t0.engaging_user_id = f2.engaged_user_id
                LEFT JOIN follow_edges f3 ON t0.engaging_user_id = f3.engaged_user_id AND t0.engaged_user_id = f3.engaging_user_id
                GROUP BY t0.tweet_id, t0.engaging_user_id
                ORDER BY t0.tweet_id, t0.engaging_user_id
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
        train_2nd_connected = self._read_2nd_connected_count_from_bigquery(
            self.train_table, self.test_table, self.train_table
        )
        test_2nd_connected = self._read_2nd_connected_count_from_bigquery(
            self.train_table, self.test_table, self.test_table
        )
        feature_names = ["engaging_to_engaged_2nd", "engaging_to_engaged_1st_and_2nd"]
        print(train_2nd_connected.shape)
        print(test_2nd_connected.shape)

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        for fe in feature_names:
            df_train_features[fe] = train_2nd_connected[fe].values
            df_test_features[fe] = test_2nd_connected[fe].values

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    Connected2ndEngagingToEngaged.main()
