import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class EngagingUserFollowsEngagedUser(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id"
        ]

    def _read_1nd_connected_count_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH follow_edges AS (
                SELECT
                    engaged_user_id, engaging_user_id,
                    MAX(CAST(engagee_follows_engager AS INT64)) AS engaged_user_follows_engaging_user
                FROM (
                    (
                        SELECT engaged_user_id, engaging_user_id, engagee_follows_engager
                        FROM {}
                    )
                    UNION ALL
                    (
                        SELECT engaged_user_id, engaging_user_id, engagee_follows_engager
                        FROM {}
                    )
                )
                GROUP BY
                    engaged_user_id, engaging_user_id
                )
                SELECT
                -- t0.engaged_user_id,
                -- t0.engaging_user_id,
                -- t0.engagee_follows_engager,
                f1.engaged_user_follows_engaging_user AS engaging_user_follows_engaged_user,
                FROM {} AS t0
                LEFT OUTER JOIN follow_edges AS f1
                ON t0.engaging_user_id = f1.engaged_user_id
                AND t0.engaged_user_id = f1.engaging_user_id
                ORDER BY t0.tweet_id, t0.engaging_user_id
        """.format(train_table_name, test_table_name, read_table_name)
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
        # read unnested present_media
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
        train_1nd_connected = self._read_1nd_connected_count_from_bigquery(train_table, test_table, train_table)
        test_1nd_connected = self._read_1nd_connected_count_from_bigquery(train_table, test_table, test_table)
        feature_names = ["engaging_user_follows_engaged_user"]
        print(train_1nd_connected.shape)
        print(train_1nd_connected.isnull().sum())
        print(test_1nd_connected.shape)
        print(test_1nd_connected.isnull().sum())

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        for fe in feature_names:
            df_train_features[fe] = train_1nd_connected[fe].values
            df_test_features[fe] = test_1nd_connected[fe].values

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    EngagingUserFollowsEngagedUser.main()
