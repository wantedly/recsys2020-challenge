import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class AggFollowFollowingCountOf1dConnectUsers(BaseFeature):
    def import_columns(self):
        return [
            "1"
        ]

    def _read_features_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH follow_edges AS (
                    SELECT
                        engaged_user_id, engaging_user_id,
                        MAX(engaged_follower_count) as engaged_follower_count, MAX(engaged_following_count) as engaged_following_count,
                        MAX(engaging_follower_count) as engaging_follower_count, MAX(engaging_following_count) as engaging_following_count
                    FROM (
                            (
                            SELECT engaged_user_id, engaging_user_id, engagee_follows_engager, engaged_follower_count, engaged_following_count, engaging_follower_count, engaging_following_count
                            FROM {}
                            )
                        UNION ALL
                            (
                            SELECT engaged_user_id, engaging_user_id, engagee_follows_engager, engaged_follower_count, engaged_following_count, engaging_follower_count, engaging_following_count
                            FROM {}
                            )
                    )
                    GROUP BY engaged_user_id, engaging_user_id
                    HAVING MAX(CAST(engagee_follows_engager AS INT64)) = 1
                )
                , following_users_attr AS (
                    SELECT
                        engaged_user_id AS user_id,
                        AVG(engaging_follower_count) AS mean_follower_count_of_following_users,
                        AVG(engaging_following_count) AS mean_following_count_of_following_users,
                    FROM follow_edges
                    GROUP BY engaged_user_id
                )
                , follower_users_attr AS (
                    SELECT
                        engaging_user_id AS user_id,
                        AVG(engaged_follower_count) AS mean_follower_count_of_follower_users,
                        AVG(engaged_following_count) AS mean_following_count_of_follower_users,
                    FROM follow_edges
                    GROUP BY engaging_user_id
                )

                SELECT
                    B.mean_follower_count_of_follower_users AS mean_follower_count_of_follower_users__engaging_user,
                    B.mean_following_count_of_follower_users AS mean_following_count_of_follower_users__engaging_user,
                    C.mean_follower_count_of_follower_users AS mean_follower_count_of_follower_users__engaged_user,
                    C.mean_following_count_of_follower_users AS mean_following_count_of_follower_users__engaged_user,
                    D.mean_follower_count_of_following_users AS mean_follower_count_of_following_users__engaging_user,
                    D.mean_following_count_of_following_users AS mean_following_count_of_following_users__engaging_user,
                    E.mean_follower_count_of_following_users AS mean_follower_count_of_following_users__engaged_user,
                    E.mean_following_count_of_following_users AS mean_following_count_of_following_users__engaged_user,
                FROM {} AS A
                LEFT OUTER JOIN follower_users_attr AS B
                ON A.engaging_user_id = B.user_id
                LEFT OUTER JOIN follower_users_attr AS C
                ON A.engaged_user_id = C.user_id
                LEFT OUTER JOIN following_users_attr AS D
                ON A.engaging_user_id = D.user_id
                LEFT OUTER JOIN following_users_attr AS E
                ON A.engaged_user_id = E.user_id
                ORDER BY A.tweet_id, A.engaging_user_id
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
        # read unnested present_media
        df_train_features = self._read_features_from_bigquery
            self.train_table, self.test_table, self.train_table)
        df_test_features = self._read_features_from_bigquery(
            self.train_table, self.test_table, self.test_table)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    AggFollowFollowingCountOf1dConnectUsers.main()
