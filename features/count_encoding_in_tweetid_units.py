import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEncodingInTweetidUnits(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id"
        ]

    def _read_count_info_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH agg AS (
                SELECT
                    engaged_user_id,
                    count(distinct tweet_id) as n_tweet,
                    sum(n_engagement) / count(distinct tweet_id) as n_engagement_per_tweet
                FROM (
                    (
                        SELECT tweet_id, any_value(engaged_user_id) as engaged_user_id, count(*) as n_engagement
                        FROM {}
                        GROUP BY tweet_id
                    )
                    UNION ALL
                    (
                        SELECT tweet_id, any_value(engaged_user_id) as engaged_user_id, count(*) as n_engagement
                        FROM {}
                        GROUP BY tweet_id
                    )
                )
                GROUP BY
                    engaged_user_id
                )
                SELECT
                -- t0.engaged_user_id,
                f1.n_tweet,
                f1.n_engagement_per_tweet,
                FROM {} AS t0
                LEFT OUTER JOIN agg AS f1
                ON t0.engaged_user_id = f1.engaged_user_id
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
        train_info = self._read_count_info_from_bigquery(
            self.train_table, self.test_table, self.train_table
        )
        test_info = self._read_count_info_from_bigquery(
            self.train_table, self.test_table, self.test_table
        )
        feature_names = ["n_tweet", "n_engagement_per_tweet"]
        print(train_info.shape)
        print(train_info.isnull().sum())
        print(test_info.shape)
        print(test_info.isnull().sum())

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        for fe in feature_names:
            df_train_features[fe] = train_info[fe].values
            df_test_features[fe] = test_info[fe].values

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingInTweetidUnits.main()
