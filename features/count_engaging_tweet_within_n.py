import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEngagingTweetWithinN(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                with data as (
                  select
                    engaging_user_id, tweet_id, timestamp
                  from (
                    (SELECT engaging_user_id, tweet_id, timestamp, FROM {})
                    UNION ALL
                    (SELECT engaging_user_id, tweet_id, timestamp, FROM {})
                  )
                ),
                subset as (
                  select
                    d1.engaging_user_id,
                    d1.tweet_id,
                    d1.timestamp - d2.timestamp timestamp_lag,
                  from
                    data d1
                  left join
                    data d2
                  on
                    d1.engaging_user_id = d2.engaging_user_id
                ),
                cnt_within_time as (
                  select
                    engaging_user_id,
                    tweet_id,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 1) -1 as cnt_within_1min_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * (-1)) - 1 as  cnt_within_1min_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 5) -1 as cnt_within_5min_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * (-5)) - 1 as  cnt_within_5min_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 10) -1 as cnt_within_10min_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * (-10)) - 1 as  cnt_within_10min_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 30) -1 as cnt_within_30min_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * (-30)) - 1 as  cnt_within_30min_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 60 * 1) -1 as cnt_within_1hour_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * 60 * (-1)) - 1 as  cnt_within_1hour_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 60 * 2) -1 as cnt_within_2hours_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * 60 * (-2)) - 1 as  cnt_within_2hours_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 60 * 4) -1 as cnt_within_4hours_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * 60 * (-4)) - 1 as  cnt_within_4hours_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 60 * 8) -1 as cnt_within_8hours_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * 60 * (-8)) - 1 as  cnt_within_8hours_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 60 * 12) - 1 as cnt_within_12hours_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * 60 * (-12)) - 1 as cnt_within_12hours_later,
                    countif(timestamp_lag >= 0 and timestamp_lag < 60 * 60 * 24) -1 as cnt_within_24hours_earlier,
                    countif(timestamp_lag <= 0 and timestamp_lag > 60 * 60 * (-24)) - 1 as  cnt_within_24hours_later,
                  from
                    subset
                  group by
                    engaging_user_id, tweet_id
                )
                select
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  B.* except(tweet_id, engaging_user_id)
                from
                  {} as A
                left join
                  cnt_within_time AS B
                on
                  A.tweet_id = B.tweet_id and A.engaging_user_id = B.engaging_user_id
                order by
                  A.tweet_id, A.engaging_user_id
        """.format(
            train_table_name, test_table_name, read_table_name)
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
        df_train_features = self._read_features_from_bigquery(self.train_table, self.test_table, self.train_table)
        df_test_features = self._read_features_from_bigquery(self.train_table, self.test_table, self.test_table)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEngagingTweetWithinN.main()
