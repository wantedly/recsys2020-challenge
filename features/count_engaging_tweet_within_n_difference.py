import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEngagingTweetWithinNDifference(BaseFeature):
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
                ),
                cnt_within_time_diff as (
                  select
                    engaging_user_id,
                    tweet_id,
                    cnt_within_1min_later - cnt_within_1min_earlier as diff_1min,
                    cnt_within_5min_later - cnt_within_5min_earlier as diff_5min,
                    cnt_within_10min_later - cnt_within_10min_earlier as diff_10min,
                    cnt_within_30min_later - cnt_within_30min_earlier as diff_30min,
                    cnt_within_1hour_later - cnt_within_1hour_earlier as diff_1hour,
                    cnt_within_2hours_later - cnt_within_2hours_earlier as diff_2hours,
                    cnt_within_4hours_later - cnt_within_4hours_earlier as diff_4hours,
                    cnt_within_8hours_later - cnt_within_8hours_earlier as diff_8hours,
                    cnt_within_12hours_later - cnt_within_12hours_earlier as diff_12hours,
                    cnt_within_24hours_later - cnt_within_24hours_earlier as diff_24hours,
                  from
                    cnt_within_time
                ),
                agg as (
                  select
                    engaging_user_id,
                    max(abs(diff_1min)) as max_abs_diff_1min,
                    abs(avg(diff_1min)) as abs_avg_diff_1min,
                    max(abs(diff_5min)) as max_abs_diff_5min,
                    abs(avg(diff_5min)) as abs_avg_diff_5min,
                    max(abs(diff_10min)) as max_abs_diff_10min,
                    abs(avg(diff_10min)) as abs_avg_diff_10min,
                    max(abs(diff_30min)) as max_abs_diff_30min,
                    abs(avg(diff_30min)) as abs_avg_diff_30min,
                    max(abs(diff_1hour)) as max_abs_diff_1hour,
                    abs(avg(diff_1hour)) as abs_avg_diff_1hour,
                    max(abs(diff_2hours)) as max_abs_diff_2hours,
                    abs(avg(diff_2hours)) as abs_avg_diff_2hours,
                    max(abs(diff_4hours)) as max_abs_diff_4hours,
                    abs(avg(diff_4hours)) as abs_avg_diff_4hours,
                    max(abs(diff_8hours)) as max_abs_diff_8hours,
                    abs(avg(diff_8hours)) as abs_avg_diff_8hours,
                    max(abs(diff_12hours)) as max_abs_diff_12hours,
                    abs(avg(diff_12hours)) as abs_avg_diff_12hours,
                    max(abs(diff_24hours)) as max_abs_diff_24hours,
                    abs(avg(diff_24hours)) as abs_avg_diff_24hours,
                  from
                    cnt_within_time_diff
                  group by
                    engaging_user_id
                ),
                features as (
                  select
                    diff.engaging_user_id,
                    diff.tweet_id,
                    diff_1min,
                    diff_1min / (max_abs_diff_1min + 1) as diff_1min_divided_by_max_abs,
                    diff_1min / (abs_avg_diff_1min + 1) as diff_1min_divided_by_abs_avg,
                    diff_5min,
                    diff_5min / (max_abs_diff_5min + 1) as diff_5min_divided_by_max_abs,
                    diff_5min / (abs_avg_diff_5min + 1) as diff_5min_divided_by_abs_avg,
                    diff_10min,
                    diff_10min / (max_abs_diff_10min + 1) as diff_10min_divided_by_max_abs,
                    diff_10min / (abs_avg_diff_10min + 1) as diff_10min_divided_by_abs_avg,
                    diff_30min,
                    diff_30min / (max_abs_diff_30min + 1) as diff_30min_divided_by_max_abs,
                    diff_30min / (abs_avg_diff_30min + 1) as diff_30min_divided_by_abs_avg,
                    diff_1hour,
                    diff_1hour / (max_abs_diff_1hour + 1) as diff_1hour_divided_by_max_abs,
                    diff_1hour / (abs_avg_diff_1hour + 1) as diff_1hour_divided_by_abs_avg,
                    diff_2hours,
                    diff_2hours / (max_abs_diff_2hours + 1) as diff_2hours_divided_by_max_abs,
                    diff_2hours / (abs_avg_diff_2hours + 1) as diff_2hours_divided_by_abs_avg,
                    diff_4hours,
                    diff_4hours / (max_abs_diff_4hours + 1) as diff_4hours_divided_by_max_abs,
                    diff_4hours / (abs_avg_diff_4hours + 1) as diff_4hours_divided_by_abs_avg,
                    diff_8hours,
                    diff_8hours / (max_abs_diff_8hours + 1) as diff_8hours_divided_by_max_abs,
                    diff_8hours / (abs_avg_diff_8hours + 1) as diff_8hours_divided_by_abs_avg,
                    diff_12hours,
                    diff_12hours / (max_abs_diff_12hours + 1) as diff_12hours_divided_by_max_abs,
                    diff_12hours / (abs_avg_diff_12hours + 1) as diff_12hours_divided_by_abs_avg,
                    diff_24hours,
                    diff_24hours / (max_abs_diff_24hours + 1) as diff_24hours_divided_by_max_abs,
                    diff_24hours / (abs_avg_diff_24hours + 1) as diff_24hours_divided_by_abs_avg,
                  from
                    cnt_within_time_diff as diff
                  left join
                    agg
                  on
                    diff.engaging_user_id = agg.engaging_user_id
                )

                select
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  B.* except(tweet_id, engaging_user_id)
                from
                  {} as A
                left join
                  features AS B
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
    CountEngagingTweetWithinNDifference.main()
