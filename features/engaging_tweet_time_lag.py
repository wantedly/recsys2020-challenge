import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class EngagingTweetTimeLag(BaseFeature):
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
                    engaging_user_id,
                    tweet_id,
                    timestamp,
                    lag(timestamp) over(partition by engaging_user_id order by timestamp asc) timestamp_of_prev,
                    lead(timestamp) over(partition by engaging_user_id order by timestamp asc) timestamp_of_next,
                  from
                    data
                ),
                time_lags as (
                  select
                    engaging_user_id,
                    tweet_id,
                    timestamp - timestamp_of_prev + 1 as time_lag_to_prev,
                    timestamp_of_next - timestamp + 1 as time_lag_to_next,
                  from
                    subset
                ),
                agg as (
                  select
                    engaging_user_id,
                    avg( time_lag_to_prev) as avg_time_lag_to_prev,
                    max( time_lag_to_prev) as max_time_lag_to_prev,
                    avg(time_lag_to_next) as avg_time_lag_to_next,
                    max(time_lag_to_next) max_time_lag_to_next,
                  from
                    time_lags
                  group by
                    engaging_user_id
                )
                select
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  D.time_lag_to_prev,
                  D.time_lag_to_next,
                  D.time_lag_to_prev_divided_by_avg,
                  D.time_lag_to_prev_divided_by_max,
                  D.time_lag_to_next_divided_by_avg,
                  D.time_lag_to_next_divided_by_max,
                from
                  {} as A
                left join
                (
                select
                  B.engaging_user_id,
                  B.tweet_id,
                  B.time_lag_to_prev,
                  B.time_lag_to_next,
                  B.time_lag_to_prev / C.avg_time_lag_to_prev as time_lag_to_prev_divided_by_avg,
                  B.time_lag_to_prev / C.max_time_lag_to_prev as time_lag_to_prev_divided_by_max,
                  B.time_lag_to_next / C.avg_time_lag_to_next as time_lag_to_next_divided_by_avg,
                  B.time_lag_to_next / C.max_time_lag_to_next as time_lag_to_next_divided_by_max,
                from
                  time_lags B
                left join
                  agg C
                on
                  B.engaging_user_id = C.engaging_user_id
                ) AS D
                on
                  A.tweet_id = D.tweet_id and A.engaging_user_id = D.engaging_user_id
                order by
                  A.tweet_id, A.engaging_user_id
        """.format(
            train_table_name, test_table_name, read_table_name)
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
        # read features
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"

        df_train_features = self._read_features_from_bigquery(train_table, test_table, train_table)
        df_test_features = self._read_features_from_bigquery(train_table, test_table, test_table)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    EngagingTweetTimeLag.main()
