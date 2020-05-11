import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class FFFeatures(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                CREATE TEMPORARY FUNCTION binom_confidence_intervals(conf FLOAT64, x FLOAT64, n FLOAT64)
                RETURNS ARRAY<FLOAT64>
                LANGUAGE js AS '''
                  lower = x == 0 ? 0 : jStat.ibetainv((1 - conf)/2, x, n - x + 1)
                  upper = jStat.ibetainv(1 - (1-conf)/2, x + 1, n - x)
                  return [lower, upper]
                '''
                OPTIONS (
                library="gs://wantedly-1371/jstat.js"
                );

                WITH subset AS (
                SELECT
                    *
                FROM (
                    (SELECT engaged_user_id, engaging_user_id, engaged_following_count, engaged_follower_count, engaging_following_count, engaging_follower_count,
                      CASE WHEN engaged_follower_count >= engaged_following_count THEN binom_confidence_intervals(0.90, engaged_following_count, engaged_follower_count + 1e-2)[ORDINAL(1)] ELSE NULL END engaged_ff_power,
                      CASE WHEN engaging_follower_count >= engaging_following_count THEN binom_confidence_intervals(0.90, engaging_following_count, engaging_follower_count + 1e-2)[ORDINAL(1)] ELSE NULL END engaging_ff_power,
                      FROM {})
                    UNION ALL
                    (SELECT engaged_user_id, engaging_user_id, engaged_following_count, engaged_follower_count, engaging_following_count, engaging_follower_count,
                      CASE WHEN engaged_follower_count >= engaged_following_count THEN binom_confidence_intervals(0.90, engaged_following_count, engaged_follower_count + 1e-2)[ORDINAL(1)] ELSE NULL END engaged_ff_power,
                      CASE WHEN engaging_follower_count >= engaging_following_count THEN binom_confidence_intervals(0.90, engaging_following_count, engaging_follower_count + 1e-2)[ORDINAL(1)] ELSE NULL END engaging_ff_power,
                      FROM {})
                )
                )
                , engaged_agg_features AS (
                SELECT
                  engaged_user_id,
                  ANY_VALUE(engaged_ff_power) AS engaged_ff_power,
                  ANY_VALUE(med_engaging_following_count) AS med_engaging_following_count,
                  ANY_VALUE(med_engaging_follower_count) AS med_engaging_follower_count,
                  AVG(engaging_following_count) AS avg_engaging_following_count,
                  AVG(engaging_follower_count) AS avg_engaging_follower_count,
                  SUM(engaging_following_count) AS sum_engaging_following_count,
                  SUM(engaging_follower_count) AS sum_engaging_follower_count,
                FROM
                  (
                  SELECT
                    engaged_user_id,
                    engaging_following_count,
                    engaging_follower_count,
                    engaged_ff_power,
                    PERCENTILE_CONT(engaging_following_count, 0.5) OVER(PARTITION BY engaged_user_id) AS med_engaging_following_count,
                    PERCENTILE_CONT(engaging_follower_count, 0.5) OVER(PARTITION BY engaged_user_id) AS med_engaging_follower_count,
                  FROM
                    subset
                  )
                GROUP BY
                  engaged_user_id
                )
                , engaging_agg_features AS (
                SELECT
                  engaging_user_id,
                  ANY_VALUE(engaging_ff_power) AS engaging_ff_power,
                  ANY_VALUE(med_engaged_following_count) AS med_engaged_following_count,
                  ANY_VALUE(med_engaged_follower_count) AS med_engaged_follower_count,
                  AVG(engaged_following_count) AS avg_engaged_following_count,
                  AVG(engaged_follower_count) AS avg_engaged_follower_count,
                  SUM(engaged_following_count) AS sum_engaged_following_count,
                  SUM(engaged_follower_count) AS sum_engaged_follower_count,
                FROM
                  (
                  SELECT
                    engaging_user_id,
                    engaged_following_count,
                    engaged_follower_count,
                    engaging_ff_power,
                    PERCENTILE_CONT(engaged_following_count, 0.5) OVER(PARTITION BY engaging_user_id) med_engaged_following_count,
                    PERCENTILE_CONT(engaged_follower_count, 0.5) OVER(PARTITION BY engaging_user_id) med_engaged_follower_count,
                  FROM
                    subset
                  )
                GROUP BY
                  engaging_user_id
                )

                SELECT
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                B.engaged_ff_power AS engaged_ff_power,
                B.med_engaging_following_count AS engaged_med_engaging_following_count,
                B.med_engaging_follower_count AS engaged_med_engaging_follower_count,
                B.avg_engaging_following_count AS engaged_avg_engaging_following_count,
                B.avg_engaging_follower_count AS engaged_avg_engaging_follower_count,
                B.sum_engaging_following_count AS engaged_sum_engaging_following_count,
                B.sum_engaging_follower_count AS engaged_sum_engaging_follower_count,
                C.med_engaging_following_count AS engaging_med_engaging_following_count,
                C.med_engaging_follower_count AS engaging_med_engaging_follower_count,
                C.avg_engaging_following_count AS engaging_avg_engaging_following_count,
                C.avg_engaging_follower_count AS engaging_avg_engaging_follower_count,
                C.sum_engaging_following_count AS engaging_sum_engaging_following_count,
                C.sum_engaging_follower_count AS engaging_sum_engaging_follower_count,

                D.engaging_ff_power AS engaging_ff_power,
                D.med_engaged_following_count AS engaged_med_engaged_following_count,
                D.med_engaged_follower_count AS engaged_med_engaged_follower_count,
                D.avg_engaged_following_count AS engaged_avg_engaged_following_count,
                D.avg_engaged_follower_count AS engaged_avg_engaged_follower_count,
                D.sum_engaged_following_count AS engaged_sum_engaged_following_count,
                D.sum_engaged_follower_count AS engaged_sum_engaged_follower_count,
                E.med_engaged_following_count AS engaging_med_engaged_following_count,
                E.med_engaged_follower_count AS engaging_med_engaged_follower_count,
                E.avg_engaged_following_count AS engaging_avg_engaged_following_count,
                E.avg_engaged_follower_count AS engaging_avg_engaged_follower_count,
                E.sum_engaged_following_count AS engaging_sum_engaged_following_count,
                E.sum_engaged_follower_count AS engaging_sum_engaged_follower_count,
                FROM {} AS A
                LEFT OUTER JOIN engaged_agg_features AS B
                ON A.engaged_user_id = B.engaged_user_id
                LEFT OUTER JOIN engaged_agg_features AS C
                ON A.engaging_user_id = C.engaged_user_id
                LEFT OUTER JOIN engaging_agg_features AS D
                ON A.engaged_user_id = D.engaging_user_id
                LEFT OUTER JOIN engaging_agg_features AS E
                ON A.engaging_user_id = E.engaging_user_id
                ORDER BY
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
    FFFeatures.main()
