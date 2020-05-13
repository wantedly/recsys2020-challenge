import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class CountEncodingEngagingHashtags(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                with unnest_subset as (
                select
                  engaging_user_id, tweet_id, hashtag, n_engaging
                from
                  (
                    select
                      engaging_user_id, tweet_id, hashtags,
                      count(1) over(partition by engaging_user_id) n_engaging,
                    FROM (
                      (SELECT engaging_user_id, tweet_id, hashtags, FROM {})
                      UNION ALL
                      (SELECT engaging_user_id, tweet_id, hashtags, FROM {})
                    )
                  ) t,
                  unnest(t.hashtags) as hashtag
                ),
                agg1 as (
                select
                  engaging_user_id, count(1) / any_value(n_engaging) as engaging_hashtag_tweets_rate, count(1) as n_engaging_hashtag_tweets
                from
                  (
                  select
                    engaging_user_id, tweet_id, any_value(n_engaging) as n_engaging
                  from
                    unnest_subset
                  group by
                    engaging_user_id, tweet_id
                  )
                  group by
                    engaging_user_id
                ),
                agg2 as (
                  select
                    engaging_user_id, hashtag, count(1) as n_engaging_hashtag
                  from
                    unnest_subset
                  group by
                    engaging_user_id, hashtag
                )
                select
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  B.engaging_hashtag_tweets_rate,
                  B.n_engaging_hashtag_tweets,
                  B.avg_n_engaging_hashtag,
                  B.min_n_engaging_hashtag,
                  B.max_n_engaging_hashtag,
                  B.std_n_engaging_hashtag,
                from
                  {} as A
                left join
                (
                select
                  tweet_id,
                  engaging_user_id,
                  any_value(engaging_hashtag_tweets_rate) as engaging_hashtag_tweets_rate,
                  any_value(n_engaging_hashtag_tweets) as n_engaging_hashtag_tweets,
                  avg(n_engaging_hashtag) avg_n_engaging_hashtag,
                  min(n_engaging_hashtag) min_n_engaging_hashtag,
                  max(n_engaging_hashtag) max_n_engaging_hashtag,
                  case when stddev(n_engaging_hashtag) is null then 1 else stddev(n_engaging_hashtag) end as std_n_engaging_hashtag,
                from
                (
                select
                  us.tweet_id, us.engaging_user_id, us.hashtag,
                  a1.engaging_hashtag_tweets_rate, a1.n_engaging_hashtag_tweets,
                  a2.n_engaging_hashtag,
                from
                  unnest_subset us
                left join
                  agg1 a1
                on
                  us.engaging_user_id = a1.engaging_user_id
                left join
                  agg2 a2
                on
                  us.engaging_user_id = a2.engaging_user_id and us.hashtag = a2.hashtag
                )
                group by
                  tweet_id, engaging_user_id
                ) AS B
                on
                  A.tweet_id = B.tweet_id and A.engaging_user_id = B.engaging_user_id
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
    CountEncodingEngagingHashtags.main()
