import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEncodingEngagingPresentLinks(BaseFeature):
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
                  engaging_user_id, tweet_id, present_link, n_engaging
                from
                  (
                    select
                      engaging_user_id, tweet_id, present_links,
                      count(1) over(partition by engaging_user_id) n_engaging,
                    FROM (
                      (SELECT engaging_user_id, tweet_id, present_links, FROM {})
                      UNION ALL
                      (SELECT engaging_user_id, tweet_id, present_links, FROM {})
                    )
                  ) t,
                  unnest(t.present_links) as present_link
                ),
                agg1 as (
                select
                  engaging_user_id, count(1) / any_value(n_engaging) as engaging_present_link_tweets_rate, count(1) as n_engaging_present_link_tweets
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
                    engaging_user_id, present_link, count(1) as n_engaging_present_link
                  from
                    unnest_subset
                  group by
                    engaging_user_id, present_link
                )
                select
                  -- A.tweet_id,
                  -- A.engaging_user_id,
                  B.engaging_present_link_tweets_rate,
                  B.n_engaging_present_link_tweets,
                  B.avg_n_engaging_present_link,
                  B.min_n_engaging_present_link,
                  B.max_n_engaging_present_link,
                  B.std_n_engaging_present_link,
                from
                  {} as A
                left join
                (
                select
                  tweet_id,
                  engaging_user_id,
                  any_value(engaging_present_link_tweets_rate) as engaging_present_link_tweets_rate,
                  any_value(n_engaging_present_link_tweets) as n_engaging_present_link_tweets,
                  avg(n_engaging_present_link) avg_n_engaging_present_link,
                  min(n_engaging_present_link) min_n_engaging_present_link,
                  max(n_engaging_present_link) max_n_engaging_present_link,
                  case when stddev(n_engaging_present_link) is null then 1 else stddev(n_engaging_present_link) end as std_n_engaging_present_link,
                from
                (
                select
                  us.tweet_id, us.engaging_user_id, us.present_link,
                  a1.engaging_present_link_tweets_rate, a1.n_engaging_present_link_tweets,
                  a2.n_engaging_present_link,
                from
                  unnest_subset us
                left join
                  agg1 a1
                on
                  us.engaging_user_id = a1.engaging_user_id
                left join
                  agg2 a2
                on
                  us.engaging_user_id = a2.engaging_user_id and us.present_link = a2.present_link
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
    CountEncodingEngagingPresentLinks.main()
