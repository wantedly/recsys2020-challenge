import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class CountEncodingTweetType(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_count_tweet_type_count_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH subset AS (
                  (
                    SELECT tweet_id, any_value(engaged_user_id) as engaged_user_id, any_value(tweet_type) as tweet_type
                    FROM {}
                    GROUP BY tweet_id
                  )
                UNION ALL
                  (
                    SELECT tweet_id, any_value(engaged_user_id) as engaged_user_id, any_value(tweet_type) as tweet_type
                    FROM {}
                    GROUP BY tweet_id
                  )
                )
                , count_tweet_type AS (
                SELECT
                    engaged_user_id,
                    countif(tweet_type = "Quote") as n_quote_tweet,
                    countif(tweet_type = "Retweet") as n_retweet_tweet,
                    countif(tweet_type = "Quote") + countif(tweet_type = "Retweet") as n_quote_plus_retweet,
                    countif(tweet_type = "Quote") / count(*) as ratio_quote_tweet,
                    countif(tweet_type = "Retweet") / count(*) as ratio_retweet_tweet,
                    (countif(tweet_type = "Quote") + countif(tweet_type = "Retweet")) / count(*) as ratio_quote_plus_retweet
                FROM subset
                GROUP BY engaged_user_id
                )

                SELECT
                -- t0.engaged_user_id,
                -- t0.engaging_user_id,
                f1.n_quote_tweet,
                f1.n_retweet_tweet,
                f1.n_quote_plus_retweet,
                f1.ratio_quote_tweet,
                f1.ratio_retweet_tweet,
                f1.ratio_quote_plus_retweet,
                FROM {} AS t0
                LEFT OUTER JOIN count_tweet_type AS f1
                ON t0.engaging_user_id = f1.engaged_user_id
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
        train_info = self._read_count_tweet_type_count_from_bigquery(
            self.train_table, self.test_table, self.train_table
        )
        test_info = self._read_count_tweet_type_count_from_bigquery(
            self.train_table, self.test_table, self.test_table
        )
        feature_names = [
            "n_quote_tweet",
            "n_retweet_tweet",
            "n_quote_plus_retweet",
            "ratio_quote_tweet",
            "ratio_retweet_tweet",
            "ratio_quote_plus_retweet"
        ]
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
    CountEncodingTweetType.main()
