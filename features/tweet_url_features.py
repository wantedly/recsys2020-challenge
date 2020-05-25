import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class TweetUrlFeatures(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_text: str, test_text: str,
            train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH url_features AS (
                    SELECT
                        tweet_id,
                        any_value(n_url) as n_url,
                        any_value(CASE WHEN n_url > 0 THEN 1 ELSE 0 END) as include_url,
                        any_value(only_url) as only_url
                    FROM (
                      (
                        SELECT
                        tweet_id,
                        array_length(regexp_extract_all(text,  r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+")) as n_url,
                        case when text <> "" and regexp_replace(text, r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+", "") = "" then 1 else 0 end only_url,
                        FROM {}
                      )
                      UNION ALL
                      (
                        SELECT
                            tweet_id,
                            array_length(regexp_extract_all(text,  r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+")) as n_url,
                            case when text <> "" and regexp_replace(text, r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+", "") = "" then 1 else 0 end only_url,
                        FROM {}
                      )
                    )
                    GROUP BY tweet_id
                )
                , subset AS (
                SELECT
                    *
                FROM (
                    (SELECT tweet_id, engaged_user_id, engaging_user_id FROM {} GROUP BY tweet_id, engaged_user_id, engaging_user_id)
                    UNION ALL
                    (SELECT tweet_id, engaged_user_id, engaging_user_id FROM {} GROUP BY tweet_id, engaged_user_id, engaging_user_id)
                )
                )
                , engaged_agg_features AS (
                SELECT
                    A.engaged_user_id,
                    max(B.n_url) AS max_n_url,
                    avg(B.n_url) AS avg_n_url,
                    avg(B.include_url) AS include_url_rate,
                    avg(B.only_url) AS only_url_rate
                FROM (
                  SELECT tweet_id, engaged_user_id FROM subset GROUP BY tweet_id, engaged_user_id
                ) AS A
                LEFT OUTER JOIN url_features AS B
                ON A.tweet_id = B.tweet_id
                GROUP BY A.engaged_user_id
                )
                , engaging_agg_features AS (
                SELECT
                    A.engaging_user_id,
                    avg(B.n_url) AS avg_n_url,
                    avg(B.include_url) AS include_url_rate,
                    avg(B.only_url) AS only_url_rate
                FROM (
                    SELECT tweet_id, engaging_user_id FROM subset GROUP BY tweet_id, engaging_user_id
                ) AS A
                LEFT OUTER JOIN url_features AS B
                ON A.tweet_id = B.tweet_id
                GROUP BY A.engaging_user_id
                )

                SELECT
                B.n_url,
                B.include_url,
                B.only_url,
                C.max_n_url AS engaged_engaged_max_n_url,
                C.avg_n_url AS engaged_engaged_avg_n_url,
                C.include_url_rate AS engaged_engaged_include_url_rate,
                C.only_url_rate AS engaged_engaged_only_url_rate,
                D.max_n_url AS engaging_engaged_max_n_url,
                D.avg_n_url AS engaging_engaged_avg_n_url,
                D.include_url_rate AS engaging_engaged_include_url_rate,
                D.only_url_rate AS engaging_engaged_only_url_rate,
                E.avg_n_url AS engaged_engaging_avg_n_url,
                E.include_url_rate AS engaged_engaging_include_url_rate,
                E.only_url_rate AS engaged_engaging_only_url_rate,
                F.avg_n_url AS engaging_engaging_avg_n_url,
                F.include_url_rate AS engaging_engaging_include_url_rate,
                F.only_url_rate AS engaging_engaging_only_url_rate,
                FROM {} AS A
                LEFT OUTER JOIN url_features AS B
                ON A.tweet_id = B.tweet_id
                LEFT OUTER JOIN engaged_agg_features AS C
                ON A.engaged_user_id = C.engaged_user_id
                LEFT OUTER JOIN engaged_agg_features AS D
                ON A.engaging_user_id = D.engaged_user_id
                LEFT OUTER JOIN engaging_agg_features AS E
                ON A.engaged_user_id = E.engaging_user_id
                LEFT OUTER JOIN engaging_agg_features AS F
                ON A.engaging_user_id = F.engaging_user_id
                ORDER BY
                A.tweet_id, A.engaging_user_id
        """.format(
            train_text, test_text, train_table_name, test_table_name, read_table_name)
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
        df_train_features = self._read_features_from_bigquery(self.train_text, self.test_text, self.train_table, self.test_table, self.train_table)
        df_test_features = self._read_features_from_bigquery(self.train_text, self.test_text, self.train_table, self.test_table, self.test_table)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TweetUrlFeatures.main()
