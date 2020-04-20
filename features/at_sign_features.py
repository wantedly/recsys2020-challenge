import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class AtSignFeatures(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_text: str, test_text: str,
            train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH at_sign_feature AS (
                    SELECT
                        tweet_id,
                        any_value(n_at_sign_begin) as n_at_sign_begin,
                        any_value(n_at_sign) as n_at_sign
                    FROM (
                      (
                        SELECT
                        tweet_id,
                        case when char_length(regexp_extract(regexp_replace(text, r'(@[a-zA-Z0-9_]{{1,15}})', '@'), '^@+')) is null then 0 else char_length(regexp_extract(regexp_replace(text, r'(@[a-zA-Z0-9_]{{1,15}})', '@'), '^@+')) end as n_at_sign_begin,
                        array_length(regexp_extract_all(text, r'(@[a-zA-Z0-9_]{{1,15}})')) as n_at_sign
                        FROM {}
                      )
                      UNION ALL
                      (
                        SELECT
                            tweet_id,
                            case when char_length(regexp_extract(regexp_replace(text, r'(@[a-zA-Z0-9_]{{1,15}})', '@'), '^@+')) is null then 0 else char_length(regexp_extract(regexp_replace(text, r'(@[a-zA-Z0-9_]{{1,15}})', '@'), '^@+')) end as n_at_sign_begin,
                            array_length(regexp_extract_all(text, r'(@[a-zA-Z0-9_]{{1,15}})')) as n_at_sign
                        FROM {}
                      )
                    )
                    GROUP BY tweet_id
                )
                , agg_features AS (
                SELECT
                    A.engaged_user_id,
                    max(B.n_at_sign_begin) AS max_n_at_sign_begin,
                    avg(B.n_at_sign_begin) AS avg_n_at_sign_begin,
                    max(B.n_at_sign) AS max_n_at_sign,
                    avg(B.n_at_sign) AS avg_n_at_sign
                FROM (
                    (SELECT tweet_id, engaged_user_id FROM {} GROUP BY tweet_id, engaged_user_id)
                    UNION ALL
                    (SELECT tweet_id, engaged_user_id FROM {} GROUP BY tweet_id, engaged_user_id)
                ) AS A 
                LEFT OUTER JOIN at_sign_feature AS B
                ON A.tweet_id = B.tweet_id
                GROUP BY A.engaged_user_id
                )

                SELECT
                B.n_at_sign_begin,
                B.n_at_sign,
                C.max_n_at_sign_begin,
                C.avg_n_at_sign_begin,
                C.max_n_at_sign,
                C.avg_n_at_sign
                FROM {} AS A
                LEFT OUTER JOIN at_sign_feature AS B
                ON A.tweet_id = B.tweet_id
                LEFT OUTER JOIN agg_features AS C
                ON A.engaging_user_id = C.engaged_user_id
                ORDER BY
                A.tweet_id, A.engaging_user_id
        """.format(
            train_text, test_text, train_table_name, test_table_name, read_table_name)
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
        train_text = f"`{PROJECT_ID}.recsys2020.texts_training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
            test_text = f"`{PROJECT_ID}.recsys2020.texts_test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
            test_text = f"`{PROJECT_ID}.recsys2020.texts_val_20200418`"

        df_train_features = self._read_features_from_bigquery(train_text, test_text, train_table, test_table, train_table)
        df_test_features = self._read_features_from_bigquery(train_text, test_text, train_table, test_table, test_table)
        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    AtSignFeatures.main()
