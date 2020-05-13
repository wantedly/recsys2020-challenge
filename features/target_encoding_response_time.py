import os
import pandas as pd
from base import BaseFeature
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1
from encoding_func import target_encoding


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class TargetEncodingResponseTime(BaseFeature):
    def import_columns(self):
        return [
            "1"
       ]

    def _read_features_from_bigquery(self, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                WITH
                response_times AS (
                SELECT
                    tweet_id,
                    engaging_user_id,
                    CASE WHEN n_engagement = 0 THEN NULL ELSE (
                    IF(like_response_time IS NOT NULL, like_response_time, 0) + 
                    IF(reply_response_time IS NOT NULL, reply_response_time, 0) + 
                    IF(retweet_response_time IS NOT NULL, retweet_response_time, 0) + 
                    IF(retweet_with_comment_response_time IS NOT NULL, retweet_with_comment_response_time, 0)
                    ) / n_engagement END AS avg_response_time
                FROM (
                    SELECT
                    tweet_id,
                    engaging_user_id,
                    TIMESTAMP_DIFF(TIMESTAMP_SECONDS(like_engagement_timestamp), TIMESTAMP_SECONDS(timestamp), MINUTE) AS like_response_time,
                    TIMESTAMP_DIFF(TIMESTAMP_SECONDS(reply_engagement_timestamp), TIMESTAMP_SECONDS(timestamp), MINUTE) AS reply_response_time,
                    TIMESTAMP_DIFF(TIMESTAMP_SECONDS(retweet_engagement_timestamp), TIMESTAMP_SECONDS(timestamp), MINUTE) AS retweet_response_time,
                    TIMESTAMP_DIFF(TIMESTAMP_SECONDS(retweet_with_comment_engagement_timestamp), TIMESTAMP_SECONDS(timestamp), MINUTE) AS retweet_with_comment_response_time,
                    IF(like_engagement_timestamp IS NOT NULL, 1, 0) + 
                    IF(reply_engagement_timestamp IS NOT NULL, 1, 0) + 
                    IF(retweet_engagement_timestamp IS NOT NULL, 1, 0) + 
                    IF(retweet_with_comment_engagement_timestamp IS NOT NULL, 1, 0) AS n_engagement
                    FROM
                    {}
                  )
                )

                SELECT
                  tweet_id,
                  engaging_user_id,
                  avg_response_time
                FROM
                  response_times
                ORDER BY
                  tweet_id,
                  engaging_user_id
        """.format(read_table_name)
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
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"

        df_train_input = self._read_features_from_bigquery(train_table)
        df_test_input = self._read_features_from_bigquery(test_table)

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        folds_train = self._download_from_gs(
            feather_file_name="StratifiedGroupKFold_training.ftr"
        )

        category_columns = [
            "engaging_user_id",
        ]

        target_columns = [
            "avg_response_time",
        ]

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            # Get folds
            folds_col = ["StratifiedGroupKFold_retweet_with_comment_engagement"]
            assert len(folds_col) == 1, "The number of fold column must be one"
            folds = folds_train[folds_col]
            n_fold = folds.max().values[0] + 1
            folds_ids = []

            for i in range(n_fold):
                trn_idx = folds[folds != i].dropna().index
                val_idx = folds[folds == i].dropna().index
                folds_ids.append((trn_idx, val_idx))
                print(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")

            for cat_col in category_columns:
                train_result, test_result = target_encoding(
                    cat_col, df_train_input, df_test_input, target_col, folds_ids)
                df_train_input.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_test_input.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_train_features[f"{target_col}__{cat_col}"] = train_result
                df_test_features[f"{target_col}__{cat_col}"] = test_result

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TargetEncodingResponseTime.main()
