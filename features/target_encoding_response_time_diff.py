import os
import pandas as pd
from base import BaseFeature
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1
from encoding_func import target_encoding


class TargetEncodingResponseTimeDiff(BaseFeature):
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
                    CASE WHEN n_engagement = 0 THEN NULL ELSE
                      TIMESTAMP_SECONDS(CAST( (
                      IF(like_engagement_timestamp IS NOT NULL, like_engagement_timestamp, 0) + 
                      IF(reply_engagement_timestamp IS NOT NULL, reply_engagement_timestamp, 0) + 
                      IF(retweet_engagement_timestamp IS NOT NULL, retweet_engagement_timestamp, 0) + 
                      IF(retweet_with_comment_engagement_timestamp IS NOT NULL, retweet_with_comment_engagement_timestamp, 0)
                      ) / n_engagement AS INT64)) END AS avg_engagement_timestamp
                FROM (
                    SELECT
                      tweet_id,
                      engaging_user_id,
                      like_engagement_timestamp,
                      reply_engagement_timestamp,
                      retweet_engagement_timestamp,
                      retweet_with_comment_engagement_timestamp,
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
                  TIMESTAMP_DIFF(avg_engagement_timestamp, LAG(avg_engagement_timestamp) OVER(PARTITION BY engaging_user_id ORDER BY avg_engagement_timestamp), MINUTE) AS diff_time
                FROM
                  response_times
                ORDER BY
                  tweet_id,
                  engaging_user_id
        """.format(read_table_name)
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
        df_train_input = self._read_features_from_bigquery(self.train_table)
        df_test_input = self._read_features_from_bigquery(self.test_table)

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        folds_train = self._download_from_gs(
            feather_file_name="TimeGroupKFold_training.ftr"
        )

        category_columns = [
            "engaging_user_id",
        ]

        target_columns = [
            "diff_time",
        ]

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            # Get folds
            folds_col = ["TimeGroupKFold_val_position"]
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
    TargetEncodingResponseTimeDiff.main()
