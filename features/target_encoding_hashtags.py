import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


class TargetEncodingHashtags(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaging_user_id"
        ]

    def _read_inter_table_from_bigquery(
            self, table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = """
                WITH subset AS (
                    SELECT
                        tweet_id,
                        engaging_user_id,
                        any_value(hashtags) AS hashtags,
                        any_value(reply_engagement_timestamp) AS reply_engagement_timestamp,
                        any_value(retweet_engagement_timestamp) AS retweet_engagement_timestamp,
                        any_value(retweet_with_comment_engagement_timestamp) AS retweet_with_comment_engagement_timestamp,
                        any_value(like_engagement_timestamp) AS like_engagement_timestamp,
                    FROM {}
                    GROUP BY tweet_id, engaging_user_id
                )
                , unnest_subset AS (
                SELECT
                    tweet_id,
                    engaging_user_id,
                    hashtag,
                    reply_engagement_timestamp,
                    retweet_engagement_timestamp,
                    retweet_with_comment_engagement_timestamp,
                    like_engagement_timestamp
                FROM subset,
                unnest(hashtags) AS hashtag
                )
                , use_combination AS (
                SELECT hashtag, engaging_user_id, COUNT(*) AS cnt
                FROM unnest_subset
                GROUP BY hashtag, engaging_user_id
                HAVING COUNT(*) >= 1
                )
                SELECT
                    unnest_subset.tweet_id,
                    unnest_subset.engaging_user_id,
                    unnest_subset.hashtag,
                    CASE WHEN unnest_subset.reply_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS reply_engagement,
                    CASE WHEN unnest_subset.retweet_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_engagement,
                    CASE WHEN unnest_subset.retweet_with_comment_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_with_comment_engagement,
                    CASE WHEN unnest_subset.like_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS like_engagement,
                FROM
                    unnest_subset
                INNER JOIN
                    use_combination
                    ON unnest_subset.hashtag = use_combination.hashtag
                    AND unnest_subset.engaging_user_id = use_combination.engaging_user_id
        """.format(table_name)
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
        train_data = self._read_inter_table_from_bigquery(self.train_table)
        test_data = self._read_inter_table_from_bigquery(self.test_table)

        train_data["engaging_user_id__hashtag"] = train_data["engaging_user_id"] + "_" + train_data["hashtag"]
        test_data["engaging_user_id__hashtag"] = test_data["engaging_user_id"] + "_" + test_data["hashtag"]

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        folds_train = self._download_from_gs(
            feather_file_name="StratifiedGroupKFold_training.ftr"
        )

        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        category_column = "engaging_user_id__hashtag"
        target_encoding_column = f"{category_column}_ta"

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            # Get folds
            folds_col = ["StratifiedGroupKFold_retweet_with_comment_engagement"]
            assert len(folds_col) == 1, "The number of fold column must be one"
            folds = folds_train[folds_col]
            n_fold = folds.max().values[0] + 1
            folds_ids = []
            folds_tweet_ids = []

            for i in range(n_fold):
                trn_idx = folds[folds != i].dropna().index
                val_idx = folds[folds == i].dropna().index
                folds_ids.append((trn_idx, val_idx))
                print(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")

                trn_tweet_id = df_train_input.iloc[trn_idx]["tweet_id"].unique()
                val_tweet_id = df_train_input.iloc[val_idx]["tweet_id"].unique()
                print(f"{i+1}fold: n_tweet_trn={len(trn_tweet_id)}, n_tweet_val={len(val_tweet_id)}")

                trn_tweet_idx = train_data.loc[train_data["tweet_id"].isin(trn_tweet_id)].index
                val_tweet_idx = train_data.loc[train_data["tweet_id"].isin(val_tweet_id)].index
                folds_tweet_ids.append((trn_tweet_idx, val_tweet_idx))
                print(f"{i+1}fold: n_tweet_trn={len(trn_tweet_idx)}, n_tweet_trn={len(val_tweet_idx)}")

            _, _ = target_encoding(
                category_column, train_data, test_data, target_col, folds_tweet_ids)

            train_agg = train_data.groupby(["tweet_id", "engaging_user_id"])[target_encoding_column].agg(["min", "max", "mean"]).reset_index()
            test_agg = test_data.groupby(["tweet_id", "engaging_user_id"])[target_encoding_column].agg(["min", "max", "mean"]).reset_index()
            train_data.drop(columns=[target_encoding_column], inplace=True)
            test_data.drop(columns=[target_encoding_column], inplace=True)
            feature_names = ['min', 'max', 'mean']

            for fe in feature_names:
                df_train_features[f"{target_col}_{fe}"] = pd.merge(df_train_input, train_agg, on=["tweet_id", "engaging_user_id"], how="left")[fe].values
                df_test_features[f"{target_col}_{fe}"] = pd.merge(df_test_input, test_agg, on=["tweet_id", "engaging_user_id"], how="left")[fe].values

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TargetEncodingHashtags.main()
