import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage


class TargetEncodingInTweetidUnits(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
            "engaged_user_id",
            "engaged_follower_count",
            "CASE WHEN like_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS like_engagement",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        folds_train = self._download_from_gs(
            feather_file_name="StratifiedGroupKFold_training.ftr"
        )

        category_columns = [
            "engaged_user_id",
        ]

        target_columns = [
            "like_engagement",
        ]

        # 1tweetあたりのengagementの合計値
        df_train_input_tweet_id = df_train_input.groupby(["tweet_id", "engaged_user_id"])[target_columns].sum().reset_index()
        df_test_input_tweet_id = df_test_input.groupby(["tweet_id", "engaged_user_id"])[target_columns].sum().reset_index()

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

                trn_tweet_idx = df_train_input_tweet_id.loc[df_train_input_tweet_id["tweet_id"].isin(trn_tweet_id)].index
                val_tweet_idx = df_train_input_tweet_id.loc[df_train_input_tweet_id["tweet_id"].isin(val_tweet_id)].index
                folds_tweet_ids.append((trn_tweet_idx, val_tweet_idx))
                print(f"{i+1}fold: n_tweet_trn={len(trn_tweet_idx)}, n_tweet_trn={len(val_tweet_idx)}")

            for cat_col in category_columns:
                # tweet_id単位のengagement総和による{cat_col}のtarget_encoding
                _, _ = target_encoding(
                    cat_col, df_train_input_tweet_id, df_test_input_tweet_id, target_col, folds_tweet_ids)

                df_train_features[f"{target_col}__{cat_col}"] = (
                    pd.merge(df_train_input, df_train_input_tweet_id[[f"{cat_col}_ta", "tweet_id", cat_col]], on=["tweet_id", cat_col], how="left")
                )[f"{cat_col}_ta"].values

                df_test_features[f"{target_col}__{cat_col}"] = (
                    pd.merge(df_test_input, df_test_input_tweet_id[[f"{cat_col}_ta", "tweet_id", cat_col]], on=["tweet_id", cat_col], how="left")
                )[f"{cat_col}_ta"].values

                df_train_input_tweet_id.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_test_input_tweet_id.drop(columns=[f"{cat_col}_ta"], inplace=True)

        eps = 1e-2
        df_train_features["like_follower_ratio"] = (
            df_train_features["like_engagement__engaged_user_id"] /
            (df_train_input["engaged_follower_count"] + eps)
        )
        df_test_features["like_follower_ratio"] = (
            df_test_features["like_engagement__engaged_user_id"] /
            (df_test_input["engaged_follower_count"] + eps)
        )

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TargetEncodingInTweetidUnits.main()
