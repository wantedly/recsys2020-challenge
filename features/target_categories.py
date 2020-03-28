import pandas as pd
from base import BaseFeature


class TargetCategories(BaseFeature):
    def import_columns(self):
        return [
            "CASE WHEN reply_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS reply_engagement",
            "CASE WHEN retweet_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_engagement",
            "CASE WHEN retweet_with_comment_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_with_comment_engagement",
            "CASE WHEN like_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS like_engagement",
        ]

    def make_features(self, df_train_input, df_test_input):
        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()
        for col in target_columns:
            df_train_features[col] = df_train_input[col].values
            df_test_features[col] = df_test_input[col].values

        return df_train_features, df_test_features


if __name__ == "__main__":
    TargetCategories.main()
