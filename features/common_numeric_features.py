import pandas as pd
from base import BaseFeature


class CommonNumericFeatures(BaseFeature):
    def import_columns(self):
        return [
            "engaged_follower_count",
            "engaged_following_count",
            "engaging_follower_count",
            "engaging_following_count",
            "ARRAY_LENGTH(text_tokens) AS n_text_tokens",
            "ARRAY_LENGTH(hashtags) AS n_hashtags",
            "ARRAY_LENGTH(present_media) AS n_present_media",
            "ARRAY_LENGTH(present_links) AS n_present_links",
            "ARRAY_LENGTH(present_domains) AS n_present_domains",
        ]

    def make_features(self, df_train_input, df_test_input):
        numeric_columns = [
            "engaged_follower_count",
            "engaged_following_count",
            "engaging_follower_count",
            "engaging_following_count",
            "n_text_tokens",
            "n_hashtags",
            "n_present_media",
            "n_present_links",
            "n_present_domains",
        ]

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()
        for col in numeric_columns:
            df_train_features[col] = df_train_input[col].values
            df_test_features[col] = df_test_input[col].values

        return df_train_features, df_test_features


if __name__ == "__main__":
    CommonNumericFeatures.main()
