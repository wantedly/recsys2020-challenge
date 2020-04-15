import pandas as pd
from base import BaseFeature
from encoding_func import count_encoding


class CountEncodingCrossing(BaseFeature):
    def import_columns(self):
        return [
            "concat(language, '_', tweet_type) as language_tweet_type",
            "concat(engaging_user_id, '_', tweet_type) as engaging_user_id_tweet_type",
            "concat(engaging_user_id, '_', language) as engaging_user_id_language",
            #"concat(engaged_user_id, '_', tweet_type) as engaged_user_id_tweet_type",
            #"concat(engaged_user_id, '_', engaging_user_id, '_', tweet_type) as engaged_engaging_tweet_type",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        cols = [
            "language_tweet_type",
            "engaging_user_id_tweet_type",
            "engaging_user_id_language",
            #"engaged_user_id_tweet_type",
            #"engaged_engaging_tweet_type",
        ]

        for col in cols:
            train_result, test_result = count_encoding(col, df_train_input, df_test_input)
            df_train_features[col] = train_result
            df_test_features[col] = test_result

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingCrossing.main()
