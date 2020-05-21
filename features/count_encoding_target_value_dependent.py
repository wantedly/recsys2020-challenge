import pandas as pd
from base import BaseFeature
from encoding_func import count_encoding_target_value_dependent


class CountEncodingTargetValueDependent(BaseFeature):
    def import_columns(self):
        return [
            "language",
            "engaging_user_id",
            "engaged_user_id",
            "CASE WHEN reply_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS reply_engagement",
            "CASE WHEN retweet_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_engagement",
            "CASE WHEN retweet_with_comment_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_with_comment_engagement",
            "CASE WHEN like_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS like_engagement",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()
        df_train_input["engaged_engaging"] = df_train_input["engaged_user_id"] + df_train_input["engaging_user_id"]
        df_test_input["engaged_engaging"] = df_test_input["engaged_user_id"] + df_test_input["engaging_user_id"]

        cat_columns = [
            "language",
            "engaging_user_id",
            "engaged_user_id",
            "engaged_engaging"
        ]

        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            for col in cat_columns:
                fe_col = f"{col}_{target_col}"
                train_result, test_result = count_encoding_target_value_dependent(
                    col, df_train_input, df_test_input, target_col
                )
                df_train_features[fe_col] = train_result
                df_test_features[fe_col] = test_result
 
        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncodingTargetValueDependent.main()
