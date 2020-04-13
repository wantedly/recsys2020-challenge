import pandas as pd
from base import BaseFeature
from encoding_func import count_encoding


class CountEncoding(BaseFeature):
    def import_columns(self):
        return [
            "language",
            "engaging_user_id",
            "engaged_user_id",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        for col in self.import_columns():
            train_result, test_result = count_encoding(col, df_train_input, df_test_input)
            df_train_features[col] = train_result
            df_test_features[col] = test_result

        df_train_input["engaged_engaging"] = df_train_input["engaged_user_id"] + df_train_input["engaging_user_id"]
        df_test_input["engaged_engaging"] = df_test_input["engaged_user_id"] + df_test_input["engaging_user_id"]
        for col in ["engaged_engaging"]:
            train_result, test_result = count_encoding(col, df_train_input, df_test_input)
            df_train_features[col] = train_result
            df_test_features[col] = test_result

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncoding.main()
