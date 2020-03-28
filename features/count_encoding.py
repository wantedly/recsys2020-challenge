import pandas as pd
from base import BaseFeature
from encoding_func import count_encoding


class CountEncoding(BaseFeature):
    def import_columns(self):
        return [
            "language",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        for col in self.import_columns():
            train_result, test_result = count_encoding(col, df_train_input, df_test_input)
            df_train_features[col] = train_result
            df_test_features[col] = test_result

        return df_train_features, df_test_features


if __name__ == "__main__":
    CountEncoding.main()
