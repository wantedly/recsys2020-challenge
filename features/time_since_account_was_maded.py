import pandas as pd
from base import BaseFeature


class ElapsedTimeFromAccountCreated(BaseFeature):
    def import_columns(self):
        return [
            "timestamp_diff(TIMESTAMP_SECONDS(Timestamp), TIMESTAMP_SECONDS(engaged_account_creation_time), DAY) AS engaged_account",
            "timestamp_diff(TIMESTAMP_SECONDS(Timestamp), TIMESTAMP_SECONDS(engaging_account_creation_time), DAY) AS engaging_account"
        ]

    def make_features(self, df_train_input, df_test_input):
        fe_columns = [
            "engaged_account",
            "engaging_account"
        ]

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()
        for col in fe_columns:
            df_train_features[col] = df_train_input[col].values
            df_test_features[col] = df_test_input[col].values

        return df_train_features, df_test_features


if __name__ == "__main__":
    ElapsedTimeFromAccountCreated.main()
