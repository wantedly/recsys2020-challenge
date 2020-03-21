import pandas as pd
from base import BaseFeature


class CommonFlgFeatures(BaseFeature):
    def import_columns(self):
        return [
            "engaged_is_verified",
            "engaging_is_verified",
            "engagee_follows_engager",
        ]

    def make_features(self, df_train_input, df_test_input):
        flg_columns = [
            "engaged_is_verified",
            "engaging_is_verified",
            "engagee_follows_engager",
        ]

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()
        for col in flg_columns:
            df_train_features[col] = df_train_input[col].astype(int).values
            df_test_features[col] = df_test_input[col].astype(int).values

        return df_train_features, df_test_features


if __name__ == "__main__":
    CommonFlgFeatures.main()
