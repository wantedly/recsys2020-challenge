import os
import numpy as np
import pandas as pd
from base import BaseFeature
from google.cloud import storage


class MetaFeatures(BaseFeature):
    def import_columns(self):
        return [
            "1"
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            model_outpput_path = "../model_lgb_hakubishin_20200317/data/output/model_44/"
            oof_pred_path = model_outpput_path + f"{target_col}_oof_pred.npy"
            oof_pred = np.load(oof_pred_path)
            df_train_features[target_col] = oof_pred

            import pdb; pdb.set_trace()
            test_pred_path = model_outpput_path + f"{target_col}_submission_val_20200418.csv"
            test_pred = pd.read_csv(test_pred_path, header=None).iloc[:, 2]
            df_test_features[target_col] = test_pred


        print(df_train_features.shape)
        print(df_test_features.shape)
        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    MetaFeatures.main()
