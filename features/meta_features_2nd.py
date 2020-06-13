import os
import numpy as np
import pandas as pd
from base import BaseFeature
from google.cloud import storage
import itertools


meta_features_list = [
    "MetaFeatures2ndModels",
    "MetaFeaturesMLP"
]

class MetaFeatures2nd(BaseFeature):
    def import_columns(self):
        return [
            "1"
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        meta_features_train_list = []
        for meta in meta_features_list:
            meta_features_train = self._download_from_gs(
                feather_file_name=f"{meta}_training.ftr"
            )
            if meta == "MetaFeaturesMLP":
                meta_features_train.columns = [c.replace("_TargetCategories", "") for c in meta_features_train.columns]
            meta_features_train_list.append(meta_features_train)
        meta_features_train = pd.concat(meta_features_train_list, axis=1)

        meta_features_test_list = []
        for meta in meta_features_list:
            if self.TESTING:
                meta_features_test = self._download_from_gs(
                    feather_file_name=f"{meta}_test.ftr"
                )
            else:
                meta_features_test = self._download_from_gs(
                    feather_file_name=f"{meta}_val_20200418.ftr"
                )
            if meta == "MetaFeaturesMLP":
                meta_features_test.columns = [c.replace("_TargetCategories", "") for c in meta_features_test.columns]
            meta_features_test_list.append(meta_features_test)
        meta_features_test = pd.concat(meta_features_test_list, axis=1)

        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        for target_col in target_columns:
            for meta1, meta2 in itertools.combinations(meta_features_list, 2):
                df_train_features[f"diff_{meta1}_{meta2}_{target_col}"] = (
                    meta_features_train[f"{meta1}_{target_col}"].values -
                    meta_features_train[f"{meta2}_{target_col}"].values
                )
                df_test_features[f"diff_{meta1}_{meta2}_{target_col}"] = (
                    meta_features_test[f"{meta1}_{target_col}"].values -
                    meta_features_test[f"{meta2}_{target_col}"].values
                )
                df_train_features[f"ratio_{meta1}_{meta2}_{target_col}"] = (
                    meta_features_train[f"{meta1}_{target_col}"].values /
                    meta_features_train[f"{meta2}_{target_col}"].values
                )
                df_test_features[f"ratio_{meta1}_{meta2}_{target_col}"] = (
                    meta_features_test[f"{meta1}_{target_col}"].values /
                    meta_features_test[f"{meta2}_{target_col}"].values
                )

        print(df_train_features.shape)
        print(df_test_features.shape)
        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    MetaFeatures2nd.main()
