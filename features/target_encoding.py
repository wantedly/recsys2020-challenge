import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage
from io import BytesIO


GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"

class TargetEncoding(BaseFeature):
    def import_columns(self):
        return [
            "language",
            "engaged_user_id",
            "engaging_user_id",
            "CASE WHEN reply_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS reply_engagement",
            "CASE WHEN retweet_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_engagement",
            "CASE WHEN retweet_with_comment_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS retweet_with_comment_engagement",
            "CASE WHEN like_engagement_timestamp IS NULL THEN 0 ELSE 1 END AS like_engagement",
        ]

    def download_feather_from_gs(self, feature_class_name: str,
                                 data_type: str = "training") -> pd.DataFrame:
        if self.debugging:
            bucket_dir_name = "features_debug"
        else:
            bucket_dir_name = "features"

        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        feature_file_name = f"{feature_class_name}_{data_type}.ftr"
        blob = storage.Blob(
            os.path.join(bucket_dir_name, feature_file_name),
            bucket
        )
        content = blob.download_as_string()
        print(f"Downloading {feature_file_name} to {blob.path}")
        df_feature = pd.read_feather(BytesIO(content))
        return df_feature

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        folds_train = self.download_feather_from_gs(
            data_type="training",
            feature_class_name="StratifiedGroupKFold"
        )

        category_columns = [
            "language",
            "engaged_user_id",
            "engaging_user_id",
        ]

        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            # Get folds
            folds_col = [c for c in folds_train.columns if c.find(target_col) != -1]
            assert len(folds_col) == 1, "The number of fold column must be one"
            folds = folds_train[folds_col]
            n_fold = folds.max().values[0]
            folds_ids = []

            for i in range(n_fold):
                trn_idx = folds[folds != i+1].dropna().index
                val_idx = folds[folds == i+1].dropna().index
                folds_ids.append((trn_idx, val_idx))
                print(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")

            for cat_col in category_columns:
                train_result, test_result = target_encoding(
                    cat_col, df_train_input, df_test_input, target_col, folds_ids)
                df_train_input.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_test_input.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_train_features[f"{target_col}__{cat_col}"] = train_result
                df_test_features[f"{target_col}__{cat_col}"] = test_result

        return df_train_features, df_test_features


if __name__ == "__main__":
    TargetEncoding.main()
