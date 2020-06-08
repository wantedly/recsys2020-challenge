import os
import pandas as pd
from google.cloud import storage
from io import BytesIO


PROJECT_ID = "wantedly-individual-naomichi"
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"


class FeatureLoader(object):
    def __init__(self, data_type="train", debugging: bool=False):
        self.client = storage.Client(project=PROJECT_ID)
        self.bucket = self.client.get_bucket(GCS_BUCKET_NAME)
        self.debugging = debugging
        self.data_type = data_type

    def download_feather_from_gs(self, feature_class_name: str) -> pd.DataFrame:
        if self.debugging:
            bucket_dir_name = "features_debug"
        else:
            bucket_dir_name = "features"

        feature_file_name = f"{feature_class_name}_{self.data_type}.ftr"
        blob = storage.Blob(
            os.path.join(bucket_dir_name, feature_file_name),
            self.bucket
        )
        content = blob.download_as_string()
        print(f"Downloading {feature_file_name} to {blob.path}")
        df_feature = pd.read_feather(BytesIO(content))
        return df_feature

    def download_pred_from_gs(self, model_name: str, target: str) -> np.array:
        if self.data_type == "training":
            file_name = f"{target}_oof.npy"
        elif self.data_type == "test":
            file_name = f"{target}_submission_test.csv"

        bucket_dir_name = "model_lgb_hakubishin_20200317/" + model_name 
        blob = storage.Blob(
            os.path.join(bucket_dir_name, file_name),
            self.bucket
        )
        content = blob.download_as_string()
        print(f"Downloading {file_name} to {blob.path}")

        if self.data_type == "training":
            pred = np.load(BytesIO(content))
        elif self.data_type == "test":
            pred = pd.read_csv(BytesIO(content), header=None)
            pred = pred.iloc[:, 2].values

        return pred

    def load_features(self, download_features_list):
        df_features_list = [] 
        for feature_class_name in download_features_list:
            df_feature = self.download_feather_from_gs(feature_class_name)
            df_features_list.append(df_feature)
        return pd.concat(df_features_list, axis=1)
