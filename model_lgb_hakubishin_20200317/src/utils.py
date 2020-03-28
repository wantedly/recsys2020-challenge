import os
import json
import random
import codecs
import logging
import numpy as np
from sklearn.externals import joblib
from typing import List
from google.cloud import storage, bigquery


def upload_to_gcs(bucket_dir_name: str, files: List[str]):
    GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
    PROJECT_ID = "wantedly-individual-naomichi"
    client = storage.Client(project=PROJECT_ID)
    bucket = client.get_bucket(GCS_BUCKET_NAME)

    for filename in files:
        basename = os.path.basename(filename)
        blob = storage.Blob(os.path.join(bucket_dir_name, basename), bucket)
        print(f"Uploading {basename} to {blob.path}")
        blob.upload_from_filename(str(filename))


def seed_everything(seed: int=71, gpu_mode: bool=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if gpu_mode:
        import tensorflow as tf
        import torch
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


class MyEncoder(json.JSONEncoder):
    """ encode numpy objects
    https://wtnvenga.hatenablog.com/entry/2018/05/27/113848
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def json_dump(dict_object, save_path) -> None:
    f = codecs.open(save_path, 'w', 'utf-8')
    json.dump(dict_object, f, indent=4, cls=MyEncoder, ensure_ascii=False)


class Pkl(object):
    """ https://github.com/ghmagazine/kagglebook/blob/master/ch04-model-interface/code/util.py
    """
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


def get_logger(module_name=None, save_path=None):
    logger = logging.getLogger(module_name)
    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
    logger.setLevel(logging.DEBUG)

    if save_path is None:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        handler = logging.FileHandler(save_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
