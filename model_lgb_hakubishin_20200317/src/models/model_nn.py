from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf
import lightgbm as lgb
import dataclasses
from .model import Base_Model
from src.utils import Pkl


@dataclasses.dataclass
class Config:
    hidden_dims: List[int]


def build_model(n_features: int, config: Config):
    features = tf.keras.Input(shape=(n_features,), name="features", dtype=tf.float32)
    x = features
    for hidden in config.hidden_dims:
        x = tf.keras.layers.Dense(hidden)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
    outputs = []
    target_columns = [
        "reply_engagement",
        "retweet_engagement",
        "retweet_with_comment_engagement",
        "like_engagement",
    ]
    for category in target_columns:
        outputs.append(tf.keras.layers.Dense(1, activation="sigmoid", name=category)(x))
    model = tf.keras.Model(features, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="PRAUC", curve="PR"),
            tf.keras.metrics.BinaryCrossentropy(name="BCE"),
        ],
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),],
    )
    return model


class Model_NN(Base_Model):
    def train(self, x_trn, y_trn, x_val, y_val):
        self.num_fe = len(x_trn.columns)
        validation_flg = x_val is not None

        # Setting model parameters
        model_params = self.params["model_params"]
        config = Config(hidden_dims=model_params["hidden_dims"],)
        self.model = build_model(x_trn.shape[-1], config)

        # Training
        if validation_flg:
            self.model.fit(
                np.asarray(x_trn),
                np.asarray(y_trn),
                batch_size=model_params["batch_size"],
                epochs=model_params["epochs"],
                validation_data=(np.asarray(x_val), np.asarray(y_val)),
            )
        else:
            self.model.fit(
                np.asarray(x_trn),
                np.asarray(y_trn),
                batch_size=model_params["batch_size"],
                epochs=model_params["epochs"],
            )

    def predict(self, x):
        return self.model.predict(np.asarray(x))

    def get_feature_importance(self):
        return np.zeros(self.num_fe)

    def get_best_iteration(self):
        return 0

    def save_model(self):
        model_path = self.model_output_dir / f"{self.run_fold_name}.h5"
        self.model.save(str(model_path))

    def load_model(self):
        model_path = self.model_output_dir / f"{self.run_fold_name}.h5"
        self.model = tf.keras.models.load_model(str(model_path))
