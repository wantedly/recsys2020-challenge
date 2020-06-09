from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import tensorflow as tf
import lightgbm as lgb
import dataclasses
from .model import Base_Model
from src.utils import Pkl


@dataclasses.dataclass
class Config:
    hidden_dims: List[int]


def build_model(
    n_features: int,
    config: Config,
    class_weights: List[Dict[int, float]],
    biases: List[float],
):
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
    for category, bias in zip(target_columns, biases):
        outputs.append(
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                name=category,
                bias_initializer=tf.keras.initializers.Constant(bias),
            )(x)
        )
    model = tf.keras.Model(features, outputs)

    def weighted_binary_crossentropy(class_weight):
        def loss_fn(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            bce_neg = tf.cast(y_true == 0, tf.float32) * class_weight[0] * bce
            bce_pos = tf.cast(y_true == 1, tf.float32) * class_weight[1] * bce
            weighted_bce = bce_neg + bce_pos # (batch_size, 1)
            return tf.reduce_mean(weighted_bce)  # tf.keras.losses.BinaryCrossentropy は Reduction = SUM_OVER_BATCH_SIZE
        return loss_fn

    loss_fns = [weighted_binary_crossentropy(w) for w in class_weights]

    model.compile(
        optimizer="adam",
        loss=loss_fns,
        metrics=[
            tf.keras.metrics.AUC(name="PRAUC", curve="PR"),
            tf.keras.metrics.BinaryCrossentropy(name="BCE"),
        ],
    )
    return model


class Model_NN(Base_Model):
    def train(self, x_trn, y_trn, x_val, y_val):
        self.num_fe = len(x_trn.columns)
        validation_flg = x_val is not None

        # Setting model parameters
        model_params = self.params["model_params"]
        config = Config(hidden_dims=model_params["hidden_dims"],)

        # Ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#train_a_model_with_class_weights
        class_weights = []
        biases = []
        for i in range(y_trn.shape[-1]):
            tgt = y_trn[:, i]
            pos = (tgt == 1).sum()
            neg = (tgt == 0).sum()
            total = pos + neg
            weight_for_0 = (1 / neg) * total / 2.0
            weight_for_1 = (1 / pos) * total / 2.0
            class_weights.append(
                {0: weight_for_0, 1: weight_for_1,}
            )
            biases.append(np.log(pos / neg))

        print(class_weights, y_trn.shape)

        self.model = build_model(x_trn.shape[-1], config, class_weights, biases)

        # Training
        if validation_flg:
            self.model.fit(
                np.asarray(x_trn),
                [y_trn[:, i] for i in range(4)],
                batch_size=model_params["batch_size"],
                epochs=model_params["epochs"],
                validation_data=(np.asarray(x_val), [y_val[:, i] for i in range(4)]),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10, monitor="val_PRAUC", mode="max"
                    )
                ],
            )
        else:
            self.model.fit(
                np.asarray(x_trn),
                [y_trn[:, i] for i in range(4)],
                batch_size=model_params["batch_size"],
                epochs=model_params["epochs"],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10, monitor="val_PRAUC", mode="max"
                    )
                ],
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
