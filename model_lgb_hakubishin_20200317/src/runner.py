import pathlib
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Callable
from .metrics import calc_metrics
from .models.model import Base_Model


class Runner(object):
    def __init__(self, model_cls: Callable[[str, dict, pathlib.PosixPath], Base_Model],
            model_params: dict, model_output_dir: pathlib.PosixPath, run_name: str) -> None:
        self.model_cls = model_cls
        self.model_params = model_params
        self.model_output_dir = model_output_dir
        self.run_name = run_name
        self.n_fold = None
        self.under_sampling_rate = []

    def build_model(self, i_fold):
        run_fold_name = f'{self.run_name}_{i_fold}'
        return self.model_cls(run_fold_name, self.model_params, self.model_output_dir)

    def train_one_fold(self, i_fold: int, x_trn: pd.DataFrame, y_trn: Union[pd.Series, np.array],
            x_val: pd.DataFrame, y_val: Union[pd.Series, np.array]) -> Tuple[
            Base_Model, float, np.array]:

        model = self.build_model(i_fold)
        model.train(x_trn, y_trn, x_val, y_val)
        val_preds = model.predict(x_val)

        return model, val_preds

    def train_cv(self, x_train: pd.DataFrame, y_train: Union[pd.Series, np.array], x_test: pd.DataFrame,
            folds_ids: List[Tuple[np.array]], train_settings: dict, target: str) -> Tuple[np.array, dict]:
        oof_preds = np.zeros(len(x_train))
        preds_list = []
        cv_score_list = []
        best_iteration = 0
        self.n_fold = len(folds_ids)
        n_models = train_settings["n_models"]
        np.random.seed(train_settings["random_sampling"]["random_seed"])

	    # Set params
        if target == "like_engagement":
            train_settings["model"]["model_params"]["subsample"] = 0.5
            train_settings["model"]["model_params"]["subsample_for_bin"] = int(0.1 * y_train.sum())
        else:
            train_settings["model"]["model_params"]["subsample"] = 0.9
            train_settings["model"]["model_params"]["subsample_for_bin"] = 1000000

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            print(f"{i_fold+1}fold")

            # Split arrays into train and valid subsets
            y_trn = y_train[trn_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train[val_idx]
            print(f"original train size: {len(y_trn)}")
            print(f"trn_pos={y_trn.sum()}, trn_neg={(y_trn == 0).sum()}")

            # Negative down sampling
            positive_idx_of_trn_idx = np.where(y_trn == 1)[0]  # trn_idx の何番目が positive か
            negative_idx_of_trn_idx = np.where(y_trn == 0)[0]  # trn_idx の何番目が negative か

            positive_ratio = float(y_trn.sum()) / len(trn_idx)
            under_sampling_rate = positive_ratio / (1 - positive_ratio)
            self.under_sampling_rate.append(under_sampling_rate)

            for i_model in range(n_models):
                randint = np.random.randint(10000000)
                train_settings["model"]["model_params"]["seed"] = randint
                train_settings["model"]["model_params"]["bagging_seed"] = randint
                train_settings["model"]["model_params"]["feature_fraction_seed"] = randint
                train_settings["model"]["model_params"]["drop_seed"] = randint

                positive_sampling_keys = np.random.random(len(positive_idx_of_trn_idx))
                negative_sampling_keys = np.random.random(len(negative_idx_of_trn_idx))
                required_data_size = train_settings["random_sampling"]["n_data"] // 2

                if required_data_size > len(positive_idx_of_trn_idx):
                    # required_data_sizeがpositiveサンプル数より大きかった場合
                    required_data_size = len(positive_idx_of_trn_idx)

                # 乱数が (欲しいデータ数) / (今のデータ数) より小さいものをサンプリング
                resampled_pos_idx_of_trn_idx = positive_idx_of_trn_idx[positive_sampling_keys < required_data_size / len(positive_idx_of_trn_idx)]
                resampled_neg_idx_of_trn_idx = negative_idx_of_trn_idx[negative_sampling_keys < required_data_size / len(negative_idx_of_trn_idx)]
                # trn_idx の何番目を採用するか
                resampled_idx_of_trn_idx = np.concatenate([resampled_pos_idx_of_trn_idx, resampled_neg_idx_of_trn_idx])
                # x_train の何番目を採用するか
                resampled_trn_idx = trn_idx[resampled_idx_of_trn_idx]
                resampled_x_trn = x_train.iloc[resampled_trn_idx]
                resampled_y_trn = y_train[resampled_trn_idx]

                print(f"train size after random-sampling: {len(resampled_y_trn)}")
                print(f"trn_pos={resampled_y_trn.sum()}, trn_neg={(resampled_y_trn == 0).sum()}")

                # Training
                model, val_pred = self.train_one_fold(i_fold, resampled_x_trn, resampled_y_trn, x_val, y_val)
                val_pred = val_pred / (val_pred + ((1 - val_pred) / under_sampling_rate))
                oof_preds[val_idx] += val_pred / n_models
                best_iteration += model.get_best_iteration() / len(folds_ids) / n_models

                # Predict
                y_pred = model.predict(x_test)
                y_pred = y_pred / (y_pred + ((1 - y_pred) / under_sampling_rate))
                preds_list.append(y_pred)

                # Save model
                model.save_model()

            # done calculation for one-fold
            score = calc_metrics(y_val, oof_preds[val_idx])
            cv_score_list.append(score)

        # Calculate OOF-Score
        oof_score = calc_metrics(y_train, oof_preds)
        print(f"oof: {oof_score}")

        pred_avg = np.mean(preds_list, axis=0)

        # Summary
        evals_result = {"evals_result": {
            "n_data": len(x_train),
            "n_features": len(x_train.columns),
            "best_iteration": best_iteration,
            "under_sampling_rate": {f"cv{i+1}": rate for i, rate in enumerate(self.under_sampling_rate)},
            "oof_score": oof_score,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
        }}
        return oof_preds, pred_avg, evals_result

    def predict_cv(self, x: pd.DataFrame) -> np.array:
        preds_list = []

        for i_fold in range(self.n_fold):
            model = self.build_model(i_fold)
            model.load_model()
            y_pred = model.predict(x)
            under_sampling_rate = self.under_sampling_rate[i_fold]
            y_pred = y_pred / (y_pred + ((1 - y_pred) / under_sampling_rate))
            preds_list.append(y_pred)

        pred_avg = np.mean(preds_list, axis=0)
        return pred_avg

    def train_all(self, x_train: pd.DataFrame, y_train: Union[pd.Series, np.array]) -> None:
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.train(x_train, y_train, None, None)
        model.save_model()

    def predict_all(self, x: pd.DataFrame) -> np.array:
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()

        return model.predict(x)
