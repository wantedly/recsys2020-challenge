import pathlib
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Callable
from .metrics import calc_metrics
from .models.model import Base_Model


class Runner(object):
    def __init__(self, model_cls: Callable[[str, dict, pathlib.PosixPath], Base_Model],
            params: dict, model_output_dir: pathlib.PosixPath, run_name: str) -> None:
        self.model_cls = model_cls
        self.params = params
        self.model_output_dir = model_output_dir
        self.run_name = run_name
        self.n_fold = None

    def build_model(self, i_fold):
        run_fold_name = f'{self.run_name}_{i_fold}'
        return self.model_cls(run_fold_name, self.params, self.model_output_dir)

    def train_one_fold(self, i_fold: int, x_trn: pd.DataFrame, y_trn: Union[pd.Series, np.array],
            x_val: pd.DataFrame, y_val: Union[pd.Series, np.array]) -> Tuple[
            Base_Model, float, np.array]:

        model = self.build_model(i_fold)
        model.train(x_trn, y_trn, x_val, y_val)
        val_preds = model.predict(x_val)
        scores = calc_metrics(y_val, val_preds)

        return model, scores, val_preds

    def train_cv(self, x_train: pd.DataFrame, y_train: Union[pd.Series, np.array],
            folds_ids: List[Tuple[np.array]]) -> Tuple[np.array, dict]:
        oof_preds = np.zeros(len(x_train))
        importances = pd.DataFrame(index=x_train.columns)
        cv_score_list = []
        best_iteration = 0
        self.n_fold = len(folds_ids)

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            # Split arrays into train and valid subsets
            x_trn = x_train.iloc[trn_idx]
            y_trn = y_train[trn_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train[val_idx]

            # negative sampling
            import random
            ratio = 2.0
            neg_trn_idx = trn_idx[y_train[trn_idx] == 0]
            pos_trn_idx = trn_idx[y_train[trn_idx] == 1]
            n_new_neg = max(len(neg_trn_idx), int(ratio * len(pos_trn_idx)))
            if len(neg_trn_idx) >= n_new_neg:
                new_neg_trn_idx = pd.Int64Index(random.sample(list(neg_trn_idx), n_new_neg))
                trn_idx = pos_trn_idx.union(new_neg_trn_idx)
                x_trn = x_train.iloc[trn_idx]
                y_trn = y_train[trn_idx]

            # Training
            model, score, val_pred = self.train_one_fold(i_fold, x_trn, y_trn, x_val, y_val)
            oof_preds[val_idx] = val_pred
            cv_score_list.append(score)
            best_iteration += model.get_best_iteration() / len(folds_ids)

            # Get feature importances
            importances_tmp = pd.DataFrame(
                model.get_feature_importance(),
                columns=[f'imp_{i_fold}'], index=x_train.columns
            )
            importances = importances.join(importances_tmp, how='inner')

            # Save model
            model.save_model()

        # Calculate OOF-Score
        oof_score = calc_metrics(y_train, oof_preds)
        print(f"oof: {oof_score}")

        # Summary
        evals_result = {"evals_result": {
            "n_data": len(x_train),
            "n_features": len(x_train.columns),
            "oof_score": oof_score,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
            "best_iteration": best_iteration
        }}
        return oof_preds, evals_result, importances

    def predict_cv(self, x: pd.DataFrame) -> np.array:
        preds_list = []

        for i_fold in range(self.n_fold):
            model = self.build_model(i_fold)
            model.load_model()
            y_pred = model.predict(x)
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
