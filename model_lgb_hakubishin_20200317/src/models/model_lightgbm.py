import lightgbm as lgb
from .model import Base_Model
from src.utils import Pkl


class Model_LightGBM(Base_Model):
    def train(self, x_trn, y_trn, x_val, y_val):
        validation_flg = x_val is not None

        # Setting datasets
        d_trn = lgb.Dataset(x_trn, label=y_trn)
        if validation_flg:
            d_val = lgb.Dataset(x_val, label=y_val, reference=d_trn)

        # Setting model parameters
        lgb_model_params = self.params['model_params']
        lgb_train_params = self.params['train_params']

        # Training
        if validation_flg:
            self.model = lgb.train(
                params=lgb_model_params,
                train_set=d_trn,
                valid_sets=[d_trn, d_val],
                valid_names=['train', 'valid'],
                **lgb_train_params
            )
        else:
            self.model = lgb.train(
                params=lgb_model_params,
                train_set=d_trn,
                **lgb_train_params
            )

    def predict(self, x):
        return self.model.predict(x)

    def get_feature_importance(self):
        return self.model.feature_importance(importance_type='gain')

    def get_best_iteration(self):
        return self.model.best_iteration

    def save_model(self):
        model_path = self.model_output_dir / f'{self.run_fold_name}.pkl'
        Pkl.dump(self.model, model_path)

    def load_model(self):
        model_path = self.model_output_dir / f'{self.run_fold_name}.pkl'
        self.model = Pkl.load(model_path)
