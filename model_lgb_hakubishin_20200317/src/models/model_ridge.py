import numpy as np
from .model import Base_Model
from sklearn.linear_model import Ridge
from src.utils import Pkl


class Model_Ridge(Base_Model):
    def train(self, x_trn, y_trn, x_val, y_val):
        self.num_fe = x_trn.shape[-1]
        params = self.params["model_params"]
        self.model = Ridge(**params)
        self.model.fit(x_trn, y_trn)
        print(self.model.coef_)

    def predict(self, x):
        pred = self.model.predict(x)
        assert pred >= 0, "predict error: pred < 0"
        assert pred <= 1, "predict error: pred > 1"
        return pred

    def get_feature_importance(self):
        return np.zeros(self.num_fe)

    def get_best_iteration(self):
        return 0

    def save_model(self):
        model_path = self.model_output_dir / f'{self.run_fold_name}.pkl'
        Pkl.dump(self.model, model_path)

    def load_model(self):
        model_path = self.model_output_dir / f'{self.run_fold_name}.pkl'
        self.model = Pkl.load(model_path)
