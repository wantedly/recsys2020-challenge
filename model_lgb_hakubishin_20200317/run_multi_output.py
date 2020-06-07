import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from src.utils import seed_everything, get_logger, json_dump, upload_to_gcs
from src.feature_loader import FeatureLoader
from src.runner import Runner
from src.models.model_lightgbm import Model_LightGBM
from multiprocessing import cpu_count


seed_everything(71)

model_map = {
    'lightgbm': Model_LightGBM,
}


def main():
    # =========================================
    # === Settings
    # =========================================
    # Get logger
    logger = get_logger(__name__)
    logger.info('Settings')

    # Get argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='model_lgb_hakubishin_20200317/configs/model_0.json')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logger.info(f'config: {args.config}')
    logger.info(f'debug: {args.debug}')

    # Get config
    config = json.load(open(args.config))
    config.update({
        'args': {
            'config': args.config,
            'debug': args.debug
        }
    })
    config["model"]["model_params"]["nthread"] = cpu_count()

    # Create a directory for model output
    model_no = pathlib.Path(args.config).stem
    model_output_dir = (
        pathlib.Path(config['model_dir_name']) /
        pathlib.Path(config['dataset']['output_directory']) / model_no
    )
    if not model_output_dir.exists():
        model_output_dir.mkdir()

    logger.info(f'model_output_dir: {str(model_output_dir)}')
    logger.debug(f'model_output_dir exists: {model_output_dir.exists()}')
    config.update({
        'model_output_dir': str(model_output_dir)
    })

    # =========================================
    # === Loading features
    # =========================================
    logger.info('Loading features')
    logger.info(f'targets: {config["target"]}')
    logger.info(f'features: {config["features"]}')
    logger.info(f'keys: {config["key"]}')
    logger.info(f'folds: {config["folds"]}')

    # features
    x_train = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(config["features"])
    x_test = FeatureLoader(
        data_type=config["test_data_type"], debugging=args.debug
        ).load_features(config["features"])

    # targets
    y_train_set = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(config["target"])

    # keys
    key_test = FeatureLoader(
        data_type=config["test_data_type"], debugging=args.debug
        ).load_features(config["key"])

    # folds
    folds_train = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(config["folds"])

    logger.debug(f'test_data_type: {config["test_data_type"]}')
    logger.debug(f'y_train_set: {y_train_set.shape}')
    logger.debug(f'x_train: {x_train.shape}')
    logger.debug(f'x_test: {x_test.shape}')
    logger.debug(f'key_test: {key_test.shape}')

    # =========================================
    # === Feature processing
    # =========================================
    def rank_gauss(x):
        from scipy.special import erfinv
        N = x.shape[0]
        temp = x.argsort()
        rank_x = temp.argsort() / N
        rank_x -= rank_x.mean()
        rank_x *= 2
        efi_x = erfinv(rank_x)
        efi_x -= efi_x.mean()
        return efi_x

    x_total = x_train.append(x_test).reset_index(drop=True)

    # add na flg
    for col in x_total.columns:
        x_total[f"NAFlg_{col}"] = x_total[col].isnull().astype(int)

    # one-hot encoding
    x_total = pd.get_dummies(x_total, columns=["LabelEncoding"])

    # fillna
    x_total = x_total.fillna(x_total.mean())
    logger.debug(f"the number of na: {x_total.isnull().sum().sum()}")

    # normalize
    not_numeric_feature_classes = [
        "NAFlg",
        "LabelEncoding",
        "CommonFlgFeatures",
        "EngagingUserFollowsEngagedUser",
        "Connected2ndEngagingToEngaged",
        "Connected2ndEngagedToEngaging"
    ]
    not_numeric_features = []
    for not_numeric in not_numeric_feature_classes:
        not_numeric_features += [c for c in x_total.columns if c.find(not_numeric + "_") != -1]
    logger.debug(f"not_numeric_features: {len(not_numeric_features)}")

    numeric_features = [c for c in x_total.columns if c not in not_numeric_features]
    logger.debug(f"numeric_features: {len(numeric_features)}")

    for numeric in numeric_features:
        x_total[numeric] = rank_gauss(x_total[numeric].values)

    x_train = x_total.iloc[:len(y_train_set)]
    x_test = x_total.iloc[len(y_train_set):].reset_index(drop=True)
    logger.debug(f'number of features in train: {x_train.shape}')
    logger.debug(f'number of features in test: {x_test.shape}')


    # =========================================
    # === Train model and predict
    # =========================================
    logger.info('Train model and predict')

    # Modeling
    target_columns = [
        "reply_engagement",
        "retweet_engagement",
        "retweet_with_comment_engagement",
        "like_engagement",
    ]

    # Get target values
    y_train = y_train_set[target_columns].values

    # Get folds
    folds_col = ["StratifiedGroupKFold_retweet_with_comment_engagement"]
    assert len(folds_col) == 1, "The number of fold column must be one"
    folds = folds_train[folds_col]
    n_fold = folds.max().values[0] + 1
    folds_ids = []

    logger.debug(f"total pos: {y_train.sum()}")
    for i in range(n_fold):
        trn_idx = folds[folds != i].dropna().index
        val_idx = folds[folds == i].dropna().index
        folds_ids.append((trn_idx, val_idx))
        logger.debug(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")
        logger.debug(f"{i+1}fold: trn_pos={y_train[trn_idx].sum()}, val_pos={y_train[val_idx].sum()}")

    # Train and predict
    model_cls = model_map[config['model']['name']]
    model_params = config['model']
    runner = Runner(
        model_cls, model_params, model_output_dir, f'Train_{model_cls.__name__}'
    )
    oof_preds, test_preds, evals_result = runner.train_cv(
        x_train, y_train, x_test, folds_ids, config
    )
    config.update(evals_result)

    # Save oof-pred file
    for i, cat in enumerate(target_columns):
        oof_preds_file_name = f"{cat}_oof_pred"
        np.save(model_output_dir / oof_preds_file_name, oof_preds[:, i])
        logger.info(f'Save oof-pred file: {model_output_dir/ oof_preds_file_name}')

    # Make submission file
    for i, cat in enumerate(target_columns):
        sub = pd.concat([key_test, pd.Series(test_preds[:, i]).rename("pred")], axis=1)
        sub = sub[["KeyCategories_tweet_id", "KeyCategories_engaging_user_id", "pred"]]
        sub_file_name = f"{cat}_submission_{config['test_data_type']}.csv"
        sub.to_csv(model_output_dir/ sub_file_name, index=False, header=False)
        logger.info(f'Save submission file: {model_output_dir/ sub_file_name}')

    # Save files (override)
    logger.info('Save files')
    save_path = model_output_dir / 'output.json'
    json_dump(config, save_path)
    logger.info(f'Save model log: {save_path}')

    # =========================================
    # === Upload to GCS
    # =========================================
    if not args.debug:
        logger.info('Upload to GCS')

        bucket_dir_name = config["model_dir_name"] + "/" + model_no
        logger.info(f'bucket_dir_name: {bucket_dir_name}')

        files = list(model_output_dir.iterdir())
        upload_to_gcs(bucket_dir_name, files)


if __name__ == '__main__':
    main()
