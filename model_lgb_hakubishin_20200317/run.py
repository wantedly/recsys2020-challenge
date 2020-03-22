import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from src.utils import seed_everything, get_logger, json_dump
from src.feature_loader import FeatureLoader
from src.runner import Runner
from src.models.model_lightgbm import Model_LightGBM


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

    # Create a directory for model output
    model_no = pathlib.Path(args.config).stem
    model_output_dir = \
        pathlib.Path(config['model_dir_name']) /\
        pathlib.Path(config['dataset']['output_directory']) / model_no
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
    for cat in target_columns:
        print(f'============= {cat} =============')

        # Get target values
        y_train = y_train_set[f"TargetCategories_{cat}"].values

        # Get folds
        folds_col = [c for c in folds_train.columns if c.find(cat) != -1]
        assert len(folds_col) == 1, "The number of fold column must be one"
        folds = folds_train[folds_col]
        n_fold = folds.max().values[0]
        folds_ids = []

        print(f"total pos: {y_train.sum()}")
        for i in range(n_fold):
            trn_idx = folds[folds != i+1].dropna().index
            val_idx = folds[folds == i+1].dropna().index
            folds_ids.append((trn_idx, val_idx))
            print(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")
            print(f"  trn_pos={y_train[trn_idx].sum()}, val_pos={y_train[val_idx].sum()}")

        # Train and predict
        model_cls = model_map[config['model']['name']]
        model_params = config['model']
        runner = Runner(
            model_cls, model_params, model_output_dir, f'Train_{model_cls.__name__}_{cat}'
        )
        oof_preds, evals_result, importances = runner.train_cv(
            x_train, y_train, folds_ids, config)

        importances.mean(axis=1).sort_values(ascending=False).reset_index().rename(
            columns={'index': 'feature_name', 0: 'imp'}).to_csv(
            model_output_dir / f'feature_importances_{cat}.csv', header=True, index=False
        )

        evals_result[f"evals_result_{cat}"] = evals_result["evals_result"]
        evals_result.pop("evals_result")
        config.update(evals_result)
        test_preds = runner.predict_cv(x_test)

        # Make submission file
        sub = pd.concat([key_test, pd.Series(test_preds).rename("pred")], axis=1)
        sub = sub[["KeyCategories_tweet_id", "KeyCategories_engaging_user_id", "pred"]]
        sub_file_name = f"{cat}_submission_{config['test_data_type']}.csv"
        sub.to_csv(model_output_dir/ sub_file_name, index=False, header=False)


    # =========================================
    # === Save files
    # =========================================
    save_path = model_output_dir / 'output.json'
    json_dump(config, save_path)


if __name__ == '__main__':
    main()
