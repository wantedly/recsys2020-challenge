import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from src.utils import seed_everything, get_logger, json_dump
from src.feature_loader import FeatureLoader
from src.get_folds import Fold
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

    target_list = config["target"]
    feature_list = config["features"]
    key_list = config["key"]
    logger.info(f'target: {target_list}')
    logger.info(f'feature: {feature_list}')
    logger.info(f'feature: {key_list}')

    y_train = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(target_list)
    x_train = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(feature_list)
    x_test = FeatureLoader(
        data_type=config["test_data_type"], debugging=args.debug
        ).load_features(feature_list)
    key_test = FeatureLoader(
        data_type=config["test_data_type"], debugging=args.debug
        ).load_features(key_list)

    logger.debug(f'y_train: {y_train.shape}')
    logger.debug(f'x_train: {x_train.shape}')
    logger.debug(f'x_test: {x_test.shape}')
    logger.debug(f'test_data_type: {config["test_data_type"]}')
    logger.debug(f'key_test: {key_test.shape}')


    # =========================================
    # === Train model and predict
    # =========================================
    logger.info('Train model and predict')

    # Get folds
    folds_ids = Fold(
        n_splits=config['cv']['n_splits'],
        shuffle=config['cv']['shuffle'],
        random_state=config['cv']['random_state']
    ).get_kfold(x_train)

    # Modeling
    target_categories_list = y_train.columns
    for cat in target_categories_list:
        print(f'============= {cat} =============')

        # Train and predict
        model_name = config['model']['name']
        model_cls = model_map[model_name]
        params = config['model']
        runner = Runner(model_cls, params, model_output_dir, f'Train_{model_cls.__name__}_{cat}')
        oof_preds, evals_result, importances = runner.train_cv(x_train, y_train[cat].values, folds_ids)

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
