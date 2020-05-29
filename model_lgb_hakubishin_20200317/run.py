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

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

    drop_features = [
        "FFFeatures2_engaging_sum_engaging_following_count",
        "CountEncodingHashtags_max_value",
        "CountEncoding_engaging_user_id",
        "CountEncodingTweetType_ratio_retweet_tweet",
        "CountEngagingTweetWithinNDifference_diff_8hours_divided_by_max_abs",
        "BertSimilarityBetweenTweetAndEngagingSurfacingTweetVectorsFeature_f0_",
        "CountEngagingTweetWithinN_cnt_within_8hours_earlier",
        "TweetUrlFeatures_engaged_engaging_avg_n_url",
        "CountEngagingTweetWithinN_cnt_within_24hours_earlier",
        "TweetUrlFeatures_engaging_engaged_only_url_rate",
    ]

    x_train.drop(drop_features, axis=1, inplace=True)
    x_test.drop(drop_features, axis=1, inplace=True)

    logger.debug(f'test_data_type: {config["test_data_type"]}')
    logger.debug(f'y_train_set: {y_train_set.shape}')
    logger.debug(f'x_train: {x_train.shape}')
    logger.debug(f'x_test: {x_test.shape}')
    logger.debug(f'key_test: {key_test.shape}')

    # =========================================
    # === Adversarial Validation
    # =========================================
    feature_name = x_test.columns
    logger.info("adversarial validation")
    train_adv = x_train
    test_adv = x_test
    train_adv['target'] = 0
    test_adv['target'] = 1
    train_test_adv = pd.concat([train_adv, test_adv], axis=0, sort=False).reset_index(drop=True)
    target = train_test_adv['target'].values

    train_set, val_set = train_test_split(train_test_adv, test_size=0.33, random_state=71, shuffle=True)
    x_train_adv = train_set[feature_name]
    y_train_adv = train_set['target']
    x_val_adv = val_set[feature_name]
    y_val_adv = val_set['target']
    logger.debug(f'the number of train set: {len(x_train_adv)}')
    logger.debug(f'the number of valid set: {len(x_val_adv)}')

    train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
    val_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)
    lgb_model_params = config["adversarial_validation"]["lgb_model_params"]
    lgb_train_params = config["adversarial_validation"]["lgb_train_params"]
    clf = lgb.train(
        lgb_model_params,
        train_lgb,
        valid_sets=[train_lgb, val_lgb],
        valid_names=['train', 'valid'],
        **lgb_train_params
    )

    feature_imp = pd.DataFrame(
        sorted(zip(clf.feature_importance(importance_type='gain'), feature_name)), columns=['value', 'feature']
    )
    feature_imp.to_csv(model_output_dir / f'feature_importances_adversarial_validatio.csv', header=True, index=False)
    plt.figure(figsize=(20, 10))
    sns.barplot(x='value', y='feature', data=feature_imp.sort_values(by='value', ascending=False).head(20))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig(model_output_dir / "feature_importance_adv.png")

    config.update({
        'adversarial_validation_result': {
            'score': clf.best_score,
            'feature_importances': feature_imp.set_index("feature").sort_values(by="value", ascending=False).head(20).to_dict()["value"]
        }
    })

    # # =========================================
    # # === Train model and predict
    # # =========================================
    # logger.info('Train model and predict')

    # # Modeling
    # target_columns = [
    #     "reply_engagement",
    #     "retweet_engagement",
    #     "retweet_with_comment_engagement",
    #     "like_engagement",
    # ]
    # for cat in target_columns:
    #     logger.info(f'============= {cat} =============')

    #     # Get target values
    #     y_train = y_train_set[f"TargetCategories_{cat}"].values

    #     # Get folds
    #     folds_col = ["StratifiedGroupKFold_retweet_with_comment_engagement"]
    #     assert len(folds_col) == 1, "The number of fold column must be one"
    #     folds = folds_train[folds_col]
    #     n_fold = folds.max().values[0] + 1
    #     folds_ids = []

    #     logger.debug(f"total pos: {y_train.sum()}")
    #     for i in range(n_fold):
    #         trn_idx = folds[folds != i].dropna().index
    #         val_idx = folds[folds == i].dropna().index
    #         folds_ids.append((trn_idx, val_idx))
    #         logger.debug(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")
    #         logger.debug(f"{i+1}fold: trn_pos={y_train[trn_idx].sum()}, val_pos={y_train[val_idx].sum()}")

    #     # Train and predict
    #     model_cls = model_map[config['model']['name']]
    #     model_params = config['model']
    #     runner = Runner(
    #         model_cls, model_params, model_output_dir, f'Train_{model_cls.__name__}_{cat}'
    #     )
    #     oof_preds, evals_result, importances = runner.train_cv(
    #         x_train, y_train, folds_ids, config)

    #     # Save importance
    #     importances.mean(axis=1).sort_values(ascending=False).reset_index().rename(
    #         columns={'index': 'feature_name', 0: 'imp'}).to_csv(
    #         model_output_dir / f'feature_importances_{cat}.csv', header=True, index=False
    #     )

    #     evals_result[f"evals_result_{cat}"] = evals_result["evals_result"]
    #     evals_result.pop("evals_result")
    #     config.update(evals_result)
    #     test_preds = runner.predict_cv(x_test)

    #     # Save oof-pred file
    #     oof_preds_file_name = f"{cat}_oof_pred"
    #     np.save(model_output_dir / oof_preds_file_name, oof_preds)
    #     logger.info(f'Save oof-pred file: {model_output_dir/ oof_preds_file_name}')

    #     # Make submission file
    #     sub = pd.concat([key_test, pd.Series(test_preds).rename("pred")], axis=1)
    #     sub = sub[["KeyCategories_tweet_id", "KeyCategories_engaging_user_id", "pred"]]
    #     sub_file_name = f"{cat}_submission_{config['test_data_type']}.csv"
    #     sub.to_csv(model_output_dir/ sub_file_name, index=False, header=False)
    #     logger.info(f'Save submission file: {model_output_dir/ sub_file_name}')

    #     # Save files (override)
    #     logger.info('Save files')
    #     save_path = model_output_dir / 'output.json'
    #     json_dump(config, save_path)
    #     logger.info(f'Save model log: {save_path}')

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
