import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from src.utils import seed_everything, get_logger, json_dump, upload_to_gcs, download_from_gcs
from io import BytesIO


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
    # === Loading model-outputs
    # =========================================
    
    target_columns = [
        "reply_engagement",
        "retweet_engagement",
        "retweet_with_comment_engagement",
        "like_engagement",
    ]

    models = config["models"]
    test_data_type = config["test_data_type"]

    for target_col in target_columns:
        print(f'============= {target_col} =============')

        oof_pred_values_list = []
        test_pred_values_list = []

        for model in models:
            print(model)

            oof_file_name = f"{target_col}_oof_pred.npy"
            oof_pred = download_from_gcs(
                bucket_dir_name=f"model_lgb_hakubishin_20200317/{model}",
                file_name=oof_file_name
            )
            oof_pred_value = np.load(BytesIO(oof_pred))
            oof_pred_values_list.append(oof_pred_value)

            test_file_name = f"{target_col}_submission_{test_data_type}.csv"
            test_pred = download_from_gcs(
                bucket_dir_name=f"model_lgb_hakubishin_20200317/{model}",
                file_name=test_file_name
            )
            test_pred = pd.read_csv(BytesIO(test_pred), header=None)
            test_pred_value = test_pred.iloc[:, 2].values
            test_pred_values_list.append(test_pred_value)

            print(f"oof mean: {np.mean(oof_pred_value)}, oof shape: {oof_pred_value.shape}")
            print(f"test mean: {np.mean(test_pred_value)}, test shape: {test_pred_value.shape}")

        print("ensemble")
        oof_ensemble_value = np.mean(oof_pred_values_list, axis=0)
        test_ensemble_value = np.mean(test_pred_values_list, axis=0)
        print(f"oof mean: {np.mean(oof_ensemble_value)}, oof shape: {oof_ensemble_value.shape}")
        print(f"test mean: {np.mean(test_ensemble_value)}, test shape: {test_ensemble_value.shape}")

        # Save oof-pred file
        oof_preds_file_name = f"{target_col}_oof_pred"
        np.save(model_output_dir / oof_preds_file_name, oof_ensemble_value)
        logger.info(f'Save oof-pred file: {model_output_dir/ oof_preds_file_name}')

        # Make submission file
        test_pred.loc[:, 2] = test_ensemble_value
        sub_file_name = f"{target_col}_submission_{test_data_type}.csv"
        test_pred.to_csv(model_output_dir/ sub_file_name, index=False, header=False)
        logger.info(f'Save submission file: {model_output_dir/ sub_file_name}')

    # =========================================
    # === Save files
    # =========================================
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

