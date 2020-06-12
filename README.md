# recsys2020-challenge

## Setup

- `poetry install`

## Steps

1. Download datasets.
    - Training data
    - Evaluation data
    - Final submission data
2. Upload them to Google Cloud Storage.
3. Insert them to Google BigQuery using Cloud Dataflow.
    - e.g. `python preprocessing/rawdata.py gs://path/to/training.tsv <DATASET NAME>.training --region <REGION> --requirements_file ./dataflow_requirements.txt`
4. ツイートのテキスト情報をBQに格納する
    - `./hero/hoge_train.sql` 
    - `./hero/hoge_test.sql`
5. Bertのpredictする
    - ????????? わからん
6. make features and create models
    - `./workflow.sh`

## Experiment Configuration

実験の設定はJSONファイルで管理している.

e.g.
```json
{
    "model_dir_name":
        "model_lgb_hakubishin_20200317"
    ,
    "test_data_type":
        "test"
    ,
    "features": [
        "LabelEncoding",
        "CountEncoding",
        "CommonNumericFeatures",
        "CommonFlgFeatures",
        "EngagedFollowFollowerRatio",
        "EngagingFollowFollowerRatio",
        "CountEncodingHashtags",
    ],
    "target": [
        "TargetCategories"
    ],
    "key": [
        "KeyCategories"
    ],
    "folds": [
        "StratifiedGroupKFold"
    ],
    "negative_down_sampling": {
        "enable": true,
        "bagging_size": 1,
        "random_seed": 11
    },
    "random_sampling": {
        "n_data": 20000000,
        "random_seed": 11
    },
    "model": {
        "name": "lightgbm",
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "binary",
            "learning_rate": 0.1,
            "max_depth": 10,
            "num_leaves": 256,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "min_child_weight": 0,
            "seed": 11,
            "bagging_seed": 11,
            "feature_fraction_seed": 11,
            "drop_seed": 11,
            "verbose": -1
        },
        "train_params": {
            "num_boost_round": 10000,
            "early_stopping_rounds": 100,
            "verbose_eval": 500
        }
    },
    "dataset": {
        "input_directory": "data/input/",
        "intermediate_directory": "data/interim/",
        "feature_directory": "data/features/",
        "output_directory": "data/output/"
    }
}
```
