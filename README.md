# recsys2020-challenge

## Setup

- `poetry install`
- あなたのGCPの設定をいかに代入してください
  - `./features/base.py` の
    - `PROJECT_ID`
    - `GCS_BUCKET_NAME`

## Steps

1. Download datasets.
    - Training data
    - Evaluation data
    - Final submission data
2. Upload them to Google Cloud Storage.
3. Insert them to Google BigQuery using Cloud Dataflow.
    - e.g. `python preprocessing/rawdata.py gs://path/to/training.tsv <DATASET NAME>.training --region <REGION> --requirements_file ./dataflow_requirements.txt`
4. Extract tweet texts and save them to BigQuery tables.
    - `./sqls/extract_text_train.sql`
    - `./sqls/extract_text_test.sql`
5. Embed tweet texts using pretrained multilingual BERT.
    - No fine-tuning, just embed tokens and use global average pooling to make fixed length vectors.
    - First, compute a set of tweet texts with `sqls/unnique_texts.sql`.
    - Second, `poetry run python features/pretrained_bert_gap.py`.
6. make features and create models
    - `./workflow.sh`

## Final submission

submission 1

target | output directory name and file name
-- | --
reply_engagement | 2nd_stage_model_1_lr0.01_models5_data1000000/reply_engagement_submission_test.csv
retweet_engagement | 2nd_stage_model_1_lr0.01_models5_data1000000/retweet_engagement_submission_test.csv
retweet_with_comment_engagement | 2nd_stage_model_1_lr0.01_models5_data1000000/retweet_with_comment_engagement_submission_test.csv
like_engagement | 2nd_stage_model_1_lr0.01_models5_data100000/like_engagement_submission_test.csv

submission 2

target | output directory name and file name
-- | --
reply_engagement | wip
retweet_engagement | wip
retweet_with_comment_engagement | wip
like_engagement | wip

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
