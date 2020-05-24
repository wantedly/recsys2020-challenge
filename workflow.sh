# ================================================
# === 特徴量作成/モデリングに必要なファイルの作成
# ================================================
./create_base_files.sh

# ================================================
# === Bert出力を用いた特徴量の作成
# ================================================
./create_bert_features.sh

# ================================================
# === targetに依存しない特徴量の作成
# ================================================
./create_features_independent_on_targets_1.sh
./create_features_independent_on_targets_2.sh
./create_features_independent_on_targets_3.sh

# ================================================
# === targetに依存した特徴量の作成
# ================================================
./create_features_dependent_on_targets.sh

# ================================================
# === 1st stageモデルの作成 & メタ特徴量の作成
# ================================================
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/1st_stage_model_1.json --debug
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/1st_stage_model_2.json --debug
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/1st_stage_model_3.json --debug
python -u model_lgb_hakubishin_20200317/ensemble.py --config model_lgb_hakubishin_20200317/configs/1st_stage_model_ensemble.json --debug

./create_meta_features.sh

# ================================================
# === 2nd stageモデルの作成
# ================================================
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_1.json --debug
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_2.json --debug
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_3.json --debug
python -u model_lgb_hakubishin_20200317/ensemble.py --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_ensemble.json --debug
