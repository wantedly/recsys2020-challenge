# ================================================
# === 特徴量作成/モデリングに必要なファイルの作成
# ================================================
#python -u features/key_categories.py --debug
#python -u features/target_categories.py --debug
#python -u features/stratified_groupk_fold.py --debug
#python -u features/users_groupk_fold.py --debug


# ================================================
# === targetに依存しない特徴量の作成
# ================================================
#python -u features/label_encoding.py --debug
#python -u features/count_encoding.py --debug
#python -u features/common_numeric_features.py --debug
#python -u features/common_flg_features.py --debug
#python -u features/engaged_follower_following_ratio.py --debug
#python -u features/engaging_follower_following_ratio.py --debug
#python -u features/count_encoding_hashtags.py --debug
#python -u features/count_encoding_present_media.py --debug
#python -u features/count_encoding_present_domains.py --debug
#python -u features/count_encoding_present_links.py --debug
#python -u features/time_since_account_was_maded.py --debug
#python -u features/engaging_user_follows_engaged_user.py --debug
#python -u features/connected_2nd_engaged_to_engaging.py --debug
#python -u features/connected_2nd_engaging_to_engaged.py --debug
#python -u features/count_encoding_in_tweetid_units.py --debug
#python -u features/count_encoding_tweet_type.py --debug
#python -u features/count_encoding_crossing.py --debug
#python -u features/count_encoding_crossing_2.py --debug
#python -u features/at_sign_features.py --debug
#python -u features/count_encoding_text_id.py --debug


# ================================================
# === targetに依存した特徴量の作成
# ================================================
#python -u features/target_encoding.py --debug


# ================================================
# === 1st stageモデルの作成 & メタ特徴量の作成
# ================================================
#python -u model_lgb_hakubishin_20200317/run.py --debug --config model_lgb_hakubishin_20200317/configs/model_1st_stage_1.json
#python -u model_lgb_hakubishin_20200317/run.py --debug --config model_lgb_hakubishin_20200317/configs/model_1st_stage_2.json
#python -u model_lgb_hakubishin_20200317/run.py --debug --config model_lgb_hakubishin_20200317/configs/model_1st_stage_3.json

#python -u model_lgb_hakubishin_20200317/ensemble.py --debug --config model_lgb_hakubishin_20200317/configs/model_1st_stage_ensemble.json
#python -u features/meta_features.py --debug
#python -u features/meta_features_agg_by_engaged_user_id.py --debug
#python -u features/meta_features_agg_by_engaging_user_id.py --debug
#python -u features/meta_features_agg_by_tweet_id.py --debug


# ================================================
# === 2nd stageモデルの作成 & メタ特徴量の作成
# ================================================
python -u model_lgb_hakubishin_20200317/run.py --debug --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_1.json
python -u model_lgb_hakubishin_20200317/run.py --debug --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_2.json
python -u model_lgb_hakubishin_20200317/run.py --debug --config model_lgb_hakubishin_20200317/configs/2nd_stage_model_3.json

#python -u model_lgb_hakubishin_20200317/ensemble.py --debug --config model_lgb_hakubishin_20200317/configs/model_2nd_stage_ensemble.json
