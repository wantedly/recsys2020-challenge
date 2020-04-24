python -u features/label_encoding.py
python -u features/count_encoding.py
python -u features/common_numeric_features.py
python -u features/common_flg_features.py
python -u features/engaged_follower_following_ratio.py
python -u features/engaging_follower_following_ratio.py
python -u features/target_encoding.py

python -u features/count_encoding_hashtags.py
python -u features/count_encoding_present_domains.py
python -u features/count_encoding_present_media.py
python -u features/count_encoding_present_links.py
python -u features/time_since_account_was_maded.py

python -u features/engaging_user_follows_engaged_user.py
python -u features/connected_2nd_engaged_to_engaging.py
python -u features/connected_2nd_engaging_to_engaged.py
python -u features/count_encoding_in_tweetid_units.py
python -u features/count_encoding_tweet_type.py
python -u features/count_encoding_crossing.py
python -u features/count_encoding_crossing_2.py

python -u features/at_sign_features.py
python -u features/count_encoding_text_id.py
python -u features/proving_features.py

python -u features/key_categories.py

python -u features/proving_features.py

python -u features/target_encoding_in_tweetid_units.py
python -u model_lgb_hakubishin_20200317/run.py --config model_lgb_hakubishin_20200317/configs/model_35.json
