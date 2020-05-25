import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage


class TargetEncodingSeq2Dec(BaseFeature):
    def import_columns(self):
        return [
            "engaging_user_id",
            "IF(like_engagement_timestamp is null, 0, 1)\
             +IF(lag(IF(like_engagement_timestamp is null, 0, 1)) over(partition by engaging_user_id order by timestamp) IS NOT NULL, 0.1*lag(IF(like_engagement_timestamp is null, 0, 1)) over(partition by engaging_user_id order by timestamp), 0)\
             +IF(lag(IF(like_engagement_timestamp is null, 0, 1), 2) over(partition by engaging_user_id order by timestamp) IS NOT NULL, 0.01*lag(IF(like_engagement_timestamp is null, 0, 1), 2) over(partition by engaging_user_id order by timestamp), 0)\
             +IF(lag(IF(like_engagement_timestamp is null, 0, 1), 3) over(partition by engaging_user_id order by timestamp) IS NOT NULL, 0.001*lag(IF(like_engagement_timestamp is null, 0, 1), 3) over(partition by engaging_user_id order by timestamp), 0)\
             +IF(lag(IF(like_engagement_timestamp is null, 0, 1), 4) over(partition by engaging_user_id order by timestamp) IS NOT NULL, 0.0001*lag(IF(like_engagement_timestamp is null, 0, 1), 4) over(partition by engaging_user_id order by timestamp), 0) AS like_engagement"
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        folds_train = self._download_from_gs(
            feather_file_name="StratifiedGroupKFold_training.ftr"
        )

        category_columns = [
            "engaging_user_id",
        ]

        target_columns = [
            "like_engagement",
        ]

        for target_col in target_columns:
            print(f'============= {target_col} =============')

            # Get folds
            folds_col = ["StratifiedGroupKFold_retweet_with_comment_engagement"]
            assert len(folds_col) == 1, "The number of fold column must be one"
            folds = folds_train[folds_col]
            n_fold = folds.max().values[0] + 1
            folds_ids = []

            for i in range(n_fold):
                trn_idx = folds[folds != i].dropna().index
                val_idx = folds[folds == i].dropna().index
                folds_ids.append((trn_idx, val_idx))
                print(f"{i+1}fold: n_trn={len(trn_idx)}, n_val={len(val_idx)}")

            for cat_col in category_columns:
                train_result, test_result = target_encoding(
                    cat_col, df_train_input, df_test_input, target_col, folds_ids)
                df_train_input.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_test_input.drop(columns=[f"{cat_col}_ta"], inplace=True)
                df_train_features[f"{target_col}__{cat_col}"] = train_result
                df_test_features[f"{target_col}__{cat_col}"] = test_result

        return df_train_features, df_test_features


if __name__ == "__main__":
    TargetEncodingSeq2Dec.main()
