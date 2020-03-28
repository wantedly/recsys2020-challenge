import pandas as pd
from base import BaseFeature


class EngagedFollowFollowerRatio(BaseFeature):
    def import_columns(self):
        return [
            "engaged_follower_count",
            "engaged_following_count",
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        eps = 1e-2

        # n_follower / n_following
        df_train_features["engaged_follow_follower_ratio"] = (
            df_train_input["engaged_follower_count"] /
            (df_train_input["engaged_following_count"] + eps)
        )
        df_test_features["engaged_follow_follower_ratio"] = (
            df_test_input["engaged_follower_count"] /
            (df_test_input["engaged_following_count"] + eps)
        )

        # n_follower - n_following 
        df_train_features["engaged_follow_follower_diff"] = (
            df_train_input["engaged_follower_count"] -
            df_train_input["engaged_following_count"]
        )
        df_test_features["engaged_follow_follower_diff"] = (
            df_test_input["engaged_follower_count"] -
            df_test_input["engaged_following_count"]
        )

        return df_train_features, df_test_features


if __name__ == "__main__":
    EngagedFollowFollowerRatio.main()
