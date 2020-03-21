import pandas as pd

from features.base import BaseFeature


class EngagedFollowerFollowingRatio(BaseFeature):
    def import_columns(self):
        return ["engaged_follower_count", "engaged_following_count"]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame(
            {
                "engaged_follow_follower_ratio": df_train_input[
                    "engaged_follower_count"
                ].values
                / (df_train_input["engaged_following_count"] + 1e-5)
            }
        )
        df_test_features = pd.DataFrame(
            {
                "engaged_follow_follower_ratio": df_test_input[
                    "engaged_follower_count"
                ].values
                / (df_test_input["engaged_following_count"] + 1e-5)
            }
        )
        return df_train_features, df_test_features


if __name__ == "__main__":
    EngagedFollowFollowerRatio.main()
