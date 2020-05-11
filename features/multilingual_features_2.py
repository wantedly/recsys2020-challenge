import os
import pandas as pd
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


class MultilingualFeatures2(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(
            self, train_table_name: str, test_table_name: str, read_table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {read_table_name}")
        query = """
                with multilingual_features as (
                    select
                        engaging_user_id,
                        languages[ordinal(1)] first_lan,
                        n_lans[ordinal(1)]  n_first_lan,
                        n_lans[ordinal(1)] / n_all first_lan_rate,
                        case when array_length(n_lans) > 1 then languages[ordinal(2)] else null end second_lan,
                        case when array_length(n_lans) > 1 then n_lans[ordinal(2)] else null end n_second_lan,
                        case when array_length(n_lans) > 1 then n_lans[ordinal(2)] / n_all else null end second_lan_rate,
                        case when array_length(n_lans) > 2 then languages[ordinal(3)] else null end third_lan,
                        case when array_length(n_lans) > 2 then n_lans[ordinal(3)] else null end n_third_lan,
                        case when array_length(n_lans) > 2 then n_lans[ordinal(3)] / n_all else null end third_lan_rate,
                    from
                        (
                        select
                            engaging_user_id,
                            array_agg(language order by n_lan desc) languages,
                            array_agg(n_lan order by n_lan desc) n_lans,
                            sum(n_lan) n_all,
                        from
                        (
                        select
                            engaging_user_id, language, count(1) n_lan,
                        from
                        (
                        select
                            engaging_user_id, tweet_id, language,
                        from
                          (
                            (
                            SELECT engaging_user_id, tweet_id, language
                            FROM {}
                            )
                            UNION ALL
                            (
                            SELECT engaging_user_id, tweet_id, language
                            FROM {}
                            )
                          )
                        group by
                          engaging_user_id, tweet_id, language
                        )
                        group by
                          engaging_user_id, language
                        )
                        group by
                          engaging_user_id
                        )
                  )
                SELECT
                -- A.engaged_user_id,
                -- A.engaging_user_id,
                A.language,
                B.* EXCEPT(engaging_user_id),
                CASE
                  WHEN A.language = B.first_lan THEN 1
                  WHEN A.language = B.second_lan AND B.n_first_lan = B.n_second_lan THEN 1
                  WHEN A.language = B.third_lan AND B.n_first_lan = B.n_third_lan THEN 1
                  ELSE 0
                END AS main_language,
                FROM {} AS A
                LEFT OUTER JOIN multilingual_features AS B
                ON A.engaging_user_id = B.engaging_user_id
                ORDER BY A.tweet_id, A.engaging_user_id
        """.format(train_table_name, test_table_name, read_table_name)

        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def make_features(self, df_train_input, df_test_input):
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        label_train = self._download_from_gs(
            feather_file_name="LabelEncoding_training.ftr"
        )

        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
            label_test = self._download_from_gs(
                feather_file_name="LabelEncoding_test.ftr"
            )
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
            label_test = self._download_from_gs(
                feather_file_name="LabelEncoding_val_20200418.ftr"
            )

        df_train_features = self._read_features_from_bigquery(train_table, test_table, train_table)
        df_test_features = self._read_features_from_bigquery(train_table, test_table, test_table)

        label_keys = ["language", "LabelEncoding_language"]

        label_train["language"] = df_train_features.language
        df_train_features.drop("language", axis=1, inplace=True)
        label_train = label_train[~label_train.duplicated(subset=label_keys)][label_keys]

        label_test["language"] = df_test_features.language
        df_test_features.drop("language", axis=1, inplace=True)
        label_test = label_test[~label_test.duplicated(subset=label_keys)][label_keys]

        language_label = pd.concat([label_train, label_test])
        language_label = language_label[~language_label.duplicated(subset=label_keys)][label_keys]
        max_label = language_label.LabelEncoding_language.max()
        language_label = dict(zip(language_label.language, language_label.LabelEncoding_language))

        labeling_targets = ["first_lan", "second_lan", "third_lan"]
        for labeling_target in labeling_targets:
            df_train_features[labeling_target] = df_train_features[labeling_target].map(language_label).fillna(max_label+1).astype(int)
            df_test_features[labeling_target] = df_test_features[labeling_target].map(language_label).fillna(max_label+1).astype(int)

        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    MultilingualFeatures2.main()
