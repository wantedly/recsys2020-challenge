import os
import pandas as pd
import numpy as np
from base import BaseFeature
from encoding_func import target_encoding
from google.cloud import storage, bigquery
from google.cloud import bigquery_storage_v1beta1
from tokenizer_jp import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.pipeline import make_pipeline, make_union


TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"
N_COMPONENTS = 15
RANDOM_SEED = 71


class TextJPFeatureTfidfDecomp(BaseFeature):
    def import_columns(self):
        return [
            "tweet_id",
        ]

    def _read_text_from_bigquery(self, train_text_name: str, test_text_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {train_text_name} and {test_text_name}")
        query = """
                (
                SELECT
                    tweet_id,
                    regexp_replace(regexp_replace(text, r"https?://[w/:%#$&?()~.=+-…]+", ""), r'(@[a-zA-Z0-9_]{{1,15}})', "") as text,
                FROM {}
                WHERE language = '22C448FF81263D4BAF2A176145EE9EAD'   -- japanese
                )
                UNION ALL
                (
                SELECT
                    tweet_id,
                    regexp_replace(regexp_replace(text, r"https?://[w/:%#$&?()~.=+-…]+", ""), r'(@[a-zA-Z0-9_]{{1,15}})', "") as text,
                FROM {}
                WHERE language = '22C448FF81263D4BAF2A176145EE9EAD'   -- japanese
                )
        """.format(train_text_name, test_text_name)
        if self.debugging:
            query += " limit 100000"

        bqclient = bigquery.Client(project=PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def make_features(self, df_train_input, df_test_input):
        # read features
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        train_text = f"`{PROJECT_ID}.recsys2020.texts_training`"
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
            test_text = f"`{PROJECT_ID}.recsys2020.texts_test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
            test_text = f"`{PROJECT_ID}.recsys2020.texts_val_20200418`"

        df_text = self._read_text_from_bigquery(train_text, test_text)

        tk = Tokenizer()

        # tf-idf
        vectorizer = make_pipeline(
            TfidfVectorizer(analyzer=tk.tokenize, min_df=100),
            make_union(
                TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_SEED),
                NMF(n_components=N_COMPONENTS, random_state=RANDOM_SEED),
                n_jobs=1,
            ),
        )
        X = vectorizer.fit_transform(df_text['text']).astype(np.float32)
        X = pd.DataFrame(X, columns=[f'tfidf_svd_{i}' for i in range(N_COMPONENTS)] + [f'tfidf_nmf_{i}' for i in range(N_COMPONENTS)])
        X["tweet_id"] = df_text["tweet_id"]

        df_train_features = pd.merge(df_train_input, X, on="tweet_id", how="left").drop(columns="tweet_id")
        df_test_features = pd.merge(df_test_input, X, on="tweet_id", how="left").drop(columns="tweet_id")

        print(df_train_features.shape)
        print(df_test_features.shape)

        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    TextJPFeatureTfidfDecomp.main()
