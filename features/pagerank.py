import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sparse
from base import BaseFeature


class PageRank(BaseFeature):
    def import_columns(self):
        return ["engaged_user_id", "engaging_user_id", "engagee_follows_engager"]

    def make_features(self, df_train_input, df_test_input):
        self._make_graph(df_train_input, df_test_input)
        pagerank = nx.pagerank_scipy(self.G)

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        def get_pagerank(user):
            user_int = self.user2id.get(user)
            if user_int is None:
                return 0.0
            return pagerank.get(user_int, 0.0)

        for col in ["engaged_user_id", "engaging_user_id"]:
            df_train_features[col] = df_train_input[col].apply(get_pagerank)
            df_test_features[col] = df_test_input[col].apply(get_pagerank)

        return df_train_features, df_test_features

    def _make_graph(self, df_train_input, df_test_input):
        df_train_edges = df_train_input[df_train_input["engagee_follows_engager"]]
        df_test_edges = df_test_input[df_test_input["engagee_follows_engager"]]
        self.users = sorted(
            set(df_train_edges["engaged_user_id"].unique())
            .union(set(df_train_edges["engaging_user_id"].unique()))
            .union(set(df_test_edges["engaged_user_id"].unique()))
            .union(set(df_test_edges["engaging_user_id"].unique()))
        )
        self.user2id = {u: i for i, u in enumerate(self.users)}
        rows = np.concatenate(
            [
                df_train_edges["engaged_user_id"]
                .apply(self.user2id.get)
                .astype(np.int32),
                df_test_edges["engaged_user_id"]
                .apply(self.user2id.get)
                .astype(np.int32),
            ]
        )
        cols = np.concatenate(
            [
                df_train_edges["engaging_user_id"]
                .apply(self.user2id.get)
                .astype(np.int32),
                df_test_edges["engaging_user_id"]
                .apply(self.user2id.get)
                .astype(np.int32),
            ]
        )
        ones = np.ones(len(rows), dtype=np.int32)
        adjacency_matrix = sparse.coo_matrix(
            (ones, (rows, cols)), shape=(len(self.users), len(self.users))
        )
        self.G = nx.from_scipy_sparse_matrix(adjacency_matrix)


if __name__ == "__main__":
    PageRank.main()
