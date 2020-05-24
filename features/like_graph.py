from collections import defaultdict
import itertools

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sparse
import tqdm

from base import BaseFeature


class LikeGraph(BaseFeature):
    def import_columns(self):
        return ["engaged_user_id", "engaging_user_id", "engagee_follows_engager", "like_engagement_timestamp is not null as liked"]

    def make_features(self, df_train_input, df_test_input):
        df_folds = self._download_from_gs("StratifiedGroupKFold_training.ftr")["StratifiedGroupKFold_retweet_with_comment_engagement"]
        self._logger.info("Constructing graph...")
        self._make_graph(df_train_input, df_test_input)

        df_train_features = pd.DataFrame({
            "engager_like_out_degree": np.zeros(len(df_train_input), dtype=np.float32),
            "engaged_like_in_degree": np.zeros(len(df_train_input), dtype=np.float32),
            "like_personalized_pagerank": np.zeros(len(df_train_input), dtype=np.float32),
        })
        df_test_features = [
            pd.DataFrame({
                "engager_like_out_degree": np.zeros(len(df_test_input), dtype=np.float32),
                "engaged_like_in_degree": np.zeros(len(df_test_input), dtype=np.float32),
                "like_personalized_pagerank": np.zeros(len(df_test_input), dtype=np.float32),
            })
            for _ in range(len(df_folds.unique()))
        ]

        cv_graphs = []
        for i, fold in enumerate(sorted(df_folds.unique())):
            self._logger.info(f"Start fold {fold}")
            self._logger.info("Constructing subgraph...")
            df = df_train_input[df_folds == fold]
            edges = [
                (u1, u2)
                for u1, u2
                in (
                    (self.user2id.get(u1), self.user2id.get(u2))
                    for u1, u2
                    in zip(df.engaged_user_id, df.engaging_user_id)
                    if u1 is not None and u2 is not None
                )
            ]
            graph = self.G.copy()
            graph.remove_edges_from(edges)

            ## Personalized PageRank
            walk_graph = graph.to_directed(as_view=True)
            ppl_cache = {}
            ppls = defaultdict(lambda: 0)
            for engaging_user_id, engaged_user_id in tqdm.tqdm(itertools.chain(
                zip(df.engaging_user_id.values, df.engaged_user_id.values),
                zip(df_test_input.engaging_user_id.values, df_test_input.engaged_user_id.values),
            ), total=len(df) + len(df_test_input), desc="ppl"):
                engaging_node_id = self.user2id.get(engaging_user_id)
                engaged_node_id = self.user2id.get(engaged_user_id)
                if engaging_node_id is None or engaged_node_id is None:
                    continue
                p = ppl_cache.get(engaging_user_id)
                if p is None:
                    steps = 100
                    jumpback = np.random.random(steps) < 0.15
                    counter = defaultdict(lambda: 0)
                    last_node = engaging_node_id
                    for step in range(steps):
                        counter[last_node] += 1
                        if jumpback[step]:
                            last_node = engaging_node_id
                            continue
                        successors = list(graph.successors(last_node))
                        if not successors:
                            last_node = engaging_node_id
                        else:
                            last_node = successors[np.random.randint(len(successors))]
                    p = counter
                    ppl_cache[engaging_user_id] = p
                ppls[(engaging_user_id, engaged_user_id)] = p.get(engaged_node_id, 0.0)
            df_train_features.loc[df_folds == fold, "like_personalized_pagerank"] = \
                df_train_input.loc[df_folds == fold, ["engaging_user_id", "engaged_user_id"]].apply(lambda r: ppls.get((r.engaging_user_id, r.engaged_user_id), 0.0), axis=1)
            df_test_features[i].loc[:, "like_personalized_pagerank"] = \
                df_test_input.loc[:, ["engaging_user_id", "engaged_user_id"]].apply(lambda r: ppls.get((r.engaging_user_id, r.engaged_user_id), 0.0), axis=1)


            engaging_user_ids = set(df.engaging_user_id.unique()).union(df_test_input.engaging_user_id.unique())
            engaged_user_ids = set(df.engaged_user_id.unique()).union(df_test_input.engaged_user_id.unique())
            engaging_like_out_degrees = {}
            for engaging_user_id in tqdm.tqdm(engaging_user_ids, desc=f"fold {fold} (engaging)"):
                engaging_user_id_int = self.user2id.get(engaging_user_id)
                if engaging_user_id_int is None:
                    continue
                engaging_like_out_degrees[engaging_user_id] = graph.out_degree(engaging_user_id_int)
            df_train_features.loc[df_folds == fold, "engager_like_out_degree"] = df_train_input.loc[df_folds == fold, "engaging_user_id"].apply(lambda x: engaging_like_out_degrees.get(x, 0.0))
            df_test_features[i].loc[:, "engager_like_out_degree"] = df_test_input.loc[:, "engaging_user_id"].apply(lambda x: engaging_like_out_degrees.get(x, 0.0))

            engaged_like_in_degrees = {}
            for engaged_user_id in tqdm.tqdm(engaged_user_ids, desc=f"fold {fold} (engaged)"):
                engaged_user_id_int = self.user2id.get(engaged_user_id)
                if engaged_user_id_int is None:
                    continue
                engaged_like_in_degrees[engaged_user_id] = graph.in_degree(engaged_user_id_int)
            df_train_features.loc[df_folds == fold, "engaged_like_in_degree"] = df_train_input.loc[df_folds == fold, "engaged_user_id"].apply(lambda x: engaged_like_in_degrees.get(x, 0.0))
            df_test_features[i].loc[:, "engaged_like_out_degree"] = df_test_input.loc[:, "engaged_user_id"].apply(lambda x: engaged_like_in_degrees.get(x, 0.0))

        df_test_features_avg = pd.DataFrame()
        for column in df_train_features.columns:
            df_test_features_avg[column] = sum(d[column].values for d in df_test_features) / len(df_test_features)

        return df_train_features, df_test_features_avg


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
        self.G = nx.from_scipy_sparse_matrix(adjacency_matrix, create_using=nx.DiGraph)


if __name__ == "__main__":
    LikeGraph.main()

