import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import transformers
import tqdm
import torch

from base import BaseFeature


class PretrainedBertGAP(BaseFeature):
    def __init__(self, batch_size=64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def import_columns(self):
        return ["text_tokens"]

    @classmethod
    def add_feature_specific_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--batch_size", type=int, default=64)

    @torch.no_grad()
    def make_features(self, df_train_input, df_test_input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", do_lowercase=False
        )
        bert = transformers.BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
        bert.eval()
        train_tokens = df_train_input["text_tokens"].apply(
            lambda x: ",".join(map(str, x))
        )
        test_tokens = df_test_input["text_tokens"].apply(
            lambda x: ",".join(map(str, x))
        )

        unique_tokens = list(set(train_tokens.unique()).union(test_tokens.unique()))
        embeddings = {}  # dict[str, np.ndarray]
        for start in tqdm.trange(0, len(unique_tokens), self.batch_size):
            target = unique_tokens[start : start + self.batch_size]
            target_tokens = [[int(x) for x in tgt.split(",")] for tgt in target]
            max_length = min(512, max(len(tgt) for tgt in target_tokens))
            padded = tokenizer.batch_encode_plus(
                target,
                return_tensors="pt",
                pad_to_max_length=True,
                max_length=max_length,
            )
            input_ids = padded["input_ids"].to(device)
            attention_mask = padded["attention_mask"].to(device)
            last_hidden_states = bert(input_ids, attention_mask=attention_mask)[0]
            for tgt, v in zip(target, last_hidden_states.mean(1).cpu().numpy()):
                embeddings[tgt] = v

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        train_embeddings = np.stack(train_tokens.apply(lambda x: embeddings[x]).values, axis=0)
        test_embeddings = np.stack(test_tokens.apply(lambda x: embeddings[x]).values, axis=0)

        for i in range(768):
            df_train_features[f"dim{i}"] = train_embeddings[:, i]
            df_test_features[f"dim{i}"] = test_embeddings[:, i]

        return df_train_features, df_test_features


if __name__ == "__main__":
    PretrainedBertGAP.main()
