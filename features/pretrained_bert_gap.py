import os
import tempfile
import argparse
from typing import List, Tuple, Dict, Generator
import threading
import queue

import numpy as np
import pandas as pd
import transformers
import tqdm
import torch
from google.cloud import storage, bigquery, exceptions
from google.cloud import bigquery_storage_v1beta1

from base import BaseFeature

TESTING = False
GCS_BUCKET_NAME = "recsys2020-challenge-wantedly"
PROJECT_ID = "wantedly-individual-naomichi"


def _pad_ids(ids_list: List[List[int]], max_length: int) -> Dict[str, torch.Tensor]:
    input_ids = torch.zeros(len(ids_list), max_length).long()
    attention_mask = torch.zeros(len(ids_list), max_length).float()
    for i, ids in enumerate(ids_list):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1.0
    return dict(input_ids=input_ids, attention_mask=attention_mask)


class PretrainedBertGAP(BaseFeature):
    def __init__(self, batch_size=64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def import_columns(self) -> List[str]:
        ...

    def make_features(
        self, df_train_input: pd.DataFrame, df_test_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    @classmethod
    def add_feature_specific_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--batch_size", type=int, default=64)

    def run(self):
        self._logger.info(f"Running with debugging={self.debugging}")
        if TESTING:
            test_table = f"`{PROJECT_ID}.recsys2020.test`"
        else:
            test_table = f"`{PROJECT_ID}.recsys2020.val_20200418`"
        train_table = f"`{PROJECT_ID}.recsys2020.training`"
        output_table_name = f"{PROJECT_ID}.recsys2020.pretrained_bert_gap"
        if self.debugging:
            output_table_name += "_debug"
        self._extract_features(
            train_table, test_table, output_table_name
        )

    @torch.no_grad()
    def _extract_features(
        self, table_name: str, test_table_name: str, output_table_name: str
    ):
        bqclient = bigquery.Client(project=PROJECT_ID)
        schema = [
            bigquery.schema.SchemaField(name="tweet_id", field_type="STRING", mode="REQUIRED"),
        ] + [
            bigquery.schema.SchemaField(name=f"gap_{i}", field_type="FLOAT64", mode="REQUIRED")
            for i in range(768)
        ]
        try:
            bqclient.get_table(output_table_name)
            if not self.debugging:
                raise RuntimeError(f"Table {output_table_name} already exists.")
        except exceptions.NotFound:
            output_table = bigquery.Table(output_table_name, schema=schema)
            bqclient.create_table(output_table)

        total_rows, iterator = self._read_from_bigquery(bqclient, table_name, test_table_name)
        num_workers = 8
        insert_queue = queue.Queue(maxsize=8)

        pbar = tqdm.tqdm(desc="train", total=total_rows)
        def sender():
            client = bigquery.Client(project=PROJECT_ID)
            output_table = client.get_table(output_table_name)
            while True:
                item = insert_queue.get()
                client.insert_rows_from_dataframe(output_table, item, schema)
                pbar.update(len(item))
                insert_queue.task_done()

        for _ in range(num_workers):
            threading.Thread(target=sender, daemon=True).start()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", do_lowercase=False
        )
        bert = transformers.BertModel.from_pretrained(
            "bert-base-multilingual-cased"
        ).to(device)
        bert.eval()

        input_queue = queue.Queue(maxsize=128)

        @torch.no_grad()
        def embedder():
            while True:
                tweet_ids, target_tokens = input_queue.get()
                max_length = min(512, max(len(tgt) for tgt in target_tokens))
                padded = _pad_ids(target_tokens, max_length=max_length)
                input_ids = padded["input_ids"].to(device)
                attention_mask = padded["attention_mask"].to(device)
                last_hidden_states = bert(input_ids, attention_mask=attention_mask)[0]
                gaps = last_hidden_states.mean(1).cpu().numpy().astype(np.float32)
                embedded = {f"gap_{i}": gaps[:, i] for i in range(768)}
                embedded["tweet_id"] = tweet_ids
                insert_data = pd.DataFrame(embedded)
                insert_queue.put(insert_data)
                input_queue.task_done()

        threading.Thread(target=embedder, daemon=True).start()

        for df in iterator:
            for start in range(0, len(df), self.batch_size):
                tweet_ids = df.tweet_id.values[start : start + self.batch_size]
                target_tokens = df.text_tokens.values[start : start + self.batch_size].tolist()
                input_queue.put((tweet_ids, target_tokens))

        input_queue.join()
        insert_queue.join()

    def _read_from_bigquery(
        self, bqclient: bigquery.Client, table_name: str, test_table_name: str
    ) -> Tuple[int, Generator]:
        query = """
            select tweet_id, any_value(text_tokens) as text_tokens
            from (
                select tweet_id, any_value(text_tokens) as text_tokens
                from {}
                group by tweet_id
                union all
                select tweet_id, any_value(text_tokens) as text_tokens
                from {}
                group by tweet_id
            )
            group by tweet_id
        """.format(
            table_name, test_table_name
        )
        if self.debugging:
            query += " limit 10000"
        result = bqclient.query(query).result()
        return result.total_rows, result.to_dataframe_iterable()


if __name__ == "__main__":
    PretrainedBertGAP.main()
