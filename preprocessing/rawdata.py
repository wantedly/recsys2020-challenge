from typing import (
    List,
    Optional,
    NamedTuple,
    Type,
    Callable,
    Iterable,
    Any,
    DefaultDict,
    Generator,
)
import enum
from logging import getLogger
import itertools

import pandas as pd
from google.cloud import bigquery
import numpy as np


_logger = getLogger(__name__)


class Column(NamedTuple):
    name: str
    typ: Type
    is_target: bool = False

    def bigquery_schema(self):
        if self.typ == int:
            return bigquery.SchemaField(self.name, "INT64", mode="REQUIRED")
        if self.typ == str:
            return bigquery.SchemaField(self.name, "STRING", mode="REQUIRED")
        if self.typ == bool:
            return bigquery.SchemaField(self.name, "BOOL", mode="REQUIRED")
        if self.typ == List[int]:
            return bigquery.SchemaField(self.name, "INT64", mode="REPEATED")
        if self.typ == List[str]:
            return bigquery.SchemaField(self.name, "STRING", mode="REPEATED")
        if self.typ == Optional[int]:
            return bigquery.SchemaField(self.name, "INT64")


COLUMNS = [
    Column("text_tokens", List[int]),
    Column("hashtags", List[str]),
    Column("tweet_id", str),
    Column("present_media", List[str]),
    Column("present_links", List[str]),
    Column("present_domains", List[str]),
    Column("tweet_type", str),
    Column("language", str),
    Column("timestamp", int),
    Column("engaged_user_id", str),
    Column("engaged_follower_count", int),
    Column("engaged_following_count", int),
    Column("engaged_is_verified", bool),
    Column("engaged_account_creation_time", int),
    Column("engaging_user_id", str),
    Column("engaging_follower_count", int),
    Column("engaging_following_count", int),
    Column("engaging_is_verified", bool),
    Column("engaging_account_creation_time", int),
    Column("engagee_follows_engager", bool),
    Column("reply_engagement_timestamp", Optional[int]),
    Column("retweet_engagement_timestamp", Optional[int]),
    Column("retweet_with_comment_engagement_timestamp", Optional[int]),
    Column("like_engagement_timestamp", Optional[int]),
]


def schema():
    return [x.bigquery_schema() for x in COLUMNS]


def parse_to_dataframes(
    lines: Iterable[str], verbose: bool = False, chunk_size: int = 2 ** 20
) -> Generator[pd.DataFrame, None, None]:
    if verbose:
        try:
            import tqdm

            lines = tqdm.tqdm(lines, desc="Read and parsing")
        except ImportError:
            _logger.warning("`tqdm` is missing")

    chunks: List[pd.DataFrame] = []
    values: DefaultDict[str, List[Any]] = DefaultDict(list)

    for ith, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        columns = line.split("\x01")
        assert len(COLUMNS) == len(columns) or len(COLUMNS) - 4 == len(
            columns
        ), "expected {} or {} columns, but got {} columns.".format(
            len(COLUMNS), len(COLUMNS) - 4, len(columns)
        )
        for definition, value in itertools.zip_longest(COLUMNS, columns):
            try:
                v: Any = None
                if definition.typ == List[int]:
                    v = np.asarray(
                        [int(x) for x in value.split("\t") if x], dtype=np.int32
                    )
                elif definition.typ == List[str]:
                    v = np.asarray([x for x in value.split("\t") if x], dtype=str)
                elif definition.typ == int:
                    v = int(value)
                elif definition.typ == str:
                    v = value
                elif definition.typ == bool:
                    if value == "true":
                        v = True
                    elif value == "false":
                        v = False
                    else:
                        raise RuntimeError(
                            "value of boolean column {} must be either `true` or `false`, but {}".format(
                                definition.name, value
                            )
                        )
                elif definition.typ == Optional[int]:
                    if not value:
                        v = None
                    else:
                        v = int(value)
                else:
                    raise RuntimeError(
                        "unsupported value type ({} is of type {})".format(
                            definition.name, definition.typ
                        )
                    )
                values[definition.name].append(v)
            except RuntimeError as ex:
                raise RuntimeError(
                    "error ({}) occurs while processing {}th {}".format(
                        ex, ith, definition.name
                    )
                )
        if (ith + 1) % chunk_size == 0:
            yield pd.DataFrame(values).convert_dtypes()
            values = DefaultDict(list)
    if len(values["text_tokens"]) == 0:
        yield pd.DataFrame(values).convert_dtypes()
