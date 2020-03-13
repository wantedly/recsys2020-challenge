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
    Dict,
)
import enum
from logging import getLogger
import itertools
from pathlib import Path

import click
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


_logger = getLogger(__name__)


class Column(NamedTuple):
    name: str
    typ: Type
    is_target: bool = False

    def bigquery_schema(self):
        if self.typ == int:
            return dict(name=self.name, type="INT64", mode="REQUIRED")
        if self.typ == str:
            return dict(name=self.name, type="STRING", mode="REQUIRED")
        if self.typ == bool:
            return dict(name=self.name, type="BOOL", mode="REQUIRED")
        if self.typ == List[int]:
            return dict(name=self.name, type="INT64", mode="REPEATED")
        if self.typ == List[str]:
            return dict(name=self.name, type="STRING", mode="REPEATED")
        if self.typ == Optional[int]:
            return dict(name=self.name, type="INT64", mode="NULLABLE")


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
    return {"fields": [x.bigquery_schema() for x in COLUMNS]}


class ParseFn(beam.DoFn):
    def __init__(self, columns: List[Column]) -> None:
        self.columns = columns

    def process(self, element):
        line = element.strip()
        target_columns = self.columns
        columns = line.split("\x01")
        assert len(target_columns) == len(columns) or len(target_columns) - 4 == len(
            columns
        ), "expected {} or {} columns, but got {} columns.".format(
            len(target_columns), len(target_columns) - 4, len(columns)
        )
        values = {}
        for definition, value in itertools.zip_longest(target_columns, columns):
            v: Any = None
            if definition.typ == List[int]:
                v = [int(x) for x in value.split("\t") if x]
            elif definition.typ == List[str]:
                v = [x for x in value.split("\t") if x]
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
            values[definition.name] = v
        yield values


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("input")
@click.argument("table")
@click.option("--project", default="wantedly-individual-naomichi")
@click.argument("pipeline_args", nargs=-1, type=click.UNPROCESSED)
def main(input: str, table: str, project: str, pipeline_args):
    pipeline_args = list(pipeline_args)
    pipeline_args.extend([
        "--runner=DataflowRunner",
        f"--project={project}",
        "--temp_location=gs://recsys2020-challenge-wantedly/temp",
        "--staging_location=gs://recsys2020-challenge-wantedly/staging",
    ])
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:
        lines = p | ReadFromText(input)
        rows = lines | "Parse" >> beam.ParDo(ParseFn(COLUMNS))
        rows | beam.io.gcp.bigquery.WriteToBigQuery(
            table, schema=schema(), project=project,
        )


if __name__ == "__main__":
    main()
