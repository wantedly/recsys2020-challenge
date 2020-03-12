import os
from typing import Optional
from pathlib import Path

import click
import tensorflow as tf

import preprocessing.rawdata


@click.group()
def cli():
    pass


@cli.group()
def rawdata():
    pass


@rawdata.command()
@click.argument("filepath")
@click.option("-o", "--output", required=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "--project",
    default="wantedly-individual-{}".format(os.environ["USER"]),
    show_default=True,
)
@click.option("--dataset", default="recsys2020", show_default=True)
@click.option("--table", required=True)
def parse_to_gcs(
    filepath: str,
    output: Optional[str],
    verbose: bool,
    project: str,
    dataset: str,
    table: str,
):
    if "{}" not in output:
        print("output must contains {{}}, but got {}".format(repr(output)))
        return 1
    with tf.io.gfile.GFile(filepath, "r") as fp:
        dataframes = preprocessing.rawdata.parse_to_dataframes(fp, verbose=verbose)
        for i, df in enumerate(dataframes):
            with tf.io.gfile.GFile(output.format(i), "w") as w:
                for _, row in df.iterrows():
                    s = row.to_json()
                    w.write(s)
                    w.write("\n")


if __name__ == "__main__":
    cli()
