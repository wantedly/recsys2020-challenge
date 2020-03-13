# recsys2020-challenge

## setup

```bash
poetry install
```

## Dataflow を使って生データを BigQuery に入れるスクリプト

```bash
python preprocessing/rawdata.py gs://recsys2020-challenge-wantedly/training.tsv recsys2020.training --region us-west1 --requirements_file ./dataflow_requirements.txt
```
