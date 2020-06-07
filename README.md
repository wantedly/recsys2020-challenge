# recsys2020-challenge

## setup

```bash
poetry install

sudo apt-get update
sudo apt-get install -y make g++ libmecab-dev mecab-ipadic-utf8 curl xz-utils mecab git file sudo unzip
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -y
```

## Dataflow を使って生データを BigQuery に入れるスクリプト

```bash
python preprocessing/rawdata.py gs://recsys2020-challenge-wantedly/training.tsv recsys2020.training --region us-west1 --requirements_file ./dataflow_requirements.txt
```


## Steps

1. Download datasets.
2. Upload them to Google Cloud Storage.
3. Insert them to Google BigQuery using Cloud Dataflow.
  - e.g. `python preprocessing/rawdata.py gs://path/to/training.tsv recsys202.training --region <REGION> --requirements_file ./dataflow_requirements.txt`
