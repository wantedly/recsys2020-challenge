from typing import List, Tuple

from google.cloud import bigquery, bigquery_storage_v1beta1
import pandas as pd

from base import BaseFeature, reduce_mem_usage
import bert_utils


class BertSimilarityBetweenEngagedAndEngagingSurfacingTweetVectorsFeature(BaseFeature):
    # 使わない
    def import_columns(self) -> List[str]:
        ...
    def make_features(
        self, df_train_input: pd.DataFrame, df_test_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def read_and_save_features(
        self,
        train_table_name: str,
        test_table_name: str,
        train_output_path: str,
        test_output_path: str,
    ) -> None:
        df_train_features = self._read_from_bigquery(train_table_name, [train_table_name, test_table_name])
        df_test_features = self._read_from_bigquery(test_table_name, [train_table_name, test_table_name])
        df_train_features.columns = f"{self.name}_" + df_train_features.columns
        df_test_features.columns = f"{self.name}_" + df_test_features.columns

        if self.save_memory:
            self._logger.info("Reduce memory size - train data")
            df_train_features = reduce_mem_usage(df_train_features)
            self._logger.info("Reduce memory size - test data")
            df_test_features = reduce_mem_usage(df_test_features)

        self._logger.info(f"Saving features to {train_output_path}")
        df_train_features.to_feather(train_output_path)
        self._logger.info(f"Saving features to {test_output_path}")
        df_test_features.to_feather(test_output_path)

    def _read_from_bigquery(self, table_name: str, input_tables: List[str]) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = bert_utils.get_similarity_between_engaged_and_engaging_user_using_surfacing_tweets_vector(
            self.PROJECT_ID,
            self.TESTING,
            input_tables,
            table_name
        )
        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=self.PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

if __name__ == "__main__":
    BertSimilarityBetweenEngagedAndEngagingSurfacingTweetVectorsFeature.main()
