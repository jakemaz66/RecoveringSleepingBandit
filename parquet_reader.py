import pandas as pd
import tarfile
import io
import pyarrow
from fastparquet import ParquetFile
import pyarrow.parquet as pq

class DataReader:
    def __init__(self, file_path, chunk_size: int, max_elms: int):

        """This class reads in a parquet file and returns a formatted Pandas dataframe"""
    
        self.df = pd.DataFrame()
        self.chunk_size = chunk_size
        self.max_elms = max_elms
        self.parquet_file = pq.ParquetFile(file_path)


    def read(self):
        counter = 0
        for i in self.parquet_file.iter_batches(batch_size=self.chunk_size):
            i = i.to_pandas()
            self.df = pd.concat([self.df, i], axis=0)
            counter += self.chunk_size

            if counter > self.max_elms:
                break
        mapper = {True: 1, False: 0}
        self.df['session_end_completed'] = self.df['session_end_completed'].map(mapper)
        
        return self.df.reset_index()
    

if __name__ == '__main__':
    import numpy as np

    df = DataReader('data/train1.snappy.parquet', 500, 80000)
    df = df.read()

    df.to_csv('data/traincsv.csv')