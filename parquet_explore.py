import pyarrow as pa
import pyarrow.parquet as pq
import os
import pandas as pd

def get_first_parquet_from_path(path):
    """This function returns the first parquet file in a directory"""
    for (dir_path, _, files) in os.walk(path):
        for f in files:
            if f.endswith(".parquet"):
                first_pq_path = os.path.join(dir_path, f)
                return first_pq_path
        
def discover_df():
    """This function quickly returns statistics about a parquet file"""
    parquet_file = pq.ParquetFile(first_pq)
    ts=parquet_file.metadata.row_group(0)

    beautiful_df = pd.DataFrame()
    for nm in range(ts.num_columns):
        path_in_schema = ts.column(nm).path_in_schema
        compressed_size = ts.column(nm).total_compressed_size
        stats = ts.column(nm).statistics
        min_value = stats.min
        max_value = stats.max
        physical_type = stats.physical_type
        beautiful_df[path_in_schema] = pd.DataFrame([physical_type, min_value, max_value, compressed_size])
    df = beautiful_df.T
    df.columns = ['DTYPE', 'Min', 'Max', 'Compressed_Size_(KO)']

    return df


if __name__ == '__main__':

    path = r"C:\Users\jakem\Downloads\dataverse\train-part-1"
    first_pq = get_first_parquet_from_path(path)

    first_ds = pq.read_table(first_pq)
    first_ds.num_rows, first_ds.num_columns, first_ds.schema
    first_ds.schema

    df = discover_df()
    df



