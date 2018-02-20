import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_hdf5_to_parquet(h5_file, h5key, parquet_file, chunksize=100000):

    stream = pd.read_hdf(h5_file, h5key, chunksize=chunksize)

    for i, chunk in enumerate(stream):
        print("Chunk {}".format(i))

        if i == 0:
            # Infer schema and open parquet file on first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')

        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)

    parquet_writer.close()


h5file = 'datasets/store1.h5'
h5key = 'gallons_per_hour'
pqfile = 'datasets/store1.parquet'

convert_hdf5_to_parquet(h5file, h5key, pqfile)
