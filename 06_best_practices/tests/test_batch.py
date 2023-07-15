import numpy as np
import pandas as pd

from datetime import datetime
from typing import List

def prepare_data(df: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def dt(hour: int, minute: int, second=0) -> datetime:

    return datetime(2022, 1, 1, hour, minute, second)

def test_prepare_data():

    expected_df = pd.read_csv(filepath_or_buffer="expected_df.csv", index_col=0)
    expected_df.PULocationID = expected_df.PULocationID.astype(int)
    expected_df.DOLocationID = expected_df.DOLocationID.astype(int)

    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df_teste = prepare_data(df=df, categorical=['PULocationID', 'DOLocationID'])

    df_teste.PULocationID = df_teste.PULocationID.astype(int)
    df_teste.DOLocationID = df_teste.DOLocationID.astype(int)

    assert np.allclose(np.ones(shape=(3,5)), (df_teste == expected_df).values)
