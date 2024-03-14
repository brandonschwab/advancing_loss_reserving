import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count

def prep_data(data):
    
    df = data.copy()
    
    # Clean column names
    df.columns = [col.replace('Pay_', 'Pay').replace('_true', '').zfill(4) if 'Pay' in col else col for col in df.columns]
    df.rename(columns={'Pay00': 'Pay0'}, inplace=True)
    
    # Bring data into long format
    id_vars = ['ClNr', 'LoB', 'cc', 'AY', 'age', 'inj_part', 'RepDel']
    df = pd.melt(df, id_vars=id_vars, var_name='dev_year', value_name='loss',
                value_vars=[f'Pay{str(i)}' for i in range(12)])

    # Create cumulative values and calculate development period
    df['dev_year'] = df['dev_year'].str.replace('Pay', '').astype(int)
    df = df.sort_values(by=['ClNr', 'dev_year']).reset_index(drop=True)
    df['cum_loss'] = df.groupby('ClNr')['loss'].cumsum()
    df['zero_amount'] = np.where(df['loss'] == 0, 0, 1)

    # Times to integers
    df[["ClNr", "AY", "dev_year"]] = df[["ClNr", "AY", "dev_year"]].astype("int")
    df['ay'] = df['AY'] - 1994

    # Add dev year as a predictor - dev_year_predictor will also be scaled
    dev = df.loc[:, 'dev_year'].copy()
    df['dev_year_predictor'] = dev
    
    return df

def square_to_triangle(df, ay_col='accident_year', time_col='time'):

    data = df.copy(deep=True)

    df_list = []

    min_year = data[ay_col].min()
    max_year = data[ay_col].max()

    # Remove claim developments after evaluation date
    for i in range(min_year, max_year+1):
        df_list.append(data.loc[(data[ay_col] == i) & (data[time_col].isin(range(min_year, max_year+1-i)))])

    return pd.concat(df_list)

def rm_single_claims(df, ay_col='accident_year'):

    data = df.copy(deep=True)
    max_ay = data[ay_col].max()
    data = data.loc[(df[ay_col] != max_ay)]

    return data

def perform_label_encoding(dataframe, columns_to_encode):

    df = dataframe.copy()
    encoders = {}  # To store encoders for each column

    for column in columns_to_encode:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder  # Store encoder for possible future use
    
    return df, encoders

def get_standard_scalers(dataframe, columns_to_encode):

    df = dataframe.copy()
    scalers = {}
    
    for column in columns_to_encode:
        scalers[column] = StandardScaler().fit(df[[column]])

    return scalers

def apply_scaling(dataframe, scalers):

    df = dataframe.copy()

    for column, scaler in scalers.items():
        df[column] = scaler.transform(df[[column]])
    
    return df

def expand_time_series(group):
    #create sub-series for a given time series
    group = group.sort_values("dev_year")
    expanded_rows = []
    for length in range(2, len(group) + 1):
        subset = group.iloc[:length, :]
        subset['ClNr_sub'] = f"{subset['ClNr'].iloc[0]}_{length}"
        expanded_rows.append(subset)
    return pd.concat(expanded_rows, ignore_index=True)

def train_val_split_temporal(df, ay_col='accident_year', id_col='claim_id', train_size=0.8, seed=123):

    np.random.seed(seed)

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # Get unique accident years
    unique_ays = df[ay_col].unique()

    # Stratify based on accident years
    for ay in unique_ays:

        # Filter dataframe for specific accident year
        ay_df = df[df[ay_col] == ay]

        # Get unique claim_ids for this accident year
        claim_ids = ay_df[id_col].unique()
        np.random.shuffle(claim_ids)

        # Split claim_ids for training and validation
        train_idx = int(math.ceil(len(claim_ids) * train_size))
        train_ids = claim_ids[:train_idx]
        val_ids = claim_ids[train_idx:]

        # Append dataframes for training and validation
        train_df = pd.concat([train_df, ay_df[ay_df[id_col].isin(train_ids)]])
        val_df = pd.concat([val_df, ay_df[ay_df[id_col].isin(val_ids)]])

    return train_df, val_df

def create_chunks(df, n_chunks, id='ClNr'):
    grouped = df.groupby(id)
    groups = [group for _, group in grouped]
    chunk_size = len(groups) // n_chunks
    chunks = [pd.concat(groups[i:i + chunk_size]) for i in range(0, len(groups), chunk_size)]
    return chunks

def process_chunk_expand(chunk, id='ClNr'):
    # Apply the operation on the chunk
    return chunk.groupby(id).apply(expand_time_series)

def parallelize_expand_df(df, func, n_cores=None, id='ClNr'):
    if n_cores is None:
        n_cores = 1
    chunks = create_chunks(df, n_cores, id)
    with Pool(n_cores) as pool:
        result = pd.concat(pool.map(func, chunks))
    return result

def calculate_rmse(group, true_col='ln_losses_incurred', pred_col="pred"):
    return root_mean_squared_error(group[true_col], group[pred_col])

