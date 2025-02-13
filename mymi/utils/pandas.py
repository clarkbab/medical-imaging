import numpy as np
import pandas as pd
from typing import *

def append_row(
    df: pd.DataFrame,
    data: Dict[str, Union[int, float, str]],
    index: Optional[Union[Union[int, str], List[Union[int, str]]]] = None) -> pd.DataFrame:
    # Create new index if necessary.
    if index is not None:
        # Create row index.
        if type(index) == list or type(index) == tuple:
            # Handle multi-indexes.
            index = pd.MultiIndex(levels=[[i] for i in index], codes=[[0] for i in index], names=df.index.names)
        else:
            index = pd.Index(data=[index], name=df.index.name)
    else:
        # Assign index to new row based on existing index.
        max_index = df.index.max()
        if np.isnan(max_index):
            idx = 0
        else:
            idx = max_index + 1
        index = pd.Index(data=[idx], name=df.index.name)

    # Create new dataframe.
    new_df = pd.DataFrame([data], columns=df.columns, index=index)

    # Preserve types when adding to any empty dataframe.
    use_new_df_types = True if len(df) == 0 else False
    
    # Perform concat.
    df = pd.concat((df, new_df), axis=0)

    # Automatic assigning of types to columns will break when we concat with an empty dataframe.
    if use_new_df_types:
        df = df.astype(new_df.dtypes.to_dict())
    
    return df

def concat_dataframes(f: Callable, keys: List[str]) -> pd.DataFrame:
    dfs = [f(k) for k in keys]
    return pd.concat(dfs, axis=0)
