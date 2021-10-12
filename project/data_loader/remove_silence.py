import numpy as np
import pandas as pd

def remove_silence_start_end(x, thresh=0.01):
    """
    Remove silence at the beginning and the end of the signal.

    Args:
        x : Tensor or Numpy array, with dimension [1, N]

    Returns:
        x_without_silence: Tensor or Numpy array, with dimension [1, N'] where N'<=N
    """
    x_without_silence = np.abs(x)-thresh
    x_without_silence[x_without_silence<0] = 0
    cum_aux = np.cumsum(np.squeeze(x_without_silence))
    aux_start = np.where(cum_aux>0)[0]
    aux_end = np.where(cum_aux<cum_aux[-1])[0]
    if len(aux_start)>0:
        start = aux_start[0]
    else:
        start = 0
    if len(aux_end)>0:
        end = aux_end[-1]
    else:
        end = x_without_silence.shape[1]
    x_without_silence = x[:,start:end]
    return x_without_silence

def remove_silence_intervals(x, thresh=0.01, window=360):
    """
    Remove intervals of silence from the signal.
    The intervals of silence are defined as intervals where the signal's value is smaller than the threshold thres
    during an interval of size window.

    Args:
        x : Tensor or Numpy array, with dimension [1, N]

    Returns:
        x_without_silence: Tensor or Numpy array, with dimension [1, N'] where N'<=N
    """
    df = pd.DataFrame({'time': np.arange(x.shape[1]), 'value': x[0,]})
    df['time'] = pd.to_datetime(df['time'])
    #
    df['thresholded_value'] = df['value'].apply(lambda x: x if x>thresh else 0)
    df['rolling_sum'] = df['thresholded_value'].rolling(window=window, center=True, min_periods=1).sum()
    x_without_silence = x[:,df['rolling_sum']>0]
    return x_without_silence