import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def get_mean_interval(df: pd.DataFrame, date_column="timestamp") -> pd.Timedelta:
    """
    Calculates the mean time interval between consecutive timestamps in a DataFrame.
    """
    diffs = df[date_column].diff().dropna()
    return diffs.mean()


def get_median_interval(df: pd.DataFrame, date_column="timestamp") -> pd.Timedelta:
    """
    Calculates the median time interval between consecutive timestamps in a DataFrame.
    """
    diffs = df[date_column].diff().dropna()
    return diffs.median()


def plot_two_ts(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    time_col: str,
    value_col: str,
    label1: str,
    label2: str,
    title: str,
) -> None:
    """
    Plots two time series on the same graph for comparison.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(df1[time_col], df1[value_col], label=label1)
    plt.plot(df2[time_col], df2[value_col], label=label2)
    plt.xlabel("Time")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.show()


def get_bland_altman_plot(y_true, y_test) -> None:
    """
    Generates a Bland-Altman plot to compare two sets of measurements.
    """
    f, ax = plt.subplots(1, figsize=(8, 5))
    sm.graphics.mean_diff_plot(y_true, y_test, ax=ax)
    plt.show()


def get_mean_error(y_true, y_test) -> float:
    """
    Calculates the mean error (bias) between two sets of measurements.
    """
    return np.mean(y_test - y_true)


def get_mae(y_true, y_test) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between two sets of measurements.
    """
    return np.mean(np.abs(y_test - y_true))


def get_reliability(y_true, y_test, threshold=10) -> float:
    """
    Calculates the reliability, defined as the percentage of measurements within a specified threshold.
    """
    return np.mean(np.abs(y_test - y_true) <= threshold)


def get_cross_correlation_lag(
    benchmark_df: pd.DataFrame, test_df: pd.DataFrame, sampling_rate_hz=25
) -> float:
    """
    Calculate the time lag between two time series using cross-correlation.
    """

    # Create an interpolation function based on the benchmark data.
    interp_func = interp1d(
        benchmark_df.index.astype(np.int64) // 10**9,  # x-values as unix seconds
        benchmark_df["hr"],  # y-values
        kind="cubic",  # interpolation method
        bounds_error=False,  # Don't throw error if out of bounds
        fill_value="extrapolate",  # Extrapolate for edges
    )

    # Use this function to create the temporary upsampled benchmark signal
    # at the same timestamps as the test signal.
    benchmark_upsampled = pd.Series(
        interp_func(test_df.index.astype(np.int64) // 10**9),
        index=test_df.index,
        name="benchmark_upsampled",
    )

    # We correlate the test signal with the temporary upsampled benchmark.
    signal1 = test_df["hr"].dropna()
    signal2 = benchmark_upsampled.loc[
        signal1.index
    ].dropna()  # Align indices before correlating

    correlation = np.correlate(
        signal1 - signal1.mean(), signal2 - signal2.mean(), mode="full"
    )

    lag_in_samples = np.argmax(correlation) - (len(signal2) - 1)

    sampling_period = 1.0 / sampling_rate_hz  # seconds
    time_lag_seconds = lag_in_samples * sampling_period

    return time_lag_seconds
