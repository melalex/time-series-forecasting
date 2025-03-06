import pandas as pd

from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_stock_prices(df: pd.DataFrame, size=(12, 6)) -> None:
    plt.figure(figsize=size)

    for col in [it for it in df.columns if it.endswith("_Price")]:
        plt.plot(df["Date"], df[col], label=col)

    plt.title("Stock prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_stock_volumes(df: pd.DataFrame, size=(12, 6)):
    plt.figure(figsize=size)

    for col in [it for it in df.columns if it.endswith("_Vol.")]:
        plt.plot(df["Date"], df[col], label=col)

    plt.title("Stock volumes")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.show()


def plot_value(series: pd.Series, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(series.index, series, label="value")
    plt.title("Date to value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot(series: pd.Series, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(series.index, series, label="value")
    plt.title("Date to value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_rolling_statistic(series: pd.Series, window: int = 12, size=(12, 6)):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    plt.figure(figsize=size)
    plt.plot(series, label="Original Data")
    plt.plot(rolling_mean, label="Rolling Mean", color="red")
    plt.plot(rolling_std, label="Rolling Std Dev", color="black")
    plt.legend()
    plt.show()


def plot_acf_and_pacf(series: pd.Series, lags=300, size=(12, 6)):
    plt.figure(figsize=size)
    plt.subplot(121)
    plot_acf(series, ax=plt.gca(), lags=lags)
    plt.subplot(122)
    plot_pacf(series, ax=plt.gca(), lags=lags)
    plt.show()


def plot_forecast(test, forecast):
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test, label="Test", color="#01ef63")
    plt.plot(test.index, forecast, label="Forecast", color="orange")
    plt.title("Close Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()
