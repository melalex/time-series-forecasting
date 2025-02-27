from matplotlib import pyplot as plt
import pandas as pd


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
