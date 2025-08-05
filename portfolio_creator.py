import pandas as pd
import numpy as np
import random

def generate_random_weights(read_from: str, save_to: str) -> None:
    """
    Generates random weights for each stock in the portfolio.
    The weights are normalized to sum to 1.

    Args:
        read_from (str): Path to the CSV file containing stock tickers.
        save_to (str): Path to save the new CSV file with weights.
    Returns:
        None
    """
    df = pd.read_csv(read_from)
    count = df.count().iloc[0]

    weights = []
    for i in range(count):
        weights.append(random.uniform(0, 1))

    sum_weights = sum(weights)

    for i in range(count):
        weights[i] = weights[i] / sum_weights
        df.loc[i, 'weights'] = weights[i]

    df.to_csv(save_to, index=False)

if __name__ == "__main__":
    read_from = 'sp500_tickers_test.csv'    #test set of 20 tickers, use 'sp500_tickers.csv' for full set
    save_to = 'sp500_tickers_with_weights_test.csv'
    generate_random_weights(read_from, save_to)
    print(f"Random weights generated and saved to {save_to}.")