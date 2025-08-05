# Financial Portfolio Risk Analysis Dashboard

This project provides a comprehensive risk analysis of a financial portfolio using three industry-standard methodologies: Historical Simulation, eGARCH modeling, and a multi-asset Monte Carlo simulation. It is designed to analyze a portfolio of US equities from the perspective of a UK-based (GBP) investor, incorporating currency risk into the calculations.

## Features

-   **Three Risk Models**:
    -   **Historical Simulation**: Calculates risk based on the actual historical distribution of returns.
    -   **GARCH**: Models time-varying volatility using a GARCH(1,1) model driven by a Student's t-distribution.
    -   **Monte Carlo Simulation**: Simulates a specified number of future portfolio outcomes based on individual asset implied volatilities calculated from option prices and their historical correlations, using a geometric Brownian motion model.
-   **Core Risk Metrics**:
    -   **Value at Risk (VaR)**: Estimates the maximum potential loss over a given time horizon at a specific confidence level.
    -   **Conditional Value at Risk (CVaR)**: Also known as Expected Shortfall, it measures the expected loss if the VaR threshold is breached.
-   **Multi-Currency Analysis**: Automatically incorporates GBP/USD exchange rate movements into the risk calculation for a UK-based portfolio.
-   **Modular and Customizable**: The project is split into logical modules for data gathering, portfolio generation, and risk modeling, making it easy to extend and adapt.

## Requirements

-   Python 3.10+
-   Required packages are listed in `requirements.txt`.

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/bmoss03/Financial-Risk-Dashboard](https://github.com/bmoss03/Financial-Risk-Dashboard)
    ```
2.  Navigate into the project directory:
    ```bash
    cd Financial-Risk-Dashboard
    ```
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Create Your Portfolio**:
    -   Run the `portfolio_creator.py` script to generate a `sp500_tickers_with_weights_test.csv` file with randomly assigned weights for the 20 stocks in the `sp500_tickers_test.csv` file.
        ```bash
        python portfolio_creator.py
        ```
    -   You may choose to edit the `read_from` parameter to `sp500_tickers.csv` to generate a portfolio with weights for all S&P500 stocks, though running the Monte Carlo simulation on this larger dataset may take a while. Ensure if so, that you update `portfolio_file` in `risk_models.py` to `sp500_tickers_with_weights.csv`.
    -   Alternatively, you can manually create your `portfolio.csv` file. It must contain two columns: `tickers` and `weights`. If so, update `portfolio_file` in `risk_models.py` as above.

2.  **Run the Risk Analysis**:
    -   You can configure the parameters (portfolio file, date range, confidence level, etc.) at the top of the `risk_models.py` script.
    -   Execute the main script from your terminal:
        ```bash
        python risk_models.py
        ```

## Interpreting the Output
The script will print the VaR and CVaR for the specified time horizon and confidence level for each of the three models.
