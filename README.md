# Financial Portfolio Risk Analysis Dashboard

This project provides a comprehensive risk analysis of a financial portfolio by connecting directly to an Interactive Brokers Trading Workstation or Gateway (or alternatively the default backup_portfolio.csv for demonstration). It uses three industry-standard methodologies: Historical Simulation, GARCH modeling, and a multi-asset Monte Carlo simulation. The analysis is performed from the perspective of a local investor using a base currency (default GBP), correctly incorporating currency risk for all foreign assets and cash balances.

## Features

-   **Live Portfolio Integration**: Connects to a running IBKR TWS or Gateway instance to fetch real-time portfolio positions, including equities, bonds, and multi-currency cash balances.
-   **Backup / Test System**: If the IBKR connection fails, the application automatically falls back to the last successfully fetched portfolio stored in `backup_portfolio.csv`.
-   **Three Core Risk Models**:
    -   **Historical Simulation**: Calculates risk based on the actual historical distribution of returns.
    -   **GARCH**: Models time-varying volatility using a GARCH(1,1) model driven by a Student's t-distribution.
    -   **Monte Carlo Simulation**: Simulates a specified number of future portfolio outcomes based on individual asset implied volatilities calculated from option prices and their historical correlations, using a geometric Brownian motion model.
-   **Core Risk Metrics**:
    -   **Value at Risk (VaR)**: Estimates the maximum potential loss over a given time horizon at a specific confidence level.
    -   **Conditional Value at Risk (CVaR)**: Also known as Expected Shortfall, it measures the expected loss if the VaR threshold is breached.
-   **Multi-Currency Analysis**: Automatically incorporates exchange rate movements into the risk calculation for a base and pair currencies, for GBP, USD, EUR, CAD, CHF, JPY, AUD, HKD and SGD.
-   **Modular and Customizable**: The project is split into logical modules for data gathering, portfolio generation, and risk modeling, making it easy to extend and adapt.

## Requirements

-   Python 3.10+
-   Recommended: An Interactive Brokers account with TWS or Gateway running.
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

1.  **Optional: Start Interactive Brokers TWS or Gateway**:
    -   Log in to your IBKR trading workstation or Gateway.
    -   Ensure the API is enabled. In TWS, go to `File > Global Configuration > API > Settings` and check "Enable ActiveX and Socket Clients".

2.  **Run the Risk Analysis**:
    -   You can configure key parameters (date range, confidence level, time horizon) in the `if __name__ == "__main__"` block of `risk_models.py`.
    -   The base currency can be configured at the top of `risk_models.py`
    -   The IBKR connection details can be configured in the `get_ib_portfolio` function in `ib_connect.py`.
    -   Execute the main script from your terminal:
        ```bash
        python risk_models.py
        ```
    -   The script will connect to IBKR, fetch your portfolio, and then run the three risk models. If it cannot connect, it will use `backup_portfolio.csv`.

## Interpreting the Output
The script will print the VaR and CVaR for the specified time horizon and confidence level for each of the three models.
