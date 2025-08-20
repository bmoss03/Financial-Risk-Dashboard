import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t
import gather_financial_data as gfd
import ib_connect as ibc

TRADING_DAYS_PER_YEAR = 252
BASE_CURRENCY = 'GBP'
RISK_FREE_RATE = 0.04101

def calculate_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates logarithmic returns for a given DataFrame of stock prices.
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices.
    Returns:
        pd.DataFrame: DataFrame with logarithmic returns.
    """
    processed_df = price_df.ffill().bfill()
    log_returns_df = np.log(processed_df / processed_df.shift(1))
    return log_returns_df.dropna()

def prepare_portfolio_data(portfolio: pd.DataFrame, start_date: str, end_date: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Prepares portfolio data, calculates weights, and computes historical returns in the base currency.

    Args:
        portfolio (pd.DataFrame): Portfolio DataFrame from ib_connect.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Historical daily portfolio returns in the base currency.
            - pd.Series: Weights of each asset.
            - pd.DataFrame: The portfolio DataFrame with calculated weights and yfinance tickers.
    """
    # Fetch stock and currency data
    equity_portfolio = portfolio[portfolio['asset_class'] == 'EQUITY']
    yfinance_equity_tickers = equity_portfolio['yfinance_ticker'].unique().tolist()

    foreign_currencies = portfolio[portfolio['currency'] != BASE_CURRENCY]['currency'].unique().tolist()
    fx_tickers = [f"{currency}{BASE_CURRENCY}=X" for currency in foreign_currencies if currency != BASE_CURRENCY]

    equity_prices = gfd.fetch_adj_close_prices(yfinance_equity_tickers, start_date, end_date)
    fx_prices = gfd.fetch_currency_data(fx_tickers, start_date, end_date)

    # Create mapping from yfinance ticker to original ticker
    ticker_map = pd.Series(equity_portfolio.ticker.values, index=equity_portfolio.yfinance_ticker).to_dict()
    equity_prices.rename(columns=ticker_map, inplace=True)

    equity_log_returns = calculate_log_returns(equity_prices)
    fx_log_returns = calculate_log_returns(fx_prices)

    # Calculate portfolio weights in base currency
    if not fx_prices.empty:
        latest_fx_rates = fx_prices.iloc[-1]
        portfolio['cash_amount_gbp'] = portfolio.apply(
            lambda row: row['cash_amount'] / latest_fx_rates[f"{row['currency']}{BASE_CURRENCY}"] if row['currency'] != BASE_CURRENCY and f"{row['currency']}{BASE_CURRENCY}" in latest_fx_rates else row['cash_amount'],
            axis=1
        )
    else:
        portfolio['cash_amount_gbp'] = portfolio['cash_amount']
        if any(portfolio['currency'] != BASE_CURRENCY):
            print("Warning: Using original cash amounts for weighting due to missing FX data.")

    total_portfolio_value_gbp = portfolio['cash_amount_gbp'].sum()
    portfolio['weight'] = portfolio['cash_amount_gbp'] / total_portfolio_value_gbp
    weights = portfolio.set_index('ticker')['weight']

    # Calculate historical portfolio returns in base currency (GBP)
    all_asset_returns_gbp = pd.DataFrame()
    if not equity_log_returns.empty:
        all_asset_returns_gbp = equity_log_returns.copy()

    # Convert equity returns to GBP
    for ticker in equity_log_returns.columns:
        asset_currency = portfolio.loc[portfolio['ticker'] == ticker, 'currency'].iloc[0]
        if asset_currency != BASE_CURRENCY:
            fx_col_name = f"{asset_currency}{BASE_CURRENCY}"
            if fx_col_name in fx_log_returns.columns:
                all_asset_returns_gbp[ticker] = all_asset_returns_gbp[ticker].add(fx_log_returns[fx_col_name], fill_value=0)

    # Add cash returns 
    cash_holdings = portfolio[portfolio['asset_class'] == 'CASH']
    for _, row in cash_holdings.iterrows():
        cash_ticker = row['ticker']
        if cash_ticker != BASE_CURRENCY:
            fx_col_name = f"{cash_ticker}{BASE_CURRENCY}"
            if fx_col_name in fx_log_returns.columns:
                all_asset_returns_gbp[cash_ticker] = fx_log_returns[fx_col_name]
        else:
            # Base currency cash has zero return
            all_asset_returns_gbp[cash_ticker] = 0.0

    # Calculate total portfolio return series
    all_asset_returns_gbp.fillna(0, inplace=True)
    aligned_weights = weights.reindex(all_asset_returns_gbp.columns).fillna(0)
    portfolio_returns_gbp = all_asset_returns_gbp.dot(aligned_weights)

    return portfolio_returns_gbp, weights, portfolio


def calculate_historical_risk(portfolio_returns: pd.Series, confidence_level: float, time_horizon_days: int) -> dict:
    """
    Calculates VaR and CVaR using historical simulation for a portfolio of US stocks
    denominated in GBP.
    Args:
        portfolio_returns (pd.Series): Daily returns of the portfolio.
        confidence_level (float): Confidence level for VaR and CVaR.
        time_horizon_days (int): Time horizon in days for the risk calculation.
    Returns:
        dict: Dictionary containing VaR and CVaR.
    """
    var_1_day = portfolio_returns.quantile(1 - confidence_level, interpolation='lower')
    tail_losses = portfolio_returns[portfolio_returns <= var_1_day]
    cvar_1_day = tail_losses.mean()
    return {'VaR': var_1_day * np.sqrt(time_horizon_days), 'CVaR': cvar_1_day * np.sqrt(time_horizon_days)}

def calculate_garch_risk(portfolio_returns: pd.Series, confidence_level: float, time_horizon_days: int) -> dict:
    """
    Calculates VaR and CVaR using GARCH model for a portfolio of US stocks
    denominated in GBP.
    Args:
        portfolio_returns (pd.Series): Daily returns of the portfolio.
        confidence_level (float): Confidence level for VaR and CVaR.
        time_horizon_days (int): Time horizon in days for the risk calculation.
    Returns:
        dict: Dictionary containing VaR and CVaR.
    """
    scaled_returns = portfolio_returns * 100 # Recommended scaling for model fit
    model = arch_model(scaled_returns, dist='t')
    results = model.fit(disp='off')
    # Forecast volatility for the next day
    forecast = results.forecast(horizon=1)
    forecasted_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100 # Reverse scaling
    nu = results.params['nu'] # Degrees of freedom for t-distribution
    q = t.ppf(1 - confidence_level, df=nu) # Critical value for t-distribution
    var_1_day = forecasted_vol * q
    cvar_1_day = -forecasted_vol * (t.pdf(q, df=nu) / (1 - confidence_level)) * ((nu + q**2) / (nu - 1))
    return {'VaR': var_1_day * np.sqrt(time_horizon_days), 'CVaR': cvar_1_day * np.sqrt(time_horizon_days)}

def calculate_monte_carlo_risk(
    portfolio_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    confidence_level: float,
    time_horizon_days: int,
    num_simulations: int = 10000
) -> dict:
    """
    Calculates VaR and CVaR using Monte Carlo simulation for a multi-currency portfolio.

    Args:
        portfolio_df (pd.DataFrame): Portfolio DataFrame containing assets, currencies, and weights.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        risk_free_rate (float): Risk-free rate for discounting.
        confidence_level (float): Confidence level for VaR and CVaR.
        time_horizon_days (int): Time horizon in days for the simulation.
        num_simulations (int): Number of Monte Carlo simulations to run.

    Returns:
        dict: Dictionary containing VaR and CVaR.
    """
    # Identify all unique assets and currencies for the simulation
    equity_assets = portfolio_df[portfolio_df['asset_class'] == 'EQUITY']
    yfinance_tickers = equity_assets['yfinance_ticker'].unique().tolist()

    all_currencies = portfolio_df['currency'].unique().tolist()
    fx_tickers = [f"{cur}{BASE_CURRENCY}=X" for cur in all_currencies if cur != BASE_CURRENCY]

    # Fetch historical data for all components
    stock_prices = gfd.fetch_adj_close_prices(yfinance_tickers, start_date, end_date)
    fx_prices = gfd.fetch_currency_data(fx_tickers, start_date, end_date)

    # Map stock price columns back to original tickers
    ticker_map = pd.Series(equity_assets.ticker.values, index=equity_assets.yfinance_ticker).to_dict()
    stock_prices.rename(columns=ticker_map, inplace=True)

    # Combine all price series and calculate log returns and correlations
    all_prices = pd.concat([stock_prices, fx_prices], axis=1).ffill().bfill()
    log_returns = calculate_log_returns(all_prices)

    if log_returns.empty or len(log_returns) < 2:
        print("Error: Cannot run Monte Carlo sim, not enough historical data for correlation.")
        return {'VaR': np.nan, 'CVaR': np.nan}

    corr_matrix = log_returns.corr()

    # Get volatilities (IV if available, otherwise historical)
    component_ivs = log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    for _, row in equity_assets.iterrows():
        iv_result = gfd.calculate_atm_iv(row['yfinance_ticker'], risk_free_rate)
        if iv_result is not None:
            component_ivs[row['ticker']] = iv_result

    # Simulation setup
    num_assets = len(component_ivs)
    dt = 1 / TRADING_DAYS_PER_YEAR
    L = np.linalg.cholesky(corr_matrix)

    drift = (risk_free_rate - 0.5 * component_ivs**2) * dt
    vol_shock_component = component_ivs * np.sqrt(dt)

    # Run simulation
    simulation_outcomes = []
    for _ in range(num_simulations):
        # Generate correlated random shocks for one entire simulated path
        uncorrelated_shocks = np.random.normal(size=(num_assets, time_horizon_days))
        correlated_shocks = L @ uncorrelated_shocks # Matrix multiplication to apply correlation

         # Calculate the path of daily returns for this single simulation
        daily_returns_path = drift.values[:, np.newaxis] + vol_shock_component.values[:, np.newaxis] * correlated_shocks
        total_component_returns = pd.Series(daily_returns_path.sum(axis=1), index=component_ivs.index)

         # Calculate the final portfolio return for this single simulation
        portfolio_return_gbp = 0.0
        for _, row in equity_assets.iterrows():
            stock_return_local = total_component_returns.get(row['ticker'], 0)
            if row['currency'] == BASE_CURRENCY:
                stock_return_gbp = stock_return_local
            else:
                fx_return = total_component_returns.get(f"{row['currency']}{BASE_CURRENCY}", 0)
                stock_return_gbp = stock_return_local + fx_return
            portfolio_return_gbp += row['weight'] * stock_return_gbp

        cash_assets = portfolio_df[portfolio_df['asset_class'] == 'CASH']
        for _, row in cash_assets.iterrows():
            if row['currency'] != BASE_CURRENCY:
                fx_return = total_component_returns.get(f"{row['currency']}{BASE_CURRENCY}", 0)
                portfolio_return_gbp += row['weight'] * fx_return

        simulation_outcomes.append(portfolio_return_gbp)

    # Calculate risk metrics
    simulation_outcomes = np.array(simulation_outcomes)
    VaR = np.percentile(simulation_outcomes, (1 - confidence_level) * 100)
    CVaR = simulation_outcomes[simulation_outcomes <= VaR].mean()

    return {'VaR': VaR, 'CVaR': CVaR}

if __name__ == "__main__":
    start_date = '2005-01-01'
    end_date = '2025-07-28'
    confidence_level = 0.95
    time_horizon = 1
    risk_free_rate = RISK_FREE_RATE


    portfolio = ibc.get_ib_portfolio()
    if portfolio.empty:
        exit()

    portfolio_returns_gbp, weights, portfolio_df = prepare_portfolio_data(portfolio, start_date, end_date)

    if not portfolio_returns_gbp.empty:
        # Historical method
        print(f"\n--- Historical Simulation Results ({time_horizon}-day) ---")
        hist_results = calculate_historical_risk(portfolio_returns_gbp, confidence_level, time_horizon)
        print(f"VaR {confidence_level:.0%}: {hist_results['VaR']:.2%}")
        print(f"CVaR {confidence_level:.0%}: {hist_results['CVaR']:.2%}")

        # GARCH method
        print(f"\n--- GARCH Model Results ({time_horizon}-day) ---")
        garch_results = calculate_garch_risk(portfolio_returns_gbp, confidence_level, time_horizon)
        print(f"VaR {confidence_level:.0%}: {garch_results['VaR']:.2%}")
        print(f"CVaR {confidence_level:.0%}: {garch_results['CVaR']:.2%}")
    else:
        print("\nCould not calculate Historical/GARCH risk due to missing return data.")

    # Monte Carlo method
    print(f"\n--- Monte Carlo Simulation Results ({time_horizon}-day) ---")
    mc_results = calculate_monte_carlo_risk(
        portfolio_df,
        start_date, 
        end_date, 
        risk_free_rate, 
        confidence_level,
        time_horizon
    )
    if mc_results:
        print(f"VaR {confidence_level:.0%}: {mc_results['VaR']:.2%}")
        print(f"CVaR {confidence_level:.0%}: {mc_results['CVaR']:.2%}")