import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t
import gather_financial_data as gfd
import time

TRADING_DAYS_PER_YEAR = 252

def load_stock_portfolio_from_csv(file_path: str) -> dict | None:
    """
    Loads a stock portfolio from a CSV file.
    The CSV should have two columns: 'tickers' and 'weights'.
    Args:
        file_path (str): Path to the CSV file containing stock tickers and weights.
    Returns:
        dict: A dictionary with stock tickers as keys and their weights as values.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Warning: Portfolio file '{file_path}' is empty.")
            return None
        return {'stocks': dict(zip(df['tickers'], df['weights']))}
    except FileNotFoundError:
        print(f"Error: Portfolio file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error reading portfolio file: {e}")
        return None

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

def prepare_portfolio_data(portfolio: dict, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepares stock and FX data for the portfolio.
    Args:
        portfolio (dict): Portfolio containing stock tickers and weights.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
    Returns:
        tuple: A tuple containing stock log returns, FX log returns, and adjusted portfolio.
    """
    stock_tickers = list(portfolio.get('stocks', {}).keys())
    stock_prices = gfd.fetch_adj_close_prices(stock_tickers, start_date, end_date)
    stock_prices.dropna(axis=1, how='all', inplace=True)
    
    successful_tickers = stock_prices.columns
    adjusted_weights = {ticker: portfolio['stocks'][ticker] for ticker in successful_tickers}
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        adjusted_weights = {ticker: weight / total_weight for ticker, weight in adjusted_weights.items()}
    adjusted_portfolio = {'stocks': adjusted_weights}
    
    stock_log_returns = calculate_log_returns(stock_prices)
    fx_prices = gfd.fetch_currency_data('GBP=X', start_date, end_date)
    fx_log_returns = calculate_log_returns(fx_prices)
    
    return stock_log_returns, fx_log_returns, adjusted_portfolio

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
    portfolio: dict,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    confidence_level: float,
    time_horizon_days: int,
    num_simulations: int = 10000
) -> dict:
    """
    Calculates VaR and CVaR using Monte Carlo simulation for a portfolio of
    US stocks denominated in GBP.
    Args:
        portfolio (dict): Portfolio containing stock tickers and weights.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        risk_free_rate (float): Risk-free rate for discounting.
        confidence_level (float): Confidence level for VaR and CVaR.
        time_horizon_days (int): Time horizon in days for the simulation.
        num_simulations (int): Number of Monte Carlo simulations to run.
    Returns:
        dict: Dictionary containing VaR and CVaR.
    """
    stock_tickers = list(portfolio.get('stocks', {}).keys())

    # Fetch stock and currency data
    stock_prices = gfd.fetch_adj_close_prices(stock_tickers, start_date, end_date)
    fx_prices = gfd.fetch_currency_data('GBP=X', start_date, end_date)
    
    # Combine prices to calculate correlation across all stocks / currency
    all_prices = pd.concat([stock_prices, fx_prices], axis=1).ffill().bfill()
    log_returns = calculate_log_returns(all_prices)
    corr_matrix = log_returns.corr()

    # Get volatilities for all stocks / currency
    stock_ivs = log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    for ticker in stock_tickers:
        iv_result = gfd.calculate_atm_iv(ticker, risk_free_rate)
        if iv_result is not None:
            stock_ivs[ticker] = iv_result

    fx_iv = gfd.calculate_atm_iv('FXB', risk_free_rate)
    if fx_iv is not None:
        stock_ivs[fx_prices.columns[0]] = fx_iv # Uses the actual column name

    # Simulation setup
    num_assets = len(stock_ivs)
    dt = 1 / TRADING_DAYS_PER_YEAR 
    L = np.linalg.cholesky(corr_matrix)

    drift = (risk_free_rate - 0.5 * stock_ivs**2) * dt
    vol_shock_component = stock_ivs * np.sqrt(dt)

    # Run simulation loop
    simulation_outcomes = []
    for i in range(num_simulations):
        # Generate correlated random shocks for one entire simulated path
        uncorrelated_shocks = np.random.normal(size=(num_assets, time_horizon_days))
        correlated_shocks = L @ uncorrelated_shocks # Matrix multiplication to apply correlation
        
        # Calculate the path of daily returns for this single simulation
        daily_returns = drift.values[:, np.newaxis] + vol_shock_component.values[:, np.newaxis] * correlated_shocks
        total_asset_returns = daily_returns.sum(axis=1)
        
        # Calculate the final portfolio return for this single simulation
        weights = pd.Series(portfolio['stocks']).reindex(stock_ivs.index).fillna(0)
        stock_returns_sim = total_asset_returns[weights.index.get_indexer(stock_tickers)]
        fx_returns_sim = total_asset_returns[weights.index.get_indexer(fx_prices.columns)]
        
        portfolio_return_usd = weights[stock_tickers].values @ stock_returns_sim
        portfolio_return_gbp = portfolio_return_usd + fx_returns_sim.squeeze()
        
        simulation_outcomes.append(portfolio_return_gbp)

    # Calculate risk metrics
    simulation_outcomes = np.array(simulation_outcomes)
    VaR = np.percentile(simulation_outcomes, (1 - confidence_level) * 100)
    CVaR = simulation_outcomes[simulation_outcomes <= VaR].mean()
    
    return {'VaR': VaR, 'CVaR': CVaR}

if __name__ == "__main__":
    portfolio_file = 'sp500_tickers_with_weights_test.csv'
    start_date = '2005-01-01'
    end_date = '2025-07-28'
    confidence_level = 0.95
    time_horizon = 1
    risk_free_rate = 0.05

    portfolio = load_stock_portfolio_from_csv(portfolio_file)
    if not portfolio:
        exit()

    stock_log_returns, fx_log_returns, adjusted_portfolio = prepare_portfolio_data(portfolio, start_date, end_date)
    
    if not stock_log_returns.empty and not fx_log_returns.empty:
        weights = pd.Series(adjusted_portfolio.get('stocks', {})).reindex(stock_log_returns.columns)
        portfolio_returns_usd = stock_log_returns.dot(weights)
        fx_returns_gbp_per_usd = fx_log_returns.squeeze()
        portfolio_returns_gbp = portfolio_returns_usd.add(fx_returns_gbp_per_usd, fill_value=0)
        
        # Historical method
        print(f"\n--- Historical Simulation Results ({time_horizon}-day) ---")
        hist_results = calculate_historical_risk(portfolio_returns_gbp, confidence_level, time_horizon)
        print(f"VaR (95%): {hist_results['VaR']:.2%}")
        print(f"CVaR (95%): {hist_results['CVaR']:.2%}")

        # GARCH method
        print(f"\n--- GARCH Model Results ({time_horizon}-day) ---")
        garch_results = calculate_garch_risk(portfolio_returns_gbp, confidence_level, time_horizon)
        print(f"VaR (95%): {garch_results['VaR']:.2%}")
        print(f"CVaR (95%): {garch_results['CVaR']:.2%}")

    # Monte Carlo method
    print(f"\n--- Monte Carlo Simulation Results ({time_horizon}-day) ---")
    mc_results = calculate_monte_carlo_risk(
        portfolio, 
        start_date, 
        end_date, 
        risk_free_rate, 
        confidence_level,
        time_horizon
    )
    if mc_results:
        print(f"VaR (95%): {mc_results['VaR']:.2%}")
        print(f"CVaR (95%): {mc_results['CVaR']:.2%}")