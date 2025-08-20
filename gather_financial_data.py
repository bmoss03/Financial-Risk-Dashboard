import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import time

DEFAULT_SLEEP = 1  # Default sleep time to avoid hitting API rate limits

def fetch_adj_close_prices(tickers: str | list, start_date: str = None, end_date: str = None, max_range: bool = False, sleep_time: float = DEFAULT_SLEEP) -> pd.DataFrame:
    """
    Fetches adjusted closing prices for one or more stock tickers from Yahoo Finance.

    Args:
        tickers (str or list): A single stock ticker (e.g., 'AAPL') or a list
                               of stock tickers (e.g., ['AAPL', 'MSFT']).
        start_date (str, optional): The start date for fetching data (format 'YYYY-MM-DD').
                                  Required unless max_range is True.
        end_date (str, optional): The end date for fetching data (format 'YYYY-MM-DD').
                                  Defaults to the current date if None.
        max_range (bool, optional): If True, fetches the maximum available history for the ticker(s).
                                    Overrides start_date and end_date if set to True. Defaults to False.
        sleep_time (int/float, optional): Time in seconds to pause after each download request.
                                          Useful to avoid hitting API rate limits. Defaults to DEFAULT_SLEEP.

    Returns:
        pandas.DataFrame: A DataFrame with dates as index and adjusted closing
                          prices for the specified tickers as columns.
                          Returns an empty DataFrame if no data is found or an error occurs.
    """
    if not max_range:
        if start_date is None:
            raise ValueError("start_date must be provided if max_range is False.")
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        period_params = {'start': start_date, 'end': end_date}
    else:
        period_params = {'period': 'max'}

    try:
        data = yf.download(tickers, **period_params, auto_adjust=True, progress=False)
        time.sleep(sleep_time)

        if data.empty:
            print(f"Warning: No data downloaded for {tickers} for the specified range.")
            return pd.DataFrame()

        if 'Close' not in data.columns:
            print(f"Error: Could not extract 'Close' price data for {tickers}.")
            return pd.DataFrame()

        close_prices = data['Close']

        # Ensure the output is always a DataFrame 
        if isinstance(close_prices, pd.Series):
            adj_close_df = close_prices.to_frame(name=close_prices.name)
        else:

            adj_close_df = close_prices.copy()

        # Drop columns that are all NaN (can happen if a ticker is valid but has no data in range)
        adj_close_df.dropna(axis=1, how='all', inplace=True)

        if adj_close_df.empty:
            print(f"Warning: All tickers in {tickers} resulted in empty data for the specified range.")
            return pd.DataFrame()

        adj_close_df.index = pd.to_datetime(adj_close_df.index)
        return adj_close_df.sort_index()

    except Exception as e:
        print(f"Error fetching data for {tickers}: {e}")
        return pd.DataFrame()

def fetch_currency_data(currency_pair_tickers: str | list, start_date: str = None, end_date: str = None, max_range: bool = False, sleep_time: float = DEFAULT_SLEEP) -> pd.DataFrame:
    """
    Fetches historical exchange rate data for a specified currency pair from Yahoo Finance.
    Note: Tickers should be in the format 'FROMCURTO_CUR=X', e.g., 'USDGBP=X' for GBP per 1 USD.

    Args:
        currency_pair_ticker (str): The ticker symbol for the currency pair (e.g., 'GBP=X' for GBP/USD).
        start_date (str, optional): The start date for fetching data (format 'YYYY-MM-DD').
                                    Required unless max_range is True.
        end_date (str, optional): The end date for fetching data (format 'YYYY-MM-DD').
                                  Defaults to the current date if None.
        max_range (bool, optional): If True, fetches the maximum available history.
                                    Overrides start_date and end_date. Defaults to False.
        sleep_time (int/float, optional): Time in seconds to pause after the download request.
                                          Defaults to DEFAULT_SLEEP.

    Returns:
        pandas.DataFrame: A DataFrame with dates as index and the processed exchange rate
                          (e.g., GBP per 1 USD) as a named column.
                          Returns an empty DataFrame if no data is found or an error occurs.
    """
    if not max_range:
        if start_date is None:
            raise ValueError("start_date must be provided if max_range is False.")
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        download_params = {'start': start_date, 'end': end_date}
    else:
        download_params = {'period': 'max'}

    try:
        data = yf.download(currency_pair_tickers, **download_params, auto_adjust=True, progress=False)
        time.sleep(sleep_time)

        if data.empty:
            print(f"Warning: No data downloaded for {currency_pair_tickers} for the specified range.")
            return pd.DataFrame()

        
        if 'Close' not in data.columns:
            print(f"Error: Could not extract 'Close' price data for {currency_pair_tickers}.")
            return pd.DataFrame()

        close_prices = data['Close']

        # Ensure the output is always a DataFrame
        if isinstance(close_prices, pd.Series):
            ticker_name = currency_pair_tickers if isinstance(currency_pair_tickers, str) else currency_pair_tickers[0]
            currency_rate_df = close_prices.to_frame(name=ticker_name.replace('=X', ''))
        else:
            currency_rate_df = close_prices.copy()
            currency_rate_df.columns = [col.replace('=X', '') for col in currency_rate_df.columns]

        # Drop columns that are all NaN
        # TODO: Flag individual tickers that have no data
        currency_rate_df.dropna(axis=1, how='all', inplace=True)

        if currency_rate_df.empty:
            print(f"Warning: All tickers in {currency_pair_tickers} resulted in empty data.")
            return pd.DataFrame()

        currency_rate_df.index = pd.to_datetime(currency_rate_df.index)
        return currency_rate_df.sort_index()

    except Exception as e:
        print(f"An error occurred while fetching data for {currency_pair_tickers}: {e}")
        return pd.DataFrame()

def fetch_option_chain(ticker: str, exp_date: str = None, sleep_time_sec: float = 0.5) -> pd.DataFrame:
    # TODO: Handle multiple tickers here
    """
    Fetches the full option chain for a given stock ticker.
    If no expiration date is provided, it fetches the chain for the nearest
    available expiration date.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        exp_date (str, optional): The expiration date in 'YYYY-MM-DD' format.
                                  Defaults to None (nearest expiration).
        sleep_time_sec (int/float, optional): Time in seconds to pause after the download request.
                                              Defaults to 1.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined call and put option
                          chain. Returns an empty DataFrame if no options are
                          found or an error occurs.
    """
    try:
        # Create a Ticker object for the stock
        stock_ticker = yf.Ticker(ticker)

        # Get the list of available expiration dates
        available_dates = stock_ticker.options
        if not available_dates:
            print(f"Warning: No options found for ticker '{ticker}'.")
            return pd.DataFrame()

        # Determine the target expiration date
        target_date = exp_date
        if target_date is None:
            target_date = available_dates[0]  # Default to the nearest date
            #print(f"Note: No expiration date provided. Using nearest date: {target_date}")
        elif target_date not in available_dates:
            print(f"Error: Expiration date '{target_date}' not found for '{ticker}'.")
            print(f"Warning: Available dates are: {available_dates}")
            return pd.DataFrame()

        # Fetch the option chain for the target date
        option_chain = stock_ticker.option_chain(target_date)
        time.sleep(sleep_time_sec)  # Add sleep timer after API call

        calls_df = option_chain.calls
        puts_df = option_chain.puts

        if calls_df.empty and puts_df.empty:
            print(f"Warning: No option data found for {ticker} on {target_date}.")
            return pd.DataFrame()

        # Add a 'type' column to distinguish between calls and puts
        calls_df['type'] = 'call'
        puts_df['type'] = 'put'

        # Combine the calls and puts into a single DataFrame
        full_chain_df = pd.concat([calls_df, puts_df], ignore_index=True)

        # Add other useful information
        full_chain_df['underlying'] = ticker
        
        # Calculate Days to Expiration (DTE)
        exp_datetime = pd.to_datetime(target_date)
        # Get the current date without the time component
        today = pd.to_datetime(pd.Timestamp.now(tz='UTC').date())
        full_chain_df['dte'] = (exp_datetime - today).days
        
        return full_chain_df

    except Exception as e:
        print(f"Error: An error occurred while fetching option data for '{ticker}': {e}")
        return pd.DataFrame()

def calculate_atm_iv(ticker: str, risk_free_rate: float) -> float | None:
    # TODO: Handle multiple tickers here
    """
    Calculates IV for the ATM option on the furthest date, using a call or put.

    This function prioritizes a call option but will automatically use a put
    if no calls are available for the ATM strike on the selected date.

    Args:
        ticker (str): The stock ticker symbol.
        risk_free_rate (float): The current risk-free interest rate (e.g., 0.05 for 5%).

    Returns:
        float: The calculated implied volatility, or None if calculation fails.
    """
    #print(f"\n--- Calculating IV for {ticker} on furthest available date ---")
    try:
        # Step 1: Find the furthest available expiration date
        stock_ticker = yf.Ticker(ticker)
        available_dates = stock_ticker.options
        if not available_dates:
            print(f"Error: No options found for ticker '{ticker}'.")
            return None

        target_date = available_dates[-1]
        today = pd.to_datetime(pd.Timestamp.now(tz='UTC').date())
        exp_datetime = pd.to_datetime(target_date)
        dte = (exp_datetime - today).days
        
        #print(f"Found furthest expiration: {target_date} ({dte} DTE)")

        # Step 2: Fetch option chain and underlying price
        option_chain = stock_ticker.option_chain(target_date)
        calls_df = option_chain.calls
        puts_df = option_chain.puts
        S = stock_ticker.history(period='1d')['Close'].iloc[0]

        # Step 3: Prioritize ATM Call, but fall back to ATM Put
        # Find the single strike price closest to the underlying price
        combined_df = pd.concat([calls_df, puts_df])
        if combined_df.empty:
            print(f"Error: No options available for {ticker} on {target_date}.")
            return None
            
        atm_strike = combined_df.iloc[(combined_df['strike'] - S).abs().idxmin()]['strike']
        
        # Try to get the ATM call first
        atm_option = calls_df[calls_df['strike'] == atm_strike]
        option_type = 'Call'

        # If no ATM call exists, try to get the ATM put
        if atm_option.empty:
            atm_option = puts_df[puts_df['strike'] == atm_strike]
            option_type = 'Put'
            if atm_option.empty:
                print(f"Error: No ATM call or put found at strike {atm_strike} for {target_date}.")
                return None
        
        # Step 4: Extract parameters for calculation
        atm_option = atm_option.iloc[0]
        K = atm_option['strike']
        
        if atm_option['bid'] > 0 and atm_option['ask'] > 0:
            market_price = (atm_option['bid'] + atm_option['ask']) / 2
        else:
            market_price = atm_option['lastPrice']
        
        T = dte / 365.25
        r = risk_free_rate
        

        # Use Newton-Raphson method to estimate implied volatility
        sigma = 0.5
        for i in range(100):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Use the correct Black-Scholes formula based on option type
            if option_type == 'Call':
                model_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
            else: # Put option decreases as the underlying increases
                model_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

            vega = S * norm.pdf(d1) * np.sqrt(T)
            price_diff = model_price - market_price

            if abs(price_diff) < 1e-5:
                break
            if vega < 1e-6:
                return None
            sigma = sigma - price_diff / vega
        
        return sigma

    except Exception as e:
        print(f"Error: An unexpected error occurred for '{ticker}': {e}")
        return None

if __name__ == "__main__":
    # TODO: Dynamic risk-free rate fetching for specific countries
    current_risk_free_rate = 0.04101 # Current 3 year Gilt yield

    # Example 1: Single stock with a specific range
    aapl_prices = fetch_adj_close_prices('AAPL', '2020-01-01', '2023-12-31')
    if not aapl_prices.empty:
        print("\nAAPL Adjusted Close Prices (Head):")
        print(aapl_prices.head())
        print("\nAAPL Adjusted Close Prices (Tail):")
        print(aapl_prices.tail())
    else:
        print("Example 1: AAPL data not fetched.")

    # Example 2: Multiple stocks with a specific range
    my_stocks = ['MSFT', 'GOOGL', 'AMZN']
    multi_stock_prices = fetch_adj_close_prices(my_stocks, '2021-01-01', '2024-07-28')
    if not multi_stock_prices.empty:
        print(f"\n{my_stocks} Adjusted Close Prices (Head):")
        print(multi_stock_prices.head())
        print(f"\n{my_stocks} Adjusted Close Prices (Tail):")
        print(multi_stock_prices.tail())
    else:
        print(f"Example 2: {my_stocks} data not fetched.")

    # Example 3: Invalid ticker
    invalid_prices = fetch_adj_close_prices('INVALIDTICKER', '2022-01-01', '2022-03-01')
    if invalid_prices.empty:
        print("\nAttempted to fetch data for an invalid ticker. Returned empty DataFrame as expected.")

    # Example 4: Max available history for AAPL
    max_aapl_prices = fetch_adj_close_prices('AAPL', max_range=True, sleep_time=2)
    if not max_aapl_prices.empty:
        print("\nAAPL Adjusted Close Prices (Max Range, Head):")
        print(max_aapl_prices.head())
        print("\nAAPL Adjusted Close Prices (Max Range, Tail):")
        print(max_aapl_prices.tail())
    else:
        print("\nExample 4: Max range AAPL data not fetched.")

    # Example 5: AAPL data up to last close (using default end_date = today)
    recent_aapl = fetch_adj_close_prices('AAPL', '2024-06-01')
    if not recent_aapl.empty:
        print("\nAAPL Adjusted Close Prices (Recent, without end_date):")
        print(recent_aapl.tail())
    else:
        print("\nNo recent data for AAPL from 2024-06-01 to present (check dates/internet connection).")

    # Example 6: Fetching GBP=X (GBP per USD) for a specific range
    gbp_usd_rate = fetch_currency_data('GBP=X', '2020-01-01', '2023-12-31')
    if not gbp_usd_rate.empty:
        print("\nGBP per USD Rate (Head):")
        print(gbp_usd_rate.head())
        print("\nGBP per USD Rate (Tail):")
        print(gbp_usd_rate.tail())
    else:
        print("Example 1: GBP/USD data not fetched.")

    # Example 7: Fetching max available history for GBP=X
    max_gbp_usd_rate = fetch_currency_data('GBP=X', max_range=True, sleep_time=2)
    if not max_gbp_usd_rate.empty:
        print("\nGBP per USD Rate (Max Range, Head):")
        print(max_gbp_usd_rate.head())
        print("\nGBP per USD Rate (Max Range, Tail):")
        print(max_gbp_usd_rate.tail())
    else:
        print("\nExample 2: Max range GBP/USD data not fetched.")

    # Example 8: Invalid currency ticker
    invalid_currency = fetch_currency_data('INVALIDFX', '2022-01-01', '2022-03-01')
    if invalid_currency.empty:
        print("\nAttempted to fetch data for an invalid currency ticker. Returned empty DataFrame as expected.")

    # Example 9: Fetch options for the nearest expiration date for TSLA
    tsla_options = fetch_option_chain('TSLA')
    if not tsla_options.empty:
        print("\nTSLA Option Chain (Nearest Expiry) - First 5 Rows:")
        print(tsla_options.head())
        print("\nTSLA Option Chain (Nearest Expiry) - Last 5 Rows (likely puts):")
        print(tsla_options.tail())

    print("-" * 50)

    # Example 10: Fetch options for the nearest expiration date for GBP/USD
    # Using FXB as a proxy for GBP/USD
    gbpusd_options = fetch_option_chain('FXB')
    if not gbpusd_options.empty:
        print("\nGBP/USD Option Chain (Nearest Expiry) - First 5 Rows:")
        print(gbpusd_options.head())
        print("\nGBP/USD Option Chain (Nearest Expiry) - Last 5 Rows (likely puts):")
        print(gbpusd_options.tail())

    print("-" * 50)

    # Example 11: Fetch options for a specific expiration date for MSFT
    # To make this example robust, we first get available dates and pick one
    msft_ticker = yf.Ticker('MSFT')
    msft_exp_dates = msft_ticker.options
    if len(msft_exp_dates) > 1:
        # Pick the second available date for this example
        specific_date = msft_exp_dates[1] 
        msft_options = fetch_option_chain('MSFT', exp_date=specific_date)
        if not msft_options.empty:
            print(f"\nMSFT Option Chain for {specific_date} - Sample of Columns:")
            # Display a subset of columns for readability
            cols_to_show = ['contractSymbol', 'type', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'dte']
            print(msft_options[cols_to_show].head())
    else:
        print("\nNot enough expiration dates available for MSFT to run Example 2.")

    print("-" * 50)
    
    # Example 12: Ticker with no options (e.g., Berkshire Hathaway Class A)
    no_options = fetch_option_chain('BRK-A')
    if no_options.empty:
        print("\nAttempt to fetch options for 'BRK-A' correctly returned an empty DataFrame.")

    # Example 13: Manually calculate IV for SPY
    spy_iv = calculate_atm_iv('SPY', risk_free_rate=current_risk_free_rate)
    if spy_iv is not None:
        print(f"\nManually Calculated SPY IV: {spy_iv:.2%}")

    print("-" * 50)
    # Example 14: Manually calculate IV for FXB (GBP/USD proxy) 
    cable_iv = calculate_atm_iv('FXB', risk_free_rate=current_risk_free_rate)
    if cable_iv is not None:
        print(f"\nManually Calculated Cable (FXB) IV: {cable_iv:.2%}")

    # Example 15: Manually calculate IV for TSLA
    tesla_iv = calculate_atm_iv('TSLA', risk_free_rate=current_risk_free_rate)
    if tesla_iv is not None:
        print(f"\nManually Calculated Tesla (TSLA) IV: {tesla_iv:.2%}")

    # Example 16: Invalid ticker for IV calculation
    invalid_iv = calculate_atm_iv('INVALIDTICKER', risk_free_rate=current_risk_free_rate)
    if invalid_iv is None:
        print("\nAttempted to calculate IV for an invalid ticker. Returned None as expected.")