# portfolio_loader_ibkr.py (Version 4 - Filters BASE Currency)

import pandas as pd
from ib_insync import IB, util, Contract

BACKUP_PORTFOLIO = 'backup_portfolio.csv'

EXCHANGE_INFO_MAP = {
    # United Kingdom
    'LSE':      {'country': 'GBR', 'yf_suffix': '.L'},
    'LSEIOB':   {'country': 'GBR', 'yf_suffix': '.L'},
    'LSEETF':   {'country': 'GBR', 'yf_suffix': '.L'},
    # United States (no suffix needed)
    'NYSE':     {'country': 'USA', 'yf_suffix': ''},
    'NASDAQ':   {'country': 'USA', 'yf_suffix': ''},
    'ARCA':     {'country': 'USA', 'yf_suffix': ''},
    'AMEX':     {'country': 'USA', 'yf_suffix': ''},
    # Canada
    'TSE':      {'country': 'CAN', 'yf_suffix': '.TO'}, 
    'TSX':      {'country': 'CAN', 'yf_suffix': '.TO'}, 
    'TSXV':     {'country': 'CAN', 'yf_suffix': '.V'},  
    # Germany
    'IBIS':     {'country': 'DEU', 'yf_suffix': '.DE'}, 
    'FWB':      {'country': 'DEU', 'yf_suffix': '.F'},  
    # France 
    'SBF':      {'country': 'FRA', 'yf_suffix': '.PA'}, 
    # Netherlands 
    'AEB':      {'country': 'NLD', 'yf_suffix': '.AS'}, 
    # Belgium 
    'BRU':      {'country': 'BEL', 'yf_suffix': '.BR'}, 
    # Switzerland
    'SWX':      {'country': 'CHE', 'yf_suffix': '.SW'},
    'EBS':      {'country': 'CHE', 'yf_suffix': '.SW'}, 
    # Japan
    'TSEJ':     {'country': 'JPN', 'yf_suffix': '.T'},
    # Hong Kong
    'SEHK':     {'country': 'HKG', 'yf_suffix': '.HK'},
    # Australia
    'ASX':      {'country': 'AUS', 'yf_suffix': '.AX'},
    # Italy
    'BVME':     {'country': 'ITA', 'yf_suffix': '.MI'},
    # Spain
    'MCE':      {'country': 'ESP', 'yf_suffix': '.MC'},
}

CURRENCY_TO_COUNTRY_MAP = {
    'GBP': 'GBR',
    'USD': 'USA',
    'EUR': 'EUR', 
    'CAD': 'CAN',
    'CHF': 'CHE',
    'JPY': 'JPN',
    'AUD': 'AUS',
    'HKD': 'HKG',
    'SGD': 'SGP',
}

def get_ib_portfolio(host='127.0.0.1', port=7497, client_id=1):
    """
    Connects to a running IBKR TWS/Gateway instance, fetches the portfolio,
    and formats it into a standardized DataFrame.
    Args:
        host (str): Hostname of the IBKR TWS/Gateway instance.
        port (int): Port number of the IBKR TWS/Gateway instance.
        client_id (int): Client ID for the IBKR connection.

    Returns:
        pd.DataFrame: Downloaded Interactive Brokers DataFrame
    """
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        processed_data = []

        # Get equities and fixed income
        portfolio_positions = ib.portfolio()
        for position in portfolio_positions:
            contract = position.contract
            sec_type = contract.secType

            if sec_type in ['STK', 'BOND']:
                if sec_type == 'STK':
                    asset_class = 'EQUITY'
                else: # 'BOND'
                    asset_class = 'BOND'

                # Clean the ticker from IBKR - some stocks have a trailing '.'
                ticker = contract.localSymbol.rstrip('.')

                # Use primary exchange if available
                if contract.primaryExchange:
                    listing_exchange = contract.primaryExchange
                # Handle Smart routed cases
                else:
                    listing_exchange = contract.exchange

                # print(f"DEBUG: Ticker: {ticker}, Exchange: {contract.exchange}, PrimaryExchange: {listing_exchange}")

                # Get exchange info from EXCHANGE_INFO_MAP. Default to USA if not found.
                exchange_info = EXCHANGE_INFO_MAP.get(listing_exchange, {'country': 'USA', 'yf_suffix': ''})
                country = exchange_info['country']

                # Create yfinance ticker for equities
                yfinance_ticker = ticker
                if asset_class == 'EQUITY':
                    yfinance_ticker = f"{ticker}{exchange_info['yf_suffix']}"

                processed_data.append({
                    'asset_class': asset_class,
                    'ticker': ticker,
                    'yfinance_ticker': yfinance_ticker,
                    'country': country,
                    'currency': contract.currency,
                    'cash_amount': round(position.marketValue, 2)
                })

        # Get cash balances
        account_values = ib.accountValues()
        for value in account_values:
            if value.tag == 'TotalCashBalance' and value.currency != '':
                if value.currency == 'BASE':
                    continue
                cash_currency = value.currency
                cash_amount = float(value.value)

                if abs(cash_amount) < 0.01:
                    continue

                country = CURRENCY_TO_COUNTRY_MAP.get(cash_currency, 'OTHER')
                
                processed_data.append({
                    'asset_class': 'CASH',
                    'ticker': cash_currency,
                    'yfinance_ticker': cash_currency,
                    'country': country,
                    'currency': cash_currency,
                    'cash_amount': round(cash_amount, 2)
                })
        
        if not processed_data:
            print("Warning: Portfolio is empty.")
            return pd.DataFrame()

        portfolio_df = pd.DataFrame(processed_data)
        portfolio_df.to_csv(BACKUP_PORTFOLIO, index=False) # Update backup file
        return portfolio_df

    except Exception as e:
        print(f"Error: An error occurred while connecting to IBKR: {e}")
        print(f"Attempting to load from backup file: {BACKUP_PORTFOLIO}")
        try:
            portfolio_df = pd.read_csv(BACKUP_PORTFOLIO)
            return portfolio_df
        except FileNotFoundError:
            print(f"Error: Backup file '{BACKUP_PORTFOLIO}' not found.")
            return pd.DataFrame()
        except Exception as backup_e:
            print(f"Error: An error occurred while reading the backup file: {backup_e}")
            return pd.DataFrame()

    finally:
        if ib.isConnected():
            ib.disconnect()

if __name__ == '__main__':
    my_portfolio = get_ib_portfolio()
    print(my_portfolio)