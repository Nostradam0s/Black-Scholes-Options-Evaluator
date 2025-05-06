import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Black-Scholes Formula to Calculate Call and Put Option Prices
def black_scholes(S, X, T, r, sigma, option_type='call'):
    # Calculate d1 and d2
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate option price
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Choose either 'call' or 'put'.")

    return option_price

# Greeks Calculation (Delta, Gamma, Theta, Vega, Rho)
def option_greeks(S, X, T, r, sigma, option_type='call'):
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * X * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * X * np.exp(-r * T) * norm.cdf(-d2)

    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Rho
    if option_type == 'call':
        rho = X * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -X * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

# Visualization Function for Option Price vs. Volatility
def plot_option_price_vs_volatility(S, X, T, r, sigma_range, option_type='call'):
    prices = []
    for sigma in sigma_range:
        price = black_scholes(S, X, T, r, sigma, option_type)
        prices.append(price)

    plt.plot(sigma_range, prices, label=f'{option_type.capitalize()} Option Price')
    plt.title(f'Option Price vs. Volatility ({option_type.capitalize()})')
    plt.xlabel('Volatility')
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.legend()
    plt.show()

# Visualization Function for Option Price vs. Time to Expiry
def plot_option_price_vs_time(S, X, r, sigma, time_range, option_type='call'):
    prices = []
    for T in time_range:
        price = black_scholes(S, X, T, r, sigma, option_type)
        prices.append(price)

    plt.plot(time_range, prices, label=f'{option_type.capitalize()} Option Price')
    plt.title(f'Option Price vs. Time to Expiry ({option_type.capitalize()})')
    plt.xlabel('Time to Expiry (Years)')
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.legend()
    plt.show()

# Fetch live data using yfinance with fallback for stock price (S) only
def fetch_live_data(ticker_symbol, strike_price, expiry_date, r, fallback_sigma=0.2):
    ticker = yf.Ticker(ticker_symbol)  # Create a Ticker object for the stock
    data = ticker.history(period="1d")  # Fetch 1-day historical market data

    if data.empty:
        print(f"Warning: No data returned for ticker '{ticker_symbol}'. Using fallback stock price.")
        S = 192.32  # Fallback stock price (e.g., from a known good date like 2023-12-26)
    else:
        S = data['Close'].iloc[-1]  # Get the latest closing stock price

    X = strike_price  # Strike price of the option (input)
    today = datetime.today()  # Get today's date
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d")  # Convert expiry string to datetime

    # Ensure expiry is in the future
    if expiry <= today:
        raise ValueError(f"Expiry date {expiry_date} must be in the future.")

    T = (expiry - today).days / 365.0  # Time to expiration in years

    # Ensure time to expiry is greater than zero
    if T <= 0:
        raise ValueError("Time to expiry must be greater than zero.")

    # Set volatility (sigma) to a fallback value if it is not defined
    sigma = fallback_sigma

    return S, X, T, r, sigma  # Return all values needed for Black-Scholes model

# Main Execution - Test With Apple Inc. Stocks
if __name__ == "__main__":
    # Parameters
    ticker_symbol = "AAPL"  # You can change this to any stock symbol
    strike_price = 180
    expiry_date = "2029-12-26"
    r = 0.05  # Approximate risk-free rate
    sigma = 0.25  # Estimated or implied volatility

    # Fetch live values
    S, X, T, r, sigma = fetch_live_data(ticker_symbol, strike_price, expiry_date, r)

    # Calculate Call Option Price
    call_price = black_scholes(S, X, T, r, sigma, 'call')
    print(f"Call Option Price: {call_price:.2f}")

    # Calculate Put Option Price
    put_price = black_scholes(S, X, T, r, sigma, 'put')
    print(f"Put Option Price: {put_price:.2f}")

    # Calculate Greeks for Call Option
    greeks_call = option_greeks(S, X, T, r, sigma, 'call')
    print("\nCall Option Greeks:")
    for greek, value in greeks_call.items():
        print(f"{greek}: {value:.4f}")

    # Plot Option Price vs. Volatility
    sigma_range = np.linspace(0.1, 1.0, 100)
    plot_option_price_vs_volatility(S, X, T, r, sigma_range, 'call')

    # Plot Option Price vs. Time to Expiry
    time_range = np.linspace(0.01, 2.0, 100)
    plot_option_price_vs_time(S, X, r, sigma, time_range, 'call')
