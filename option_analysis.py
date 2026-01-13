"""
Options Analysis Tool for Hackathon
Analyzes option pricing, Greeks, and generates recommendations
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

class OptionAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.05):
        """
        Initialize the Option Analyzer
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            risk_free_rate: Risk-free interest rate (default 5% = 0.05)
        """
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        
    def get_historical_data(self, period="1y"):
        """Get historical stock price data"""
        print(f"Fetching historical data for {self.ticker}...")
        hist_data = self.stock.history(period=period)
        return hist_data
    
    def calculate_volatility(self, hist_data):
        """Calculate historical volatility (annualized)"""
        # Calculate daily returns
        daily_returns = hist_data['Close'].pct_change().dropna()
        
        # Annualized volatility (252 trading days per year)
        volatility = daily_returns.std() * np.sqrt(252)
        
        print(f"Historical Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        return volatility
    def volatility_regime(self, hist_data):
        daily_returns = hist_data['Close'].pct_change().dropna()
        rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)

        rolling_vol = rolling_vol.dropna()
        if rolling_vol.empty:
            return np.nan, np.nan, "INSUFFICIENT DATA"
        current_vol = rolling_vol.iloc[-1]
        long_term_vol = rolling_vol.mean()

        if current_vol < 0.8 * long_term_vol:
            regime = "LOW VOLATILITY (Calm Market)"
        elif current_vol > 1.2 * long_term_vol:
            regime = "HIGH VOLATILITY (Fearful Market)"
        else:
            regime = "NORMAL VOLATILITY"

        return current_vol, long_term_vol, regime

 
    
    def get_available_options(self):
        """Display available option expiration dates"""
        print(f"\nAvailable expiration dates for {self.ticker}:")
        expirations = self.stock.options
        for i, exp in enumerate(expirations[:10], 1):  # Show first 10
            print(f"{i}. {exp}")
        return expirations
    
    def get_option_chain(self, expiration_date):
        """Get option chain for specific expiration date"""
        opt = self.stock.option_chain(expiration_date)
        return opt.calls, opt.puts
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """
        Black-Scholes Option Pricing Model
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # put
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks
        
        Returns:
            Dictionary with Delta, Gamma, Theta, Vega, Rho
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    
    def analyze_option(self, strike_price, expiration_date, option_type='call'):
        """
        Complete analysis of a specific option
        
        Args:
            strike_price: Strike price of the option
            expiration_date: Expiration date (YYYY-MM-DD format)
            option_type: 'call' or 'put'
        """
        # Get current stock price
        current_price = self.stock.history(period='1d')['Close'].iloc[-1]
        print(f"\n{'='*60}")
        print(f"OPTION ANALYSIS: {self.ticker} ${strike_price} {option_type.upper()}")
        print(f"{'='*60}")
        print(f"Current Stock Price: ${current_price:.2f}")
        
        # Get historical data and calculate volatility
        hist_data = self.get_historical_data()
        volatility = self.calculate_volatility(hist_data)

        # --- VOLATILITY REGIME ANALYSIS ---
        curr_vol, avg_vol, regime = self.volatility_regime(hist_data)

        print(f"\n--- VOLATILITY REGIME ---")
        print(f"Current Volatility: {curr_vol:.2%}")
        print(f"Long-term Avg Volatility: {avg_vol:.2%}")
        print(f"Market Regime: {regime}")

        
        # Calculate time to expiration
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        today = datetime.now()
        days_to_exp = (exp_date - today).days
        time_to_exp = days_to_exp / 365
        
        print(f"Days to Expiration: {days_to_exp}")
        print(f"Time to Expiration: {time_to_exp:.4f} years")
        
        # Calculate Black-Scholes price
        bs_price = self.black_scholes(current_price, strike_price, time_to_exp, 
                                      self.risk_free_rate, volatility, option_type)
        print(f"\n--- PRICING ---")
        print(f"Black-Scholes Model Price: ${bs_price:.2f}")

        # --- VOLATILITY SENSITIVITY ANALYSIS ---
        print("\n--- VOLATILITY SENSITIVITY ---")

        for shock in [-0.10, 0.10]:
            shocked_vol = volatility * (1 + shock)
            shocked_price = self.black_scholes(
            current_price, strike_price, time_to_exp,
                self.risk_free_rate, shocked_vol, option_type
            )
            label = "-10%" if shock < 0 else "+10%"
            print(f"If volatility changes {label} → Option Price: ${shocked_price:.2f}")
        
        # --- TIME DECAY SCENARIO ---
        print("\n--- TIME DECAY SCENARIO ---")

        if days_to_exp > 7:
            reduced_T = (days_to_exp - 7) / 365
            price_after_7d = self.black_scholes(
                 current_price, strike_price, reduced_T,
            self.risk_free_rate, volatility, option_type
            )
            print(f"Option price after 7 days (no stock move): ${price_after_7d:.2f}")
        else:
            print("Expiry too close to simulate time decay meaningfully")
 

        # Get market price
        calls, puts = self.get_option_chain(expiration_date)
        if option_type == 'call':
            option_data = calls[calls['strike'] == strike_price]
        else:
            option_data = puts[puts['strike'] == strike_price]
        
        if not option_data.empty:
            market_price = option_data['lastPrice'].iloc[0]
            bid = option_data['bid'].iloc[0]
            ask = option_data['ask'].iloc[0]
            volume = option_data['volume'].iloc[0]
            
            print(f"Market Price (Last): ${market_price:.2f}")
            print(f"Bid: ${bid:.2f} | Ask: ${ask:.2f}")
            print(f"Volume: {volume}")
            
            difference = market_price - bs_price
            pct_diff = (difference / bs_price) * 100
            
            print(f"\nPrice Difference: ${difference:.2f} ({pct_diff:+.2f}%)")
            
            if abs(pct_diff) < 5:
                assessment = "FAIRLY PRICED"
            elif pct_diff > 5:
                assessment = "OVERPRICED (Market > Model)"
            else:
                assessment = "UNDERPRICED (Market < Model)"
            
            print(f"Assessment: {assessment}")
        else:
            print("Market price not available for this strike")
            market_price = None
        
        # Calculate Greeks
        greeks = self.calculate_greeks(current_price, strike_price, time_to_exp, 
                                       self.risk_free_rate, volatility, option_type)
        
        print(f"\n--- GREEKS ---")
        print(f"Delta (Δ): {greeks['Delta']:.4f}")
        print(f"  → If stock moves $1, option moves ${greeks['Delta']:.2f}")
        
        print(f"\nGamma (Γ): {greeks['Gamma']:.6f}")
        print(f"  → Rate of change of Delta")
        
        print(f"\nTheta (Θ): ${greeks['Theta']:.2f} per day")
        print(f"  → Option loses ${abs(greeks['Theta']):.2f} per day from time decay")
        
        print(f"\nVega (ν): ${greeks['Vega']:.2f} per 1% volatility change")
        print(f"  → If volatility increases 1%, option gains ${greeks['Vega']:.2f}")
        
        print(f"\nRho (ρ): ${greeks['Rho']:.2f} per 1% rate change")

        # --- CALL vs PUT COMPARISON ---
        alt_type = 'put' if option_type == 'call' else 'call'

        alt_price = self.black_scholes(
            current_price, strike_price, time_to_exp,
            self.risk_free_rate, volatility, alt_type
        )

        print("\n--- CALL vs PUT COMPARISON ---")
        print(f"{option_type.upper()} Price: ${bs_price:.2f}")
        print(f"{alt_type.upper()} Price: ${alt_price:.2f}")


        print("\n--- GREEK PRIORITY ---")

        if abs(greeks['Delta']) > 0.6:
            print("PRIMARY RISK: Delta (Directional Exposure)")
        elif abs(greeks['Theta']) > 0.05:
            print("PRIMARY RISK: Theta (Time Decay)")
        elif greeks['Vega'] > 0.1:
            print("PRIMARY RISK: Vega (Volatility Sensitivity)")
        else:
            print("No dominant Greek risk detected")
        
        # Risk Analysis
        print(f"\n--- RISK ANALYSIS ---")
        self._risk_analysis(greeks, current_price, option_type)
        
        # Hedging Recommendation
        print(f"\n--- HEDGING STRATEGY ---")
        self._hedging_strategy(greeks, current_price, strike_price, option_type)

        print("\n--- FINAL RECOMMENDATION ---")
        if regime.startswith("HIGH") and pct_diff > 5:
            decision = "AVOID: Option overpriced in fearful market"
        elif regime.startswith("LOW") and pct_diff < -5:
            decision = "OPPORTUNITY: Cheap option in calm market"
        elif abs(greeks['Theta']) > 0.08:
            decision = "AVOID: Severe time decay risk"
        else:
            decision = "NEUTRAL: No strong edge detected"

        print(f"FINAL CALL: {decision}")

        # --- MODEL LIMITATIONS ---
        print("\n--- MODEL LIMITATIONS ---")
        print("• Assumes constant volatility (real markets have volatility smiles)")
        print("• Ignores earnings, news, and event risk")
        print("• European-style pricing (no early exercise)")
        print("• No liquidity, bid-ask spread, or transaction cost modeling")
        
        # Create visualizations
        self._create_visualizations(current_price, strike_price, time_to_exp, 
                                   volatility, option_type, hist_data)
        
        return {
            'bs_price': bs_price,
            'market_price': market_price,
            'greeks': greeks,
            'current_price': current_price,
            'volatility': volatility
        }
    


    
    def _risk_analysis(self, greeks, current_price, option_type):
        """Analyze and explain risks"""
        print("Primary Risks:")
        
        # Delta Risk
        print(f"\n1. DELTA RISK (Directional Risk)")
        if option_type == 'call':
            if greeks['Delta'] > 0.5:
                print(f"   HIGH: Delta = {greeks['Delta']:.2f}")
                print(f"   This option is highly sensitive to stock price movements")
                print(f"   A $10 stock move = ${greeks['Delta']*10:.2f} option price change")
            else:
                print(f"   MODERATE: Delta = {greeks['Delta']:.2f}")
        else:
            print(f"   Delta = {greeks['Delta']:.2f} (negative for puts)")
        
        # Theta Risk
        print(f"\n2. THETA RISK (Time Decay)")
        theta_weekly = greeks['Theta'] * 7
        if abs(greeks['Theta']) > 0.05:
            print(f"   HIGH: Losing ${abs(greeks['Theta']):.2f}/day (${abs(theta_weekly):.2f}/week)")
            print(f"   Time decay is significant - option losing value quickly")
        else:
            print(f"   LOW: Losing ${abs(greeks['Theta']):.2f}/day")
        
        # Vega Risk
        print(f"\n3. VEGA RISK (Volatility Risk)")
        if greeks['Vega'] > 0.10:
            print(f"   MODERATE-HIGH: Vega = ${greeks['Vega']:.2f}")
            print(f"   Sensitive to volatility changes")
            print(f"   If volatility drops 5%, you lose ${greeks['Vega']*5:.2f}")
        else:
            print(f"   LOW: Vega = ${greeks['Vega']:.2f}")
    
    def _hedging_strategy(self, greeks, current_price, strike_price, option_type):
        """Provide hedging recommendations"""
        delta = greeks['Delta']
        
        print("Recommended Hedge (Delta Hedging):")
        print(f"\nFor 1 option contract (100 shares):")
        
        shares_to_hedge = int(abs(delta) * 100)
        
        if option_type == 'call':
            print(f"→ SHORT (sell) {shares_to_hedge} shares of {self.ticker}")
            print(f"   Cost: ${shares_to_hedge * current_price:.2f}")
            print(f"\nRationale:")
            print(f"   • Your call option gains ${delta:.2f} per $1 stock increase")
            print(f"   • Shorting {shares_to_hedge} shares loses ${shares_to_hedge:.2f} per $1 increase")
            print(f"   • Net position becomes delta-neutral")
        else:
            print(f"→ BUY (long) {shares_to_hedge} shares of {self.ticker}")
            print(f"   Cost: ${shares_to_hedge * current_price:.2f}")
            print(f"\nRationale:")
            print(f"   • Your put option gains when stock falls")
            print(f"   • Buying shares offsets this directional risk")
        
        print(f"\nNote: Hedge needs to be adjusted as Delta changes (Gamma effect)")
    
    def _create_visualizations(self, S, K, T, sigma, option_type, hist_data):
        """Create visualization charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} Option Analysis', fontsize=16, fontweight='bold')
        
        # 1. Historical Stock Price
        ax1 = axes[0, 0]
        ax1.plot(hist_data.index, hist_data['Close'], linewidth=2)
        ax1.set_title('Historical Stock Price (12 Months)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=S, color='r', linestyle='--', label=f'Current: ${S:.2f}')
        ax1.legend()
        
        # 2. Option Price vs Stock Price
        ax2 = axes[0, 1]
        stock_prices = np.linspace(S*0.7, S*1.3, 100)
        option_prices = [self.black_scholes(s, K, T, self.risk_free_rate, sigma, option_type) 
                        for s in stock_prices]
        ax2.plot(stock_prices, option_prices, linewidth=2)
        ax2.axvline(x=S, color='r', linestyle='--', label=f'Current Stock: ${S:.2f}')
        ax2.axvline(x=K, color='g', linestyle='--', label=f'Strike: ${K:.2f}')
        ax2.set_title(f'{option_type.capitalize()} Option Price vs Stock Price')
        ax2.set_xlabel('Stock Price ($)')
        ax2.set_ylabel('Option Price ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Greeks Profile
        ax3 = axes[1, 0]
        greeks_data = []
        for s in stock_prices:
            g = self.calculate_greeks(s, K, T, self.risk_free_rate, sigma, option_type)
            greeks_data.append(g['Delta'])
        ax3.plot(stock_prices, greeks_data, linewidth=2)
        ax3.axvline(x=S, color='r', linestyle='--', label=f'Current: ${S:.2f}')
        ax3.set_title('Delta vs Stock Price')
        ax3.set_xlabel('Stock Price ($)')
        ax3.set_ylabel('Delta')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Time Decay
        ax4 = axes[1, 1]
        days = np.linspace(T*365, 0, 50)
        time_values = [self.black_scholes(S, K, d/365, self.risk_free_rate, sigma, option_type) 
                      for d in days]
        ax4.plot(days, time_values, linewidth=2)
        ax4.set_title('Option Price Time Decay')
        ax4.set_xlabel('Days to Expiration')
        ax4.set_ylabel('Option Price ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_option_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Charts saved as '{self.ticker}_option_analysis.png'")
        plt.show()


def main():
    """Main function to run the analysis"""
    print("="*60)
    print("OPTIONS ANALYSIS TOOL - HACKATHON SOLUTION")
    print("="*60)
    
    # Step 1: Choose a stock
    print("\nAvailable stocks:")
    stocks = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL']
    for i, stock in enumerate(stocks, 1):
        print(f"{i}. {stock}")
    
    ticker = input("\nEnter stock ticker (e.g., AAPL): ").upper()
    
    if ticker not in stocks:
        print(f"Warning: {ticker} not in recommended list, but proceeding...")
    
    # Initialize analyzer
    analyzer = OptionAnalyzer(ticker)
    
    # Step 2: Show available expiration dates
    expirations = analyzer.get_available_options()
    
    exp_date = input("\nEnter expiration date (YYYY-MM-DD): ")
    
    # Step 3: Get option chain and show available strikes
    print("\nFetching option chain...")
    calls, puts = analyzer.get_option_chain(exp_date)
    
    print("\nAvailable Call Strikes:")
    print(calls[['strike', 'lastPrice', 'volume']].head(10))
    
    print("\nAvailable Put Strikes:")
    print(puts[['strike', 'lastPrice', 'volume']].head(10))
    
    # Step 4: Choose specific option
    strike = float(input("\nEnter strike price: "))
    option_type = input("Enter option type (call/put): ").lower()
    
    # Step 5: Run complete analysis
    results = analyzer.analyze_option(strike, exp_date, option_type)
    

    print("ANALYSIS COMPLETE!")


if __name__ == "__main__":
    main()