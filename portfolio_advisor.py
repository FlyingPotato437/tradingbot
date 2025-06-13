#!/usr/bin/env python3
"""
Portfolio Advisor - Standalone module for getting stock recommendations
"""
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


class PortfolioAdvisor:
    """AI-powered portfolio advisor using multi-agent analysis"""
    
    def __init__(self):
        """Initialize the portfolio advisor"""
        self.config = DEFAULT_CONFIG.copy()
        self.config["max_debate_rounds"] = 2
        self.config["max_risk_discuss_rounds"] = 2
        
    def get_stock_recommendation(self, symbol, investment_amount=10000):
        """Get AI recommendation for a specific stock"""
        try:
            # Initialize multi-agent system
            analysts = ["market", "news", "fundamentals", "social"]
            graph = TradingAgentsGraph(analysts, config=self.config, debug=False)
            
            # Run analysis
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            final_state, processed_decision = graph.propagate(symbol, analysis_date)
            
            # Extract recommendation
            decision_upper = processed_decision.upper()
            if 'BUY' in decision_upper:
                action = 'BUY'
            elif 'SELL' in decision_upper:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            return {
                'symbol': symbol,
                'action': action,
                'investment_amount': investment_amount,
                'decision': processed_decision,
                'market_analysis': final_state.get('market_report', 'N/A'),
                'news_analysis': final_state.get('news_report', 'N/A'),
                'fundamentals': final_state.get('fundamentals_report', 'N/A'),
                'sentiment': final_state.get('sentiment_report', 'N/A'),
                'final_recommendation': final_state.get('final_trade_decision', 'N/A')
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'action': 'ERROR',
                'error': str(e)
            }
    
    def compare_stocks(self, symbols, investment_amount=10000):
        """Compare multiple stocks and return recommendations"""
        results = []
        for symbol in symbols:
            print(f"Analyzing {symbol}...")
            result = self.get_stock_recommendation(symbol, investment_amount)
            results.append(result)
        return results
    
    def analyze_portfolio(self, holdings):
        """Analyze a portfolio of holdings
        
        Args:
            holdings: List of dicts with 'symbol', 'shares', 'avg_cost'
        
        Returns:
            List of recommendations for each holding
        """
        results = []
        for holding in holdings:
            symbol = holding['symbol']
            position_value = holding['shares'] * holding['avg_cost']
            print(f"Analyzing {symbol} (position value: ${position_value:,.2f})...")
            
            result = self.get_stock_recommendation(symbol, position_value)
            result['shares'] = holding['shares']
            result['avg_cost'] = holding['avg_cost']
            result['position_value'] = position_value
            
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    advisor = PortfolioAdvisor()
    
    print("Portfolio Advisor Example Usage")
    print("=" * 50)
    
    # Example 1: Single stock recommendation
    print("\n1. Single Stock Analysis:")
    result = advisor.get_stock_recommendation("AAPL", 5000)
    print(f"Symbol: {result['symbol']}")
    print(f"Recommendation: {result['action']}")
    print(f"Investment Amount: ${result['investment_amount']:,.2f}")
    
    # Example 2: Compare stocks
    print("\n2. Stock Comparison:")
    comparison = advisor.compare_stocks(["NVDA", "AMD"], 10000)
    for stock in comparison:
        print(f"{stock['symbol']}: {stock['action']}")
    
    # Example 3: Portfolio analysis
    print("\n3. Portfolio Analysis:")
    my_portfolio = [
        {'symbol': 'AAPL', 'shares': 100, 'avg_cost': 150},
        {'symbol': 'GOOGL', 'shares': 50, 'avg_cost': 100}
    ]
    portfolio_analysis = advisor.analyze_portfolio(my_portfolio)
    for holding in portfolio_analysis:
        print(f"{holding['symbol']}: {holding['action']} ({holding['shares']} shares @ ${holding['avg_cost']})")