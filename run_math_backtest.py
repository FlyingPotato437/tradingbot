#!/usr/bin/env python3
"""
Mathematical Analysis Backtest - Using real historical data with mathematical analysis
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
def load_env():
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

from tradingagents.live_trading.mathematical_analyzer import MathematicalAnalyzer
from tradingagents.dataflows.config import DATA_DIR

console = Console()


class MathematicalBacktester:
    """Backtester using mathematical analysis only"""
    
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.analyzer = MathematicalAnalyzer()
        
    def load_historical_data(self, symbol, start_date='2023-01-01', end_date='2024-06-30'):
        """Load historical data from CSV files"""
        try:
            # Try different possible data file locations
            possible_paths = [
                f"test_data/{symbol}_data.csv",  # Downloaded test data
                os.path.join(DATA_DIR, f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv"),
                os.path.join(DATA_DIR, f"data_cache/{symbol}-YFin-data-2010-06-13-2025-06-13.csv"),
                os.path.join(os.path.dirname(__file__), f"tradingagents/dataflows/data_cache/{symbol}-YFin-data-2010-06-13-2025-06-13.csv"),
                f"tradingagents/dataflows/data_cache/{symbol}-YFin-data-2010-06-13-2025-06-13.csv"
            ]
            
            data_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_file = path
                    break
            
            if not data_file:
                console.print(f"[ERROR] No data file found for {symbol}. Tried:", style="red")
                for path in possible_paths:
                    console.print(f"  {path}")
                return pd.DataFrame()
            
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)  # Remove timezone info
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()
            filtered_df = filtered_df.sort_values('Date').reset_index(drop=True)
            
            console.print(f"[OK] Loaded {len(filtered_df)} days of {symbol} data", style="green")
            return filtered_df
            
        except Exception as e:
            console.print(f"[ERROR] Failed to load data for {symbol}: {e}", style="red")
            return pd.DataFrame()
    
    def run_backtest(self, symbol, start_date='2023-01-01', end_date='2024-06-30', decision_frequency=3):
        """Run mathematical analysis backtest"""
        
        console.print(f"\n[BACKTEST] Running mathematical backtest for {symbol}", style="bold blue")
        console.print(f"Period: {start_date} to {end_date}")
        console.print(f"Decision frequency: Every {decision_frequency} days")
        console.print(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Load data
        data = self.load_historical_data(symbol, start_date, end_date)
        if data.empty:
            return None
        
        # Initialize backtest state
        cash = self.initial_capital
        position = 0  # shares held
        position_cost = 0.0
        trades = []
        portfolio_values = []
        daily_returns = []
        
        # Tracking
        max_portfolio_value = self.initial_capital
        max_drawdown = 0.0
        
        console.print(f"\n[INFO] Starting backtest simulation...", style="yellow")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            total_decisions = len(range(50, len(data), decision_frequency))
            task = progress.add_task("Processing trading decisions...", total=total_decisions)
            
            for i in range(50, len(data), decision_frequency):  # Start after 50 days for indicators
                current_data = data.iloc[:i+1]
                current_price = current_data['Close'].iloc[-1]
                previous_price = current_data['Close'].iloc[-2] if i > 0 else current_price
                
                # Calculate portfolio value
                portfolio_value = cash + (position * current_price)
                portfolio_values.append(portfolio_value)
                
                # Track performance
                if portfolio_value > max_portfolio_value:
                    max_portfolio_value = portfolio_value
                
                current_drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                
                # Calculate daily return
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
                
                # Mathematical analysis
                try:
                    # Technical indicators
                    indicators = self.analyzer.calculate_technical_indicators(current_data)
                    
                    # Risk metrics
                    returns = current_data['Close'].pct_change().dropna()
                    risk_metrics = self.analyzer.calculate_risk_metrics(returns)
                    
                    # Quantitative signals
                    signals = self.analyzer.generate_quantitative_signals(current_data, indicators)
                    
                    # Opportunity scoring
                    scores = self.analyzer.score_trading_opportunity(indicators, signals, risk_metrics)
                    
                    # Trading logic
                    confidence = scores['composite_score'] / 100.0
                    momentum = signals.momentum_score
                    rsi = indicators.rsi
                    trend_strength = signals.trend_strength
                    
                    # Enhanced trading signals
                    strong_buy_signal = (
                        momentum > 0.3 and
                        rsi < 65 and
                        confidence > 0.7 and
                        trend_strength > 0.6 and
                        signals.market_regime in ['TRENDING_UP', 'WEAK_TREND']
                    )
                    
                    buy_signal = (
                        momentum > 0.1 and
                        rsi < 70 and
                        confidence > 0.6 and
                        not strong_buy_signal
                    )
                    
                    sell_signal = (
                        momentum < -0.2 or
                        rsi > 75 or
                        confidence < 0.4 or
                        trend_strength < 0.2
                    )
                    
                    # Position sizing based on confidence and volatility
                    base_position_size = 0.15  # 15% of portfolio
                    confidence_multiplier = min(confidence * 1.5, 1.0)
                    volatility_adjustment = 1.0 / (1.0 + indicators.volatility)
                    
                    adjusted_position_size = base_position_size * confidence_multiplier * volatility_adjustment
                    
                    # Execute trades
                    if strong_buy_signal and position < portfolio_value * 0.8 / current_price:
                        # Strong buy: larger position
                        max_shares = int(portfolio_value * adjusted_position_size * 1.5 / current_price)
                        shares_to_buy = min(max_shares, int(cash * 0.9 / current_price))
                        
                        if shares_to_buy > 0 and cash >= shares_to_buy * current_price + 5:
                            cost = shares_to_buy * current_price + 5  # $5 commission
                            cash -= cost
                            position += shares_to_buy
                            
                            trades.append({
                                'date': current_data['Date'].iloc[-1],
                                'action': 'STRONG_BUY',
                                'price': current_price,
                                'shares': shares_to_buy,
                                'confidence': confidence,
                                'momentum': momentum,
                                'rsi': rsi,
                                'reasoning': f"Strong buy signal: momentum={momentum:.3f}, rsi={rsi:.1f}, confidence={confidence:.3f}"
                            })
                    
                    elif buy_signal and position < portfolio_value * 0.6 / current_price:
                        # Regular buy
                        max_shares = int(portfolio_value * adjusted_position_size / current_price)
                        shares_to_buy = min(max_shares, int(cash * 0.7 / current_price))
                        
                        if shares_to_buy > 0 and cash >= shares_to_buy * current_price + 5:
                            cost = shares_to_buy * current_price + 5
                            cash -= cost
                            position += shares_to_buy
                            
                            trades.append({
                                'date': current_data['Date'].iloc[-1],
                                'action': 'BUY',
                                'price': current_price,
                                'shares': shares_to_buy,
                                'confidence': confidence,
                                'momentum': momentum,
                                'rsi': rsi,
                                'reasoning': f"Buy signal: momentum={momentum:.3f}, rsi={rsi:.1f}, confidence={confidence:.3f}"
                            })
                    
                    elif sell_signal and position > 0:
                        # Sell position
                        sell_portion = 0.5 if confidence > 0.3 else 0.8  # Sell more if very low confidence
                        shares_to_sell = max(1, int(position * sell_portion))
                        
                        proceeds = shares_to_sell * current_price - 5  # $5 commission
                        cash += proceeds
                        
                        # Calculate P&L (simplified)
                        avg_cost = (portfolio_value - cash - (position - shares_to_sell) * current_price) / shares_to_sell if shares_to_sell > 0 else current_price
                        pnl = (current_price - avg_cost) * shares_to_sell - 5
                        
                        position -= shares_to_sell
                        
                        trades.append({
                            'date': current_data['Date'].iloc[-1],
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares_to_sell,
                            'pnl': pnl,
                            'confidence': confidence,
                            'momentum': momentum,
                            'rsi': rsi,
                            'reasoning': f"Sell signal: momentum={momentum:.3f}, rsi={rsi:.1f}, confidence={confidence:.3f}"
                        })
                
                except Exception as e:
                    console.print(f"[WARNING] Analysis failed for day {i}: {e}", style="yellow")
                
                progress.advance(task)
        
        # Calculate final results
        final_price = data['Close'].iloc[-1]
        final_portfolio_value = cash + (position * final_price)
        
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        days_traded = len(portfolio_values)
        annualized_return = (1 + total_return) ** (252 / days_traded) - 1 if days_traded > 0 else 0
        
        # Sharpe ratio
        if daily_returns and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Trade statistics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_cash': cash,
            'final_position': position,
            'final_position_value': position * final_price if position > 0 else 0,
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    
    def display_results(self, results):
        """Display backtest results"""
        if not results:
            console.print("[ERROR] No results to display", style="red")
            return
        
        # Summary table
        summary_table = Table(title=f"Mathematical Backtest Results - {results['symbol']}", 
                             show_header=True, header_style="bold blue")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Period", f"{results['start_date']} to {results['end_date']}")
        summary_table.add_row("Initial Capital", f"${results['initial_capital']:,.2f}")
        summary_table.add_row("Final Portfolio Value", f"${results['final_portfolio_value']:,.2f}")
        summary_table.add_row("Total Return", f"{results['total_return']:.2%}")
        summary_table.add_row("Annualized Return", f"{results['annualized_return']:.2%}")
        summary_table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
        summary_table.add_row("Max Drawdown", f"{results['max_drawdown']:.2%}")
        summary_table.add_row("Total Trades", str(results['total_trades']))
        summary_table.add_row("Win Rate", f"{results['win_rate']:.2%}")
        summary_table.add_row("Average Win", f"${results['avg_win']:,.2f}")
        summary_table.add_row("Average Loss", f"${results['avg_loss']:,.2f}")
        summary_table.add_row("Final Cash", f"${results['final_cash']:,.2f}")
        summary_table.add_row("Final Position", f"{results['final_position']} shares")
        summary_table.add_row("Position Value", f"${results['final_position_value']:,.2f}")
        
        console.print(summary_table)
        
        # Recent trades
        if results['trades']:
            console.print(f"\n[INFO] Recent trades (last 10):", style="yellow")
            recent_trades = results['trades'][-10:]
            
            trade_table = Table(title="Recent Trades", show_header=True, header_style="bold green")
            trade_table.add_column("Date", style="cyan")
            trade_table.add_column("Action", style="white")
            trade_table.add_column("Price", style="white")
            trade_table.add_column("Shares", style="white")
            trade_table.add_column("P&L", style="white")
            trade_table.add_column("Confidence", style="white")
            
            for trade in recent_trades:
                pnl_str = f"${trade.get('pnl', 0):.2f}" if 'pnl' in trade else "-"
                pnl_color = "green" if trade.get('pnl', 0) > 0 else "red" if trade.get('pnl', 0) < 0 else "white"
                
                action_color = "green" if "BUY" in trade['action'] else "red"
                
                trade_table.add_row(
                    trade['date'].strftime('%Y-%m-%d'),
                    f"[{action_color}]{trade['action']}[/{action_color}]",
                    f"${trade['price']:.2f}",
                    str(trade['shares']),
                    f"[{pnl_color}]{pnl_str}[/{pnl_color}]",
                    f"{trade['confidence']:.3f}"
                )
            
            console.print(trade_table)


def main():
    """Main backtest runner"""
    console.print("=" * 70)
    console.print("MATHEMATICAL ANALYSIS BACKTEST", style="bold blue")
    console.print("=" * 70)
    
    console.print("\nThis backtest uses pure mathematical analysis:", style="yellow")
    console.print("• Technical indicators (RSI, MACD, Bollinger Bands, ATR)")
    console.print("• Risk metrics (Sharpe ratio, drawdown, VaR)")
    console.print("• Quantitative signals (momentum, mean reversion, trend)")
    console.print("• Dynamic position sizing based on confidence and volatility")
    
    # Initialize backtester
    backtester = MathematicalBacktester(initial_capital=100000.0)
    
    # Test scenarios
    scenarios = [
        {
            'symbol': 'AAPL',
            'start_date': '2023-01-01',
            'end_date': '2024-06-30',
            'name': 'AAPL 18-month test'
        },
        {
            'symbol': 'NVDA',
            'start_date': '2023-01-01',
            'end_date': '2024-06-30',
            'name': 'NVDA 18-month test'
        },
        {
            'symbol': 'MSFT',
            'start_date': '2023-01-01',
            'end_date': '2024-06-30',
            'name': 'MSFT 18-month test'
        }
    ]
    
    all_results = []
    
    for scenario in scenarios:
        console.print(f"\n{'='*60}")
        console.print(f"[SCENARIO] {scenario['name']}", style="bold yellow")
        console.print(f"{'='*60}")
        
        results = backtester.run_backtest(
            symbol=scenario['symbol'],
            start_date=scenario['start_date'],
            end_date=scenario['end_date'],
            decision_frequency=3
        )
        
        if results:
            all_results.append(results)
            backtester.display_results(results)
        else:
            console.print(f"[ERROR] Backtest failed for {scenario['symbol']}", style="red")
    
    # Summary comparison
    if all_results:
        console.print(f"\n{'='*70}")
        console.print("BACKTEST COMPARISON", style="bold blue")
        console.print(f"{'='*70}")
        
        comparison_table = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
        comparison_table.add_column("Symbol", style="cyan")
        comparison_table.add_column("Total Return", style="white")
        comparison_table.add_column("Annualized Return", style="white")
        comparison_table.add_column("Sharpe Ratio", style="white")
        comparison_table.add_column("Max Drawdown", style="white")
        comparison_table.add_column("Win Rate", style="white")
        comparison_table.add_column("Trades", style="white")
        
        for results in all_results:
            return_color = "green" if results['total_return'] > 0 else "red"
            sharpe_color = "green" if results['sharpe_ratio'] > 1.0 else "yellow" if results['sharpe_ratio'] > 0.5 else "red"
            
            comparison_table.add_row(
                results['symbol'],
                f"[{return_color}]{results['total_return']:.2%}[/{return_color}]",
                f"{results['annualized_return']:.2%}",
                f"[{sharpe_color}]{results['sharpe_ratio']:.3f}[/{sharpe_color}]",
                f"{results['max_drawdown']:.2%}",
                f"{results['win_rate']:.2%}",
                str(results['total_trades'])
            )
        
        console.print(comparison_table)
        
        # Overall assessment
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
        
        console.print(f"\n[SUMMARY] Overall Performance:", style="bold blue")
        console.print(f"Average Return: {avg_return:.2%}")
        console.print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
        
        if avg_return > 0.1 and avg_sharpe > 0.8:
            console.print("[EXCELLENT] Strong mathematical trading performance!", style="bold green")
        elif avg_return > 0.05 and avg_sharpe > 0.5:
            console.print("[GOOD] Solid mathematical trading performance", style="green")
        elif avg_return > 0:
            console.print("[MODERATE] Positive but modest performance", style="yellow")
        else:
            console.print("[POOR] Mathematical system needs improvement", style="red")
    
    console.print(f"\n[COMPLETE] Mathematical backtest finished!", style="bold blue")
    return 0


if __name__ == '__main__':
    exit(main())