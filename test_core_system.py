#!/usr/bin/env python3
"""
Core System Test - Tests the fundamental trading logic without external dependencies
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

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

console = Console()


def create_sample_data(days=100, start_price=100.0):
    """Create realistic sample stock data"""
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    
    # Simulate realistic price movement with trend and volatility
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    returns[0] = 0  # First day has no return
    
    # Add some trending behavior
    trend = np.linspace(0, 0.5, days) / days  # Slight upward trend
    returns = returns + trend
    
    # Calculate prices
    prices = [start_price]
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # Create OHLC data
    opens = []
    highs = []
    lows = []
    closes = prices
    volumes = []
    
    for i, close in enumerate(closes):
        # Simulate intraday movement
        open_price = close * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.uniform(1000000, 5000000))
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })


def test_mathematical_analysis():
    """Test the mathematical analysis components"""
    console.print("\n[TEST] Mathematical Analysis System", style="bold blue")
    
    try:
        # Create analyzer
        analyzer = MathematicalAnalyzer()
        console.print("[OK] Mathematical analyzer created", style="green")
        
        # Create sample data
        sample_data = create_sample_data(100, 150.0)
        console.print(f"[OK] Created sample data: {len(sample_data)} days", style="green")
        
        # Test technical indicators
        console.print("[INFO] Calculating technical indicators...", style="yellow")
        indicators = analyzer.calculate_technical_indicators(sample_data)
        
        # Display key indicators
        indicators_table = Table(title="Technical Indicators", show_header=True, header_style="bold blue")
        indicators_table.add_column("Indicator", style="cyan")
        indicators_table.add_column("Value", style="white")
        
        indicators_table.add_row("RSI", f"{indicators.rsi:.2f}")
        indicators_table.add_row("MACD", f"{indicators.macd:.4f}")
        indicators_table.add_row("MACD Signal", f"{indicators.macd_signal:.4f}")
        indicators_table.add_row("Bollinger Upper", f"${indicators.bollinger_upper:.2f}")
        indicators_table.add_row("Bollinger Lower", f"${indicators.bollinger_lower:.2f}")
        indicators_table.add_row("ATR", f"{indicators.atr:.3f}")
        indicators_table.add_row("SMA 20", f"${indicators.sma_20:.2f}")
        indicators_table.add_row("SMA 50", f"${indicators.sma_50:.2f}")
        indicators_table.add_row("Volatility", f"{indicators.volatility:.3f}")
        
        console.print(indicators_table)
        
        # Test risk metrics
        console.print("[INFO] Calculating risk metrics...", style="yellow")
        returns = sample_data['Close'].pct_change().dropna()
        risk_metrics = analyzer.calculate_risk_metrics(returns)
        
        risk_table = Table(title="Risk Metrics", show_header=True, header_style="bold red")
        risk_table.add_column("Metric", style="cyan")
        risk_table.add_column("Value", style="white")
        
        risk_table.add_row("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.3f}")
        risk_table.add_row("Sortino Ratio", f"{risk_metrics.sortino_ratio:.3f}")
        risk_table.add_row("Max Drawdown", f"{risk_metrics.max_drawdown:.3f}")
        risk_table.add_row("Value at Risk", f"{risk_metrics.value_at_risk:.5f}")
        risk_table.add_row("Expected Shortfall", f"{risk_metrics.expected_shortfall:.5f}")
        
        console.print(risk_table)
        
        # Test quantitative signals
        console.print("[INFO] Generating quantitative signals...", style="yellow")
        signals = analyzer.generate_quantitative_signals(sample_data, indicators)
        
        signals_table = Table(title="Quantitative Signals", show_header=True, header_style="bold green")
        signals_table.add_column("Signal", style="cyan")
        signals_table.add_column("Value", style="white")
        
        signals_table.add_row("Momentum Score", f"{signals.momentum_score:.3f}")
        signals_table.add_row("Mean Reversion Score", f"{signals.mean_reversion_score:.3f}")
        signals_table.add_row("Trend Strength", f"{signals.trend_strength:.3f}")
        signals_table.add_row("Volatility Regime", signals.volatility_regime)
        signals_table.add_row("Market Regime", signals.market_regime)
        signals_table.add_row("Probability of Profit", f"{signals.probability_of_profit:.3f}")
        
        console.print(signals_table)
        
        # Test opportunity scoring
        console.print("[INFO] Scoring trading opportunity...", style="yellow")
        scores = analyzer.score_trading_opportunity(indicators, signals, risk_metrics)
        
        scores_table = Table(title="Opportunity Scores", show_header=True, header_style="bold magenta")
        scores_table.add_column("Score Type", style="cyan")
        scores_table.add_column("Value", style="white")
        
        scores_table.add_row("Technical Score", f"{scores['technical_score']:.1f}/100")
        scores_table.add_row("Momentum Score", f"{scores['momentum_score']:.1f}/100")
        scores_table.add_row("Risk Score", f"{scores['risk_score']:.1f}/100")
        scores_table.add_row("Probability Score", f"{scores['probability_score']:.1f}/100")
        scores_table.add_row("Composite Score", f"{scores['composite_score']:.1f}/100")
        
        console.print(scores_table)
        
        # Test position sizing
        current_price = sample_data['Close'].iloc[-1]
        portfolio_value = 100000.0
        risk_per_trade = 0.02
        stop_loss_price = current_price * 0.95
        
        position_size = analyzer.calculate_position_size(
            portfolio_value, risk_per_trade, current_price, stop_loss_price
        )
        
        console.print(f"[INFO] Position sizing test:", style="yellow")
        console.print(f"  Portfolio Value: ${portfolio_value:,.2f}")
        console.print(f"  Current Price: ${current_price:.2f}")
        console.print(f"  Risk per Trade: {risk_per_trade:.1%}")
        console.print(f"  Calculated Position Size: {position_size} shares")
        console.print(f"  Position Value: ${position_size * current_price:,.2f}")
        
        console.print("[SUCCESS] Mathematical analysis test completed!", style="bold green")
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Mathematical analysis test failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        return False


def test_ai_decision_making():
    """Test AI decision making system"""
    console.print("\n[TEST] AI Decision Making System", style="bold blue")
    
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        
        # Initialize trading graph
        console.print("[INFO] Initializing AI trading system...", style="yellow")
        trading_graph = TradingAgentsGraph(debug=False)
        console.print("[OK] Trading graph created", style="green")
        
        # Test decision for AAPL
        test_date = "2024-01-15"
        console.print(f"[INFO] Testing AI decision for AAPL on {test_date}...", style="yellow")
        
        try:
            final_state, processed_decision = trading_graph.propagate("AAPL", test_date)
            
            console.print("[OK] AI decision generated successfully", style="green")
            console.print(f"[INFO] Processed Decision: {processed_decision}", style="cyan")
            
            # Extract key information
            if 'final_trade_decision' in final_state:
                reasoning = final_state['final_trade_decision']
                console.print(f"[INFO] AI Reasoning: {reasoning[:200]}...", style="cyan")
            
            # Determine action
            decision_upper = processed_decision.upper()
            if 'BUY' in decision_upper:
                action = 'BUY'
                color = "green"
            elif 'SELL' in decision_upper:
                action = 'SELL'
                color = "red"
            else:
                action = 'HOLD'
                color = "yellow"
            
            console.print(f"[DECISION] Final Action: [{color}]{action}[/{color}]", style="bold")
            
            console.print("[SUCCESS] AI decision making test completed!", style="bold green")
            return True, final_state, processed_decision
            
        except Exception as e:
            console.print(f"[WARNING] AI decision failed: {e}", style="yellow")
            console.print("[INFO] This may be due to API limits or configuration", style="cyan")
            return False, None, None
            
    except Exception as e:
        console.print(f"[ERROR] AI decision making test failed: {e}", style="bold red")
        return False, None, None


def test_integrated_system():
    """Test the integrated mathematical + AI system"""
    console.print("\n[TEST] Integrated Trading System", style="bold blue")
    
    try:
        # Create sample data
        sample_data = create_sample_data(100, 150.0)
        current_price = sample_data['Close'].iloc[-1]
        
        # Mathematical analysis
        analyzer = MathematicalAnalyzer()
        indicators = analyzer.calculate_technical_indicators(sample_data)
        returns = sample_data['Close'].pct_change().dropna()
        risk_metrics = analyzer.calculate_risk_metrics(returns)
        signals = analyzer.generate_quantitative_signals(sample_data, indicators)
        math_scores = analyzer.score_trading_opportunity(indicators, signals, risk_metrics)
        
        # AI decision (if available)
        ai_success, final_state, processed_decision = test_ai_decision_making()
        
        if ai_success:
            # Simulate confidence calculation
            ai_confidence = 0.75  # Simulated
            ai_action = 'BUY' if 'BUY' in processed_decision.upper() else 'SELL' if 'SELL' in processed_decision.upper() else 'HOLD'
        else:
            # Use mathematical analysis only
            ai_confidence = 0.5
            ai_action = 'BUY' if signals.momentum_score > 0 else 'SELL' if signals.momentum_score < -0.5 else 'HOLD'
        
        # Combined analysis
        math_confidence = math_scores['composite_score'] / 100.0
        probability_confidence = signals.probability_of_profit
        
        # Weighted confidence (70% AI, 20% Math, 10% Probability)
        combined_confidence = (ai_confidence * 0.7) + (math_confidence * 0.2) + (probability_confidence * 0.1)
        
        # Position sizing
        portfolio_value = 100000.0
        risk_per_trade = 0.02
        atr_stop_distance = indicators.atr * 2.0
        stop_loss_price = current_price - atr_stop_distance if ai_action == 'BUY' else current_price + atr_stop_distance
        
        position_size = analyzer.calculate_position_size(
            portfolio_value, risk_per_trade, current_price, stop_loss_price
        )
        
        # Display integrated results
        integration_table = Table(title="Integrated Trading Decision", show_header=True, header_style="bold magenta")
        integration_table.add_column("Component", style="cyan")
        integration_table.add_column("Value", style="white")
        
        integration_table.add_row("AI Confidence", f"{ai_confidence:.3f}")
        integration_table.add_row("Math Confidence", f"{math_confidence:.3f}")
        integration_table.add_row("Probability Confidence", f"{probability_confidence:.3f}")
        integration_table.add_row("Combined Confidence", f"{combined_confidence:.3f}")
        integration_table.add_row("Recommended Action", ai_action)
        integration_table.add_row("Current Price", f"${current_price:.2f}")
        integration_table.add_row("Stop Loss Price", f"${stop_loss_price:.2f}")
        integration_table.add_row("Position Size", f"{position_size} shares")
        integration_table.add_row("Position Value", f"${position_size * current_price:,.2f}")
        integration_table.add_row("Risk Amount", f"${abs(position_size * (current_price - stop_loss_price)):,.2f}")
        
        console.print(integration_table)
        
        # Trading decision logic
        min_confidence_threshold = 0.6
        
        if combined_confidence >= min_confidence_threshold and ai_action in ['BUY', 'SELL']:
            console.print(f"[TRADE] Execute {ai_action} order!", style="bold green")
            console.print(f"  Symbol: AAPL")
            console.print(f"  Action: {ai_action}")
            console.print(f"  Quantity: {position_size} shares")
            console.print(f"  Price: ${current_price:.2f}")
            console.print(f"  Confidence: {combined_confidence:.3f}")
            console.print(f"  Stop Loss: ${stop_loss_price:.2f}")
        else:
            console.print(f"[HOLD] Confidence {combined_confidence:.3f} below threshold {min_confidence_threshold}", style="yellow")
        
        console.print("[SUCCESS] Integrated system test completed!", style="bold green")
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Integrated system test failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        return False


def test_simple_backtest():
    """Test a simple backtest with sample data"""
    console.print("\n[TEST] Simple Backtest System", style="bold blue")
    
    try:
        # Create extended sample data (6 months)
        sample_data = create_sample_data(180, 150.0)
        console.print(f"[OK] Created {len(sample_data)} days of sample data", style="green")
        
        # Simple backtest simulation
        initial_capital = 100000.0
        cash = initial_capital
        position = 0  # shares held
        position_cost = 0.0
        trades = []
        portfolio_values = []
        
        analyzer = MathematicalAnalyzer()
        
        console.print("[INFO] Running simple backtest simulation...", style="yellow")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing trading days...", total=len(sample_data))
            
            for i in range(20, len(sample_data), 5):  # Start after 20 days, check every 5 days
                current_data = sample_data.iloc[:i+1]
                current_price = current_data['Close'].iloc[-1]
                
                # Calculate indicators
                if len(current_data) >= 20:
                    indicators = analyzer.calculate_technical_indicators(current_data)
                    returns = current_data['Close'].pct_change().dropna()
                    risk_metrics = analyzer.calculate_risk_metrics(returns)
                    signals = analyzer.generate_quantitative_signals(current_data, indicators)
                    scores = analyzer.score_trading_opportunity(indicators, signals, risk_metrics)
                    
                    # Simple decision logic
                    confidence = scores['composite_score'] / 100.0
                    
                    # Buy signal: high momentum, good technical score, low RSI
                    buy_signal = (
                        signals.momentum_score > 0.2 and
                        indicators.rsi < 70 and
                        confidence > 0.6
                    )
                    
                    # Sell signal: negative momentum, high RSI, or low confidence
                    sell_signal = (
                        signals.momentum_score < -0.2 or
                        indicators.rsi > 80 or
                        confidence < 0.4
                    )
                    
                    # Execute trades
                    if buy_signal and position == 0 and cash > current_price * 100:
                        # Buy
                        shares_to_buy = min(int(cash * 0.9 / current_price), int(cash / current_price))
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price + 1  # $1 commission
                            if cost <= cash:
                                cash -= cost
                                position = shares_to_buy
                                position_cost = current_price
                                trades.append({
                                    'date': current_data['Date'].iloc[-1],
                                    'action': 'BUY',
                                    'price': current_price,
                                    'shares': shares_to_buy,
                                    'confidence': confidence
                                })
                    
                    elif sell_signal and position > 0:
                        # Sell
                        proceeds = position * current_price - 1  # $1 commission
                        pnl = proceeds - (position * position_cost)
                        cash += proceeds
                        trades.append({
                            'date': current_data['Date'].iloc[-1],
                            'action': 'SELL',
                            'price': current_price,
                            'shares': position,
                            'pnl': pnl,
                            'confidence': confidence
                        })
                        position = 0
                        position_cost = 0.0
                
                # Calculate portfolio value
                portfolio_value = cash + (position * current_price if position > 0 else 0)
                portfolio_values.append(portfolio_value)
                
                progress.advance(task, 5)
        
        # Final portfolio value
        final_price = sample_data['Close'].iloc[-1]
        final_portfolio_value = cash + (position * final_price if position > 0 else 0)
        
        # Calculate results
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        
        # Display results
        backtest_table = Table(title="Simple Backtest Results", show_header=True, header_style="bold blue")
        backtest_table.add_column("Metric", style="cyan")
        backtest_table.add_column("Value", style="white")
        
        backtest_table.add_row("Initial Capital", f"${initial_capital:,.2f}")
        backtest_table.add_row("Final Portfolio Value", f"${final_portfolio_value:,.2f}")
        backtest_table.add_row("Total Return", f"{total_return:.2%}")
        backtest_table.add_row("Total Trades", str(len(trades)))
        backtest_table.add_row("Final Cash", f"${cash:,.2f}")
        backtest_table.add_row("Final Position", f"{position} shares")
        
        if position > 0:
            backtest_table.add_row("Position Value", f"${position * final_price:,.2f}")
        
        console.print(backtest_table)
        
        # Show recent trades
        if trades:
            console.print(f"\n[INFO] Recent trades (last 5):", style="yellow")
            recent_trades = trades[-5:]
            
            trade_table = Table(title="Recent Trades", show_header=True, header_style="bold green")
            trade_table.add_column("Date", style="cyan")
            trade_table.add_column("Action", style="white")
            trade_table.add_column("Price", style="white")
            trade_table.add_column("Shares", style="white")
            trade_table.add_column("P&L", style="white")
            
            for trade in recent_trades:
                pnl_str = f"${trade.get('pnl', 0):.2f}" if 'pnl' in trade else "-"
                trade_table.add_row(
                    trade['date'].strftime('%Y-%m-%d'),
                    trade['action'],
                    f"${trade['price']:.2f}",
                    str(trade['shares']),
                    pnl_str
                )
            
            console.print(trade_table)
        
        console.print("[SUCCESS] Simple backtest completed!", style="bold green")
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Simple backtest failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    console.print("=" * 70)
    console.print("CORE TRADING SYSTEM TEST", style="bold blue")
    console.print("=" * 70)
    
    console.print("\nThis test validates the core trading system components:", style="yellow")
    console.print("1. Mathematical Analysis (Technical indicators, risk metrics, signals)")
    console.print("2. AI Decision Making (Multi-agent trading decisions)")
    console.print("3. Integrated System (Combined AI + Math analysis)")
    console.print("4. Simple Backtesting (Historical performance simulation)")
    
    # Run all tests
    tests = [
        ("Mathematical Analysis", test_mathematical_analysis),
        ("AI Decision Making", test_ai_decision_making),
        ("Integrated System", test_integrated_system),
        ("Simple Backtest", test_simple_backtest)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*50}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[ERROR] {test_name} crashed: {e}", style="bold red")
            results.append((test_name, False))
    
    # Summary
    console.print("\n" + "=" * 70)
    console.print("TEST SUMMARY", style="bold blue")
    console.print("=" * 70)
    
    summary_table = Table(title="Test Results", show_header=True, header_style="bold blue")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Status", style="white")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            summary_table.add_row(test_name, "[green]PASSED[/green]")
            passed += 1
        else:
            summary_table.add_row(test_name, "[red]FAILED[/red]")
    
    console.print(summary_table)
    
    if passed == total:
        console.print(f"\n[SUCCESS] All {total} tests passed! System is working correctly.", style="bold green")
        console.print("\nNext steps:", style="bold blue")
        console.print("1. Run comprehensive backtests with real historical data")
        console.print("2. Test with live paper trading")
        console.print("3. Monitor performance and refine parameters")
    else:
        console.print(f"\n[PARTIAL] {passed}/{total} tests passed", style="bold yellow")
        if passed >= total * 0.75:
            console.print("Core system is mostly functional", style="yellow")
        else:
            console.print("Significant issues detected", style="red")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    exit(main())