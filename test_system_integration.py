#!/usr/bin/env python3
"""
System Integration Test
Tests all components working together with real credentials
"""

import os
import sys
import time
import logging
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.live_trading.live_trading_engine import LiveTradingEngine, TradingConfig
from tradingagents.live_trading.alpaca_executor import AlpacaExecutor
from tradingagents.backtesting.backtest_engine import BacktestEngine


def load_environment():
    """Load environment variables"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        console.print("[OK] Environment variables loaded", style="green")
        return True
    else:
        console.print("[ERROR] .env file not found", style="red")
        return False


def test_credentials():
    """Test API credentials"""
    console.print("\n[TEST] Testing API Credentials...", style="bold blue")
    
    # Check required environment variables
    required_vars = ['ALPACA_API_KEY', 'OPENAI_API_KEY', 'FINNHUB_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            console.print(f"[OK] {var} is set", style="green")
    
    if missing_vars:
        console.print(f"[ERROR] Missing environment variables: {', '.join(missing_vars)}", style="red")
        return False
    
    return True


def test_alpaca_connection():
    """Test Alpaca API connection"""
    console.print("\n[TEST] Testing Alpaca Connection...", style="bold blue")
    
    try:
        executor = AlpacaExecutor(paper_trading=True)
        
        if executor.api is None:
            console.print("[ERROR] Failed to initialize Alpaca API", style="red")
            return False
        
        # Test account info
        account_info = executor.get_account_info()
        if account_info:
            console.print(f"[OK] Alpaca connected - Account: {account_info['account_number']}", style="green")
            console.print(f"[OK] Account Status: {account_info['status']}", style="green")
            console.print(f"[OK] Buying Power: ${account_info['buying_power']:,.2f}", style="green")
            
            # Test market status
            market_open = executor.is_market_open()
            console.print(f"[INFO] Market Open: {market_open}", style="cyan")
            
            return True
        else:
            console.print("[ERROR] Failed to get account info", style="red")
            return False
            
    except Exception as e:
        console.print(f"[ERROR] Alpaca connection failed: {e}", style="red")
        return False


def test_live_trading_engine():
    """Test live trading engine"""
    console.print("\n[TEST] Testing Live Trading Engine...", style="bold blue")
    
    try:
        # Create test configuration
        config = TradingConfig(
            trading_symbols=['AAPL'],
            paper_trading=True,
            initial_capital=10000.0,
            use_alpaca=True,
            use_mathematical_analysis=True,
            min_confidence_threshold=0.6
        )
        
        # Initialize engine
        engine = LiveTradingEngine(config)
        console.print("[OK] Live trading engine created", style="green")
        
        # Test portfolio
        portfolio = engine.get_portfolio_status()
        console.print(f"[OK] Portfolio initialized: ${portfolio['total_value']:,.2f}", style="green")
        
        # Test market data
        engine.market_data_stream.start()
        console.print("[OK] Market data stream started", style="green")
        
        # Wait for data
        console.print("[INFO] Waiting for market data...", style="yellow")
        time.sleep(3)
        
        latest_data = engine.market_data_stream.get_latest_data('AAPL')
        if latest_data:
            console.print(f"[OK] Market data received - AAPL: ${latest_data.price:.2f}", style="green")
        else:
            console.print("[WARNING] No market data (may be outside trading hours)", style="yellow")
        
        # Test decision making
        console.print("[INFO] Testing AI decision making...", style="yellow")
        try:
            decision = engine.force_decision('AAPL')
            if decision:
                console.print(f"[OK] AI decision: {decision.action} {decision.quantity} shares", style="green")
                console.print(f"[OK] Confidence: {decision.confidence:.2f}", style="green")
            else:
                console.print("[WARNING] No decision generated", style="yellow")
        except Exception as e:
            console.print(f"[WARNING] AI decision failed: {e}", style="yellow")
        
        # Cleanup
        engine.market_data_stream.stop()
        console.print("[OK] Live trading engine test completed", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Live trading engine test failed: {e}", style="red")
        return False


def test_backtest_engine():
    """Test backtest engine"""
    console.print("\n[TEST] Testing Backtest Engine...", style="bold blue")
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine(initial_capital=50000.0)
        console.print("[OK] Backtest engine created", style="green")
        
        # Test data loading
        test_data = engine.get_historical_data('AAPL', '2024-01-01', '2024-01-31')
        if not test_data.empty:
            console.print(f"[OK] Historical data loaded: {len(test_data)} days", style="green")
        else:
            console.print("[WARNING] No historical data available", style="yellow")
            return False
        
        # Run mini backtest
        console.print("[INFO] Running mini backtest...", style="yellow")
        results = engine.run_backtest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            decision_frequency=5
        )
        
        console.print(f"[OK] Backtest completed", style="green")
        console.print(f"[OK] Total Return: {results.total_return:.2%}", style="green")
        console.print(f"[OK] Total Trades: {results.total_trades}", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Backtest engine test failed: {e}", style="red")
        return False


def test_mathematical_analysis():
    """Test mathematical analysis components"""
    console.print("\n[TEST] Testing Mathematical Analysis...", style="bold blue")
    
    try:
        from tradingagents.live_trading.mathematical_analyzer import MathematicalAnalyzer
        import pandas as pd
        import numpy as np
        
        # Create analyzer
        analyzer = MathematicalAnalyzer()
        console.print("[OK] Mathematical analyzer created", style="green")
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': prices + np.random.randn(100) * 0.1,
            'High': prices + np.abs(np.random.randn(100) * 0.2),
            'Low': prices - np.abs(np.random.randn(100) * 0.2),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
        
        # Test technical indicators
        indicators = analyzer.calculate_technical_indicators(sample_data)
        console.print(f"[OK] Technical indicators calculated - RSI: {indicators.rsi:.1f}", style="green")
        
        # Test risk metrics
        returns = sample_data['Close'].pct_change().dropna()
        risk_metrics = analyzer.calculate_risk_metrics(returns)
        console.print(f"[OK] Risk metrics calculated - Sharpe: {risk_metrics.sharpe_ratio:.2f}", style="green")
        
        # Test signals
        signals = analyzer.generate_quantitative_signals(sample_data, indicators)
        console.print(f"[OK] Signals generated - Momentum: {signals.momentum_score:.2f}", style="green")
        
        # Test scoring
        scores = analyzer.score_trading_opportunity(indicators, signals, risk_metrics)
        console.print(f"[OK] Opportunity scored - Composite: {scores['composite_score']:.1f}/100", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Mathematical analysis test failed: {e}", style="red")
        return False


def run_integration_test():
    """Run comprehensive integration test"""
    console.print("=" * 60)
    console.print("TRADING SYSTEM INTEGRATION TEST", style="bold blue")
    console.print("=" * 60)
    
    # Load environment
    if not load_environment():
        return False
    
    # Test credentials
    if not test_credentials():
        return False
    
    # Run all tests
    tests = [
        ("Mathematical Analysis", test_mathematical_analysis),
        ("Alpaca Connection", test_alpaca_connection),
        ("Live Trading Engine", test_live_trading_engine),
        ("Backtest Engine", test_backtest_engine),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[ERROR] {test_name} crashed: {e}", style="red")
            results.append((test_name, False))
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("INTEGRATION TEST SUMMARY", style="bold blue")
    console.print("=" * 60)
    
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
    
    # Overall result
    if passed == total:
        console.print(f"\n[SUCCESS] All {total} tests passed!", style="bold green")
        console.print("System is ready for trading!", style="bold green")
        return True
    else:
        console.print(f"\n[PARTIAL] {passed}/{total} tests passed", style="bold yellow")
        console.print("Some components may not work correctly", style="yellow")
        return False


def main():
    """Main function"""
    success = run_integration_test()
    
    if success:
        console.print("\nNext steps:", style="bold blue")
        console.print("1. Run backtests: python run_backtest.py")
        console.print("2. Start paper trading: python live_trading_main.py")
        console.print("3. Monitor performance and adjust parameters")
        
        return 0
    else:
        console.print("\nPlease fix the failing tests before proceeding", style="yellow")
        return 1


if __name__ == '__main__':
    console = Console()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    exit(main())