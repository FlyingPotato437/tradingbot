#!/usr/bin/env python3
"""
Comprehensive Backtest Runner
Tests the trading system using historical data
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.backtesting.backtest_engine import BacktestEngine, BacktestResults


def setup_logging():
    """Setup logging for backtest"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )


def load_environment():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        console.print("[OK] Environment variables loaded", style="green")
    else:
        console.print("[WARNING] .env file not found", style="yellow")


def print_backtest_summary(results: BacktestResults):
    """Print comprehensive backtest summary"""
    
    # Summary table
    summary_table = Table(title="Backtest Summary", show_header=True, header_style="bold blue")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Total Return", f"{results.total_return:.2%}")
    summary_table.add_row("Annualized Return", f"{results.annualized_return:.2%}")
    summary_table.add_row("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
    summary_table.add_row("Max Drawdown", f"{results.max_drawdown:.2%}")
    summary_table.add_row("Total Trades", str(results.total_trades))
    summary_table.add_row("Win Rate", f"{results.win_rate:.2%}")
    summary_table.add_row("Profit Factor", f"{results.profit_factor:.2f}")
    summary_table.add_row("Avg Win", f"${results.avg_win:.2f}")
    summary_table.add_row("Avg Loss", f"${results.avg_loss:.2f}")
    
    console.print(summary_table)
    
    # Trade details
    if results.trades:
        trade_table = Table(title="Recent Trades (Last 10)", show_header=True, header_style="bold green")
        trade_table.add_column("Symbol", style="cyan")
        trade_table.add_column("Entry", style="white")
        trade_table.add_column("Exit", style="white")
        trade_table.add_column("P&L", style="white")
        trade_table.add_column("P&L %", style="white")
        trade_table.add_column("Days", style="white")
        trade_table.add_column("Confidence", style="white")
        
        # Show last 10 trades
        recent_trades = results.trades[-10:]
        for trade in recent_trades:
            pnl_color = "green" if trade.pnl > 0 else "red"
            trade_table.add_row(
                trade.symbol,
                trade.entry_date,
                trade.exit_date,
                f"[{pnl_color}]${trade.pnl:.2f}[/{pnl_color}]",
                f"[{pnl_color}]{trade.pnl_pct:.1f}%[/{pnl_color}]",
                str(trade.duration_days),
                f"{trade.confidence:.2f}"
            )
        
        console.print(trade_table)


def run_quick_backtest():
    """Run a quick backtest with default parameters"""
    console.print("[TEST] Running Quick Backtest...", style="bold blue")
    
    # Setup
    engine = BacktestEngine(initial_capital=100000.0)
    
    # Test parameters
    symbols = ['AAPL', 'NVDA', 'MSFT']
    start_date = '2024-01-01'
    end_date = '2024-06-30'
    
    console.print(f"Testing with symbols: {', '.join(symbols)}")
    console.print(f"Period: {start_date} to {end_date}")
    console.print(f"Initial capital: $100,000")
    
    # Run backtest
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)
        
        results = engine.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            decision_frequency=5  # Every 5 days
        )
        
        progress.remove_task(task)
    
    # Display results
    print_backtest_summary(results)
    
    # Save results
    engine.save_results(results, 'quick_backtest_results.json')
    console.print("[OK] Results saved to quick_backtest_results.json", style="green")
    
    return results


def run_comprehensive_backtest():
    """Run a comprehensive backtest with multiple scenarios"""
    console.print("[TEST] Running Comprehensive Backtest...", style="bold blue")
    
    scenarios = [
        {
            'name': 'Conservative Portfolio',
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'start_date': '2023-01-01',
            'end_date': '2024-01-01',
            'capital': 50000.0
        },
        {
            'name': 'Growth Portfolio',
            'symbols': ['NVDA', 'TSLA', 'AMD'],
            'start_date': '2023-01-01', 
            'end_date': '2024-01-01',
            'capital': 100000.0
        },
        {
            'name': 'Diversified Portfolio',
            'symbols': ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA'],
            'start_date': '2023-06-01',
            'end_date': '2024-06-01',
            'capital': 200000.0
        }
    ]
    
    all_results = []
    
    for scenario in scenarios:
        console.print(f"\n[SCENARIO] {scenario['name']}", style="bold yellow")
        
        engine = BacktestEngine(initial_capital=scenario['capital'])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Testing {scenario['name']}...", total=None)
            
            results = engine.run_backtest(
                symbols=scenario['symbols'],
                start_date=scenario['start_date'],
                end_date=scenario['end_date'],
                decision_frequency=3  # Every 3 days
            )
            
            progress.remove_task(task)
        
        # Store results
        results.scenario_name = scenario['name']
        all_results.append(results)
        
        # Print summary
        console.print(f"Total Return: {results.total_return:.2%}")
        console.print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        console.print(f"Max Drawdown: {results.max_drawdown:.2%}")
        console.print(f"Win Rate: {results.win_rate:.2%}")
        
        # Save individual results
        filename = f"backtest_{scenario['name'].lower().replace(' ', '_')}.json"
        engine.save_results(results, filename)
    
    # Compare scenarios
    console.print("\n[COMPARISON] Scenario Comparison", style="bold magenta")
    
    comparison_table = Table(title="Scenario Comparison", show_header=True, header_style="bold magenta")
    comparison_table.add_column("Scenario", style="cyan")
    comparison_table.add_column("Total Return", style="white")
    comparison_table.add_column("Sharpe Ratio", style="white")
    comparison_table.add_column("Max Drawdown", style="white")
    comparison_table.add_column("Win Rate", style="white")
    comparison_table.add_column("Trades", style="white")
    
    for results in all_results:
        comparison_table.add_row(
            results.scenario_name,
            f"{results.total_return:.2%}",
            f"{results.sharpe_ratio:.2f}",
            f"{results.max_drawdown:.2%}",
            f"{results.win_rate:.2%}",
            str(results.total_trades)
        )
    
    console.print(comparison_table)
    
    return all_results


def test_system_integration():
    """Test system integration and components"""
    console.print("[INTEGRATION] Testing System Components...", style="bold blue")
    
    try:
        # Test imports
        console.print("Testing imports...", style="yellow")
        from tradingagents.live_trading.live_trading_engine import LiveTradingEngine, TradingConfig
        from tradingagents.live_trading.mathematical_analyzer import MathematicalAnalyzer
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        console.print("[OK] All imports successful", style="green")
        
        # Test mathematical analyzer
        console.print("Testing mathematical analyzer...", style="yellow")
        analyzer = MathematicalAnalyzer()
        console.print("[OK] Mathematical analyzer created", style="green")
        
        # Test trading graph
        console.print("Testing trading graph...", style="yellow")
        trading_graph = TradingAgentsGraph(debug=False)
        console.print("[OK] Trading graph created", style="green")
        
        # Test backtest engine
        console.print("Testing backtest engine...", style="yellow")
        engine = BacktestEngine(initial_capital=10000.0)
        console.print("[OK] Backtest engine created", style="green")
        
        # Test data loading
        console.print("Testing data loading...", style="yellow")
        test_data = engine.get_historical_data('AAPL', '2024-01-01', '2024-01-31')
        if not test_data.empty:
            console.print(f"[OK] Loaded {len(test_data)} days of AAPL data", style="green")
        else:
            console.print("[WARNING] No test data loaded", style="yellow")
        
        console.print("[SUCCESS] All integration tests passed!", style="bold green")
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Integration test failed: {e}", style="bold red")
        return False


def main():
    """Main backtest runner"""
    console.print("=" * 60)
    console.print("TRADING SYSTEM BACKTEST RUNNER", style="bold blue")
    console.print("=" * 60)
    
    # Setup
    setup_logging()
    load_environment()
    
    # Test system integration first
    if not test_system_integration():
        console.print("[ERROR] System integration tests failed", style="bold red")
        return 1
    
    # Menu
    while True:
        console.print("\nBACKTEST MENU", style="bold blue")
        console.print("1. Quick Backtest (6 months)")
        console.print("2. Comprehensive Backtest (Multiple scenarios)")
        console.print("3. Custom Backtest")
        console.print("4. Exit")
        
        choice = input("\nSelect option [1-4]: ").strip()
        
        if choice == '1':
            try:
                results = run_quick_backtest()
                console.print(f"\n[SUMMARY] Quick backtest completed", style="bold green")
                console.print(f"Total return: {results.total_return:.2%}")
                
            except Exception as e:
                console.print(f"[ERROR] Quick backtest failed: {e}", style="bold red")
        
        elif choice == '2':
            try:
                results = run_comprehensive_backtest()
                console.print(f"\n[SUMMARY] Comprehensive backtest completed", style="bold green")
                
            except Exception as e:
                console.print(f"[ERROR] Comprehensive backtest failed: {e}", style="bold red")
        
        elif choice == '3':
            # Custom backtest
            console.print("\nCUSTOM BACKTEST SETUP", style="bold yellow")
            
            symbols_input = input("Enter symbols (comma-separated, e.g., AAPL,NVDA,MSFT): ")
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            if not symbols:
                console.print("[ERROR] No symbols provided", style="red")
                continue
            
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            
            try:
                capital = float(input("Enter initial capital (default: 100000): ") or "100000")
            except ValueError:
                capital = 100000.0
            
            try:
                engine = BacktestEngine(initial_capital=capital)
                
                console.print(f"\nRunning custom backtest...")
                console.print(f"Symbols: {', '.join(symbols)}")
                console.print(f"Period: {start_date} to {end_date}")
                console.print(f"Capital: ${capital:,.2f}")
                
                results = engine.run_backtest(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    decision_frequency=5
                )
                
                print_backtest_summary(results)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"custom_backtest_{timestamp}.json"
                engine.save_results(results, filename)
                console.print(f"[OK] Results saved to {filename}", style="green")
                
            except Exception as e:
                console.print(f"[ERROR] Custom backtest failed: {e}", style="bold red")
        
        elif choice == '4':
            console.print("Goodbye!", style="bold blue")
            break
        
        else:
            console.print("[ERROR] Invalid choice. Please select 1-4.", style="bold red")
    
    return 0


if __name__ == '__main__':
    console = Console()
    exit(main())