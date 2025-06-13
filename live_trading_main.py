#!/usr/bin/env python3
"""
Live Trading Main Entry Point
Real-time autonomous trading system using TradingAgents
"""

import asyncio
import signal
import sys
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.live_trading.live_trading_engine import LiveTradingEngine, TradingConfig
from tradingagents.live_trading.cli import LiveTradingCLI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('live_trading.log'),
            logging.StreamHandler()
        ]
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    console.print("\n🛑 Received shutdown signal. Stopping trading...", style="bold yellow")
    sys.exit(0)


def print_banner():
    """Print application banner"""
    banner_text = Text("LIVE TRADING AGENTS", style="bold white on blue")
    banner_text.append("\nAutonomous Real-Time Trading System", style="italic white")
    
    panel = Panel(
        banner_text,
        title="TradingAgents v2.0",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def print_warning():
    """Print important warning about live trading"""
    warning_text = Text("IMPORTANT WARNING", style="bold red")
    warning_text.append("\n\nThis system can execute real trades and lose real money.")
    warning_text.append("\nAlways start with paper trading to test your strategies.")
    warning_text.append("\nUse live trading only after thorough testing and with money you can afford to lose.")
    warning_text.append("\n\nThe system makes autonomous trading decisions based on AI analysis.")
    warning_text.append("\nPast performance does not guarantee future results.")
    
    panel = Panel(
        warning_text,
        title="Risk Warning",
        border_style="red",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def print_features():
    """Print system features"""
    features_text = Text("SYSTEM FEATURES", style="bold green")
    features_text.append("\n\n• Real-time market data streaming")
    features_text.append("\n• AI-powered trading decisions")
    features_text.append("\n• Automated risk management")
    features_text.append("\n• Portfolio management & tracking")
    features_text.append("\n• Multi-agent analysis (news, technical, fundamental)")
    features_text.append("\n• Paper trading simulation")
    features_text.append("\n• Live monitoring dashboard")
    features_text.append("\n• Stop-loss & take-profit automation")
    features_text.append("\n• Position sizing & diversification")
    features_text.append("\n• Real-time risk alerts")
    
    panel = Panel(
        features_text,
        title="Features",
        border_style="green",
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def quick_start_demo():
    """Run a quick demonstration"""
    console.print("[DEMO] Starting Quick Demo...", style="bold blue")
    
    try:
        # Create a demo configuration
        config = TradingConfig(
            trading_symbols=['AAPL', 'NVDA'],
            paper_trading=True,
            initial_capital=10000.0,
            decision_interval=60,  # 1 minute for demo
            max_position_size=0.2,  # 20% max position
        )
        
        # Initialize engine
        console.print("[INIT] Initializing trading engine...", style="bold yellow")
        engine = LiveTradingEngine(config)
        
        # Start trading
        console.print("[START] Starting demo trading...", style="bold green")
        engine.start_trading()
        
        # Show initial status
        console.print("\nInitial Portfolio Status:", style="bold blue")
        portfolio_status = engine.get_portfolio_status()
        console.print(f"Total Value: ${portfolio_status['total_value']:,.2f}")
        console.print(f"Cash: ${portfolio_status['cash']:,.2f}")
        
        # Force a decision for demo
        console.print("\n[ANALYZE] Forcing trading decision for AAPL...", style="bold yellow")
        decision = engine.force_decision('AAPL')
        
        if decision:
            console.print(f"Decision: {decision.action} {decision.quantity} shares at ${decision.price:.2f}")
            console.print(f"Confidence: {decision.confidence:.2f}")
            console.print(f"Reasoning: {decision.reasoning[:100]}...")
        else:
            console.print("[ERROR] No decision generated", style="bold red")
        
        # Show final status
        console.print("\nFinal Portfolio Status:", style="bold blue")
        final_status = engine.get_portfolio_status()
        console.print(f"Total Value: ${final_status['total_value']:,.2f}")
        console.print(f"Total Trades: {final_status['total_trades']}")
        
        # Stop engine
        engine.stop_trading()
        console.print("\n[SUCCESS] Demo completed successfully!", style="bold green")
        
    except Exception as e:
        console.print(f"[ERROR] Demo failed: {e}", style="bold red")


def interactive_setup():
    """Interactive setup for live trading"""
    console.print("Live Trading Setup", style="bold blue")
    console.print("Let's configure your trading parameters...\n")
    
    # Get trading symbols
    symbols_input = console.input("Enter trading symbols (comma-separated, e.g., AAPL,NVDA,TSLA): ")
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    if not symbols:
        symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
        console.print(f"Using default symbols: {', '.join(symbols)}")
    
    # Get initial capital
    try:
        capital_input = console.input("Enter initial capital (default: $100,000): ")
        initial_capital = float(capital_input) if capital_input else 100000.0
    except ValueError:
        initial_capital = 100000.0
        console.print("Using default capital: $100,000")
    
    # Trading mode
    trading_mode = console.input("Trading mode - Paper (P) or Live (L)? [P]: ").upper()
    paper_trading = trading_mode != 'L'
    
    if not paper_trading:
        confirm = console.input("[WARNING] You selected LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            console.print("[SAFETY] Switching to paper trading for safety", style="bold yellow")
            paper_trading = True
    
    # Create configuration
    config = TradingConfig(
        trading_symbols=symbols,
        paper_trading=paper_trading,
        initial_capital=initial_capital,
        decision_interval=300,  # 5 minutes
        max_position_size=0.1,  # 10% max position
        stop_loss_percent=0.05,  # 5% stop loss
        take_profit_percent=0.15,  # 15% take profit
        max_daily_trades=50,
        max_drawdown_percent=0.2  # 20% max drawdown
    )
    
    return config


def main():
    """Main entry point"""
    # Setup
    setup_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print banner and info
    print_banner()
    print_warning()
    print_features()
    
    # Main menu
    while True:
        console.print("MAIN MENU", style="bold blue")
        console.print("1. Quick Demo (Paper Trading)")
        console.print("2. Interactive Setup & Start Trading")
        console.print("3. Launch CLI Interface")
        console.print("4. Exit")
        console.print()
        
        choice = console.input("Select option [1-4]: ").strip()
        
        if choice == '1':
            quick_start_demo()
            console.print("\nPress Enter to continue...")
            input()
            
        elif choice == '2':
            try:
                config = interactive_setup()
                
                console.print("\n[START] Starting Live Trading Engine...", style="bold green")
                engine = LiveTradingEngine(config)
                engine.start_trading()
                
                # Create CLI instance for interaction
                cli = LiveTradingCLI()
                cli.engine = engine
                cli.running = True
                
                console.print("\n[SUCCESS] Trading started! Available commands:")
                console.print("  portfolio - Show portfolio status")
                console.print("  stats - Show trading statistics")
                console.print("  risk - Show risk status")
                console.print("  monitor - Start live dashboard")
                console.print("  decide <SYMBOL> - Force decision")
                console.print("  quit - Stop trading\n")
                
                # Interactive command loop
                try:
                    while True:
                        command = input("💼 Command: ").strip().lower()
                        
                        if command == 'quit':
                            break
                        elif command == 'portfolio':
                            cli.show_portfolio()
                        elif command == 'stats':
                            cli.show_stats()
                        elif command == 'risk':
                            cli.show_risk_status()
                        elif command == 'monitor':
                            cli.live_monitor()
                        elif command.startswith('decide '):
                            symbol = command.split(' ')[1].upper()
                            cli.force_decision(symbol)
                        elif command == 'help':
                            console.print("Available commands: portfolio, stats, risk, monitor, decide <SYMBOL>, quit")
                        else:
                            console.print("Unknown command. Type 'help' for available commands.")
                            
                except KeyboardInterrupt:
                    pass
                finally:
                    engine.stop_trading()
                    console.print("\n[STOPPED] Trading stopped", style="bold yellow")
                    
            except Exception as e:
                console.print(f"[ERROR] Error: {e}", style="bold red")
                
        elif choice == '3':
            console.print("\n[CLI] Launching CLI Interface...", style="bold blue")
            console.print("Use: python -m tradingagents.live_trading.cli start --help")
            break
            
        elif choice == '4':
            console.print("Goodbye!", style="bold blue")
            break
            
        else:
            console.print("[ERROR] Invalid choice. Please select 1-4.", style="bold red")


if __name__ == '__main__':
    main()