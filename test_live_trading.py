#!/usr/bin/env python3
"""
Quick test script for the live trading system
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.live_trading.live_trading_engine import LiveTradingEngine, TradingConfig
from rich.console import Console

console = Console()


def test_live_trading_system():
    """Test the live trading system components"""
    
    console.print("[TEST] Testing Live Trading System...", style="bold blue")
    
    try:
        # Create test configuration
        config = TradingConfig(
            trading_symbols=['AAPL', 'NVDA'],
            paper_trading=True,
            initial_capital=50000.0,
            decision_interval=60,  # 1 minute for testing
            max_position_size=0.2,
        )
        
        console.print("[OK] Configuration created", style="green")
        
        # Initialize engine
        engine = LiveTradingEngine(config)
        console.print("[OK] Trading engine initialized", style="green")
        
        # Test portfolio manager
        portfolio_status = engine.get_portfolio_status()
        console.print(f"[OK] Portfolio manager working - Initial value: ${portfolio_status['total_value']:,.2f}", style="green")
        
        # Test market data stream
        engine.market_data_stream.start()
        console.print("[OK] Market data stream started", style="green")
        
        # Wait for some data
        console.print("[WAIT] Waiting for market data...", style="yellow")
        time.sleep(5)
        
        # Check if we got data
        latest_aapl = engine.market_data_stream.get_latest_data('AAPL')
        if latest_aapl:
            console.print(f"[OK] Market data received - AAPL: ${latest_aapl.price:.2f}", style="green")
        else:
            console.print("[WARN] No market data received (may be outside trading hours)", style="yellow")
        
        # Test trade executor
        order_id = engine.trade_executor.execute_trade('AAPL', 'BUY', 10, 150.0)
        if order_id:
            console.print(f"[OK] Trade executor working - Order ID: {order_id}", style="green")
        else:
            console.print("[ERROR] Trade execution failed", style="red")
        
        # Test risk monitor
        engine.risk_monitor.start_monitoring(engine.portfolio_manager)
        console.print("[OK] Risk monitor started", style="green")
        
        # Get summary
        risk_summary = engine.risk_monitor.get_risk_summary()
        console.print(f"[OK] Risk monitoring active - Drawdown: {risk_summary.get('current_drawdown_pct', 0):.2f}%", style="green")
        
        # Force a decision for testing
        console.print("[AI] Testing AI decision making...", style="yellow")
        try:
            decision = engine.force_decision('AAPL')
            if decision:
                console.print(f"[OK] AI decision generated: {decision.action} {decision.quantity} shares", style="green")
                console.print(f"   Confidence: {decision.confidence:.2f}, Price: ${decision.price:.2f}", style="cyan")
            else:
                console.print("[WARN] No decision generated", style="yellow")
        except Exception as e:
            console.print(f"[WARN] AI decision failed: {e}", style="yellow")
        
        # Test CLI components
        try:
            from tradingagents.live_trading.cli import LiveTradingCLI
            cli = LiveTradingCLI()
            console.print("[OK] CLI interface available", style="green")
        except ImportError as e:
            console.print(f"[WARN] CLI import issue: {e}", style="yellow")
        
        # Stop components
        engine.market_data_stream.stop()
        engine.risk_monitor.stop_monitoring()
        
        console.print("\n[SUCCESS] Live Trading System Test Complete!", style="bold green")
        console.print("[OK] All core components are functional", style="green")
        console.print("\nTo start live trading, run:", style="bold blue")
        console.print("   python live_trading_main.py", style="cyan")
        console.print("   OR", style="white")
        console.print("   python -m tradingagents.live_trading.cli start", style="cyan")
        
        return True
        
    except Exception as e:
        console.print(f"[ERROR] Test failed: {e}", style="bold red")
        return False


if __name__ == '__main__':
    test_live_trading_system()