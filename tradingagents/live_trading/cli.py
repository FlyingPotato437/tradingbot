#!/usr/bin/env python3
"""
Live Trading CLI - Command line interface for real-time trading
"""

import asyncio
import time
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import click
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from .live_trading_engine import LiveTradingEngine, TradingConfig
from .portfolio_manager import PortfolioManager


console = Console()


class LiveTradingCLI:
    """Command line interface for live trading"""
    
    def __init__(self):
        self.engine: Optional[LiveTradingEngine] = None
        self.running = False
        
    def start_trading(self, symbols: List[str], paper_trading: bool = True, 
                     initial_capital: float = 100000.0):
        """Start the live trading engine"""
        try:
            # Create configuration
            config = TradingConfig(
                trading_symbols=symbols,
                paper_trading=paper_trading,
                initial_capital=initial_capital,
                decision_interval=300,  # 5 minutes
                max_position_size=0.1,  # 10% max position
                stop_loss_percent=0.05,  # 5% stop loss
                take_profit_percent=0.15  # 15% take profit
            )
            
            # Initialize engine
            self.engine = LiveTradingEngine(config)
            
            # Start trading
            self.engine.start_trading()
            self.running = True
            
            console.print(f"[SUCCESS] Live trading started with {len(symbols)} symbols", style="bold green")
            console.print(f"Initial capital: ${initial_capital:,.2f}")
            console.print(f"Paper trading: {'Yes' if paper_trading else 'No'}")
            console.print(f"Symbols: {', '.join(symbols)}")
            
            return True
            
        except Exception as e:
            console.print(f"[ERROR] Error starting trading: {e}", style="bold red")
            return False
    
    def stop_trading(self):
        """Stop the live trading engine"""
        if self.engine and self.running:
            self.engine.stop_trading()
            self.running = False
            console.print("[STOPPED] Trading stopped", style="bold yellow")
        else:
            console.print("[ERROR] No active trading session", style="bold red")
    
    def show_portfolio(self):
        """Display current portfolio status"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        portfolio_status = self.engine.get_portfolio_status()
        
        # Create portfolio table
        table = Table(title="Portfolio Status", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Value", f"${portfolio_status['total_value']:,.2f}")
        table.add_row("Cash", f"${portfolio_status['cash']:,.2f}")
        table.add_row("Positions Value", f"${portfolio_status['positions_value']:,.2f}")
        table.add_row("Total P&L", f"${portfolio_status['total_pnl']:,.2f}")
        table.add_row("Total Return", f"{portfolio_status['total_return_pct']:.2f}%")
        table.add_row("Max Drawdown", f"{portfolio_status['max_drawdown_pct']:.2f}%")
        table.add_row("Total Trades", str(portfolio_status['total_trades']))
        
        console.print(table)
        
        # Show positions if any
        if portfolio_status['positions']:
            positions_table = Table(title="Current Positions", show_header=True, header_style="bold blue")
            positions_table.add_column("Symbol", style="cyan")
            positions_table.add_column("Quantity", style="white")
            positions_table.add_column("Avg Cost", style="white")
            positions_table.add_column("Current Price", style="white")
            positions_table.add_column("Market Value", style="white")
            positions_table.add_column("P&L", style="white")
            positions_table.add_column("P&L %", style="white")
            
            for symbol, pos in portfolio_status['positions'].items():
                pnl_color = "green" if pos['unrealized_pnl'] >= 0 else "red"
                positions_table.add_row(
                    symbol,
                    str(pos['quantity']),
                    f"${pos['avg_cost']:.2f}",
                    f"${pos['current_price']:.2f}",
                    f"${pos['market_value']:,.2f}",
                    f"[{pnl_color}]${pos['unrealized_pnl']:,.2f}[/{pnl_color}]",
                    f"[{pnl_color}]{pos['unrealized_pnl_pct']:.2f}%[/{pnl_color}]"
                )
            
            console.print(positions_table)
    
    def show_stats(self):
        """Display trading statistics"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        stats = self.engine.get_trading_stats()
        
        table = Table(title="Trading Statistics", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Decisions", str(stats['total_decisions']))
        table.add_row("Daily Trades", str(stats['daily_trades']))
        table.add_row("Is Running", "✅ Yes" if stats['is_running'] else "❌ No")
        table.add_row("Portfolio Value", f"${stats['portfolio_value']:,.2f}")
        table.add_row("Available Capital", f"${stats['available_capital']:,.2f}")
        
        console.print(table)
    
    def force_decision(self, symbol: str):
        """Force a trading decision for a symbol"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        console.print(f"[ANALYZING] {symbol}...", style="bold yellow")
        
        try:
            decision = self.engine.force_decision(symbol)
            if decision:
                console.print(f"Decision for {symbol}:", style="bold blue")
                console.print(f"   Action: {decision.action}")
                console.print(f"   Quantity: {decision.quantity}")
                console.print(f"   Price: ${decision.price:.2f}")
                console.print(f"   Confidence: {decision.confidence:.2f}")
                console.print(f"   Reasoning: {decision.reasoning[:100]}...")
            else:
                console.print(f"[ERROR] No decision generated for {symbol}", style="bold red")
        except Exception as e:
            console.print(f"[ERROR] Error generating decision: {e}", style="bold red")
    
    def add_symbol(self, symbol: str):
        """Add a symbol to track"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        self.engine.add_symbol(symbol.upper())
        console.print(f"[ADDED] {symbol.upper()} to tracking list", style="bold green")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        self.engine.remove_symbol(symbol.upper())
        console.print(f"[REMOVED] {symbol.upper()} from tracking list", style="bold yellow")
    
    def show_risk_status(self):
        """Display risk monitoring status"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        risk_summary = self.engine.risk_monitor.get_risk_summary()
        
        if not risk_summary:
            console.print("[ERROR] No risk data available", style="bold red")
            return
        
        table = Table(title="Risk Monitor Status", show_header=True, header_style="bold red")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Risk status
        emergency_color = "red" if risk_summary.get('emergency_stop_triggered') else "green"
        table.add_row("Emergency Stop", 
                     f"[{emergency_color}]{'TRIGGERED' if risk_summary.get('emergency_stop_triggered') else 'OK'}[/{emergency_color}]")
        
        trading_color = "red" if risk_summary.get('trading_halted') else "green"
        table.add_row("Trading Status", 
                     f"[{trading_color}]{'HALTED' if risk_summary.get('trading_halted') else 'ACTIVE'}[/{trading_color}]")
        
        table.add_row("Current Drawdown", f"{risk_summary.get('current_drawdown_pct', 0):.2f}%")
        table.add_row("Max Drawdown Limit", f"{risk_summary.get('max_drawdown_limit_pct', 0):.2f}%")
        table.add_row("Active Alerts", str(risk_summary.get('active_alerts_count', 0)))
        table.add_row("Alerts (24h)", str(risk_summary.get('total_alerts_24h', 0)))
        
        console.print(table)
        
        # Show active alerts if any
        active_alerts = self.engine.risk_monitor.get_active_alerts()
        if active_alerts:
            alerts_table = Table(title="Active Risk Alerts", show_header=True, header_style="bold red")
            alerts_table.add_column("Risk Type", style="cyan")
            alerts_table.add_column("Symbol", style="white")
            alerts_table.add_column("Level", style="white")
            alerts_table.add_column("Message", style="white")
            alerts_table.add_column("Time", style="white")
            
            for alert in active_alerts:
                level_color = {
                    "low": "blue",
                    "medium": "yellow", 
                    "high": "orange",
                    "critical": "red"
                }.get(alert.risk_level.value, "white")
                
                alerts_table.add_row(
                    alert.risk_type,
                    alert.symbol,
                    f"[{level_color}]{alert.risk_level.value.upper()}[/{level_color}]",
                    alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
                    alert.timestamp.strftime("%H:%M:%S")
                )
            
            console.print(alerts_table)
    
    def live_monitor(self):
        """Start live monitoring dashboard"""
        if not self.engine:
            console.print("[ERROR] No active trading session", style="bold red")
            return
        
        console.print("[DASHBOARD] Starting live dashboard... Press Ctrl+C to exit", style="bold blue")
        
        def generate_dashboard():
            """Generate dashboard layout"""
            layout = Layout()
            
            # Create sections
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=5)
            )
            
            layout["main"].split_row(
                Layout(name="portfolio"),
                Layout(name="positions")
            )
            
            # Header
            header_text = Text("LIVE TRADING DASHBOARD", style="bold white on blue")
            header_text.append(f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            layout["header"].update(Panel(header_text, style="blue"))
            
            # Portfolio summary
            portfolio_status = self.engine.get_portfolio_status()
            portfolio_table = Table(title="Portfolio", show_header=False)
            portfolio_table.add_column("Metric", style="cyan")
            portfolio_table.add_column("Value", style="white")
            
            portfolio_table.add_row("Total Value", f"${portfolio_status['total_value']:,.2f}")
            portfolio_table.add_row("Cash", f"${portfolio_status['cash']:,.2f}")
            portfolio_table.add_row("P&L", f"${portfolio_status['total_pnl']:,.2f}")
            portfolio_table.add_row("Return", f"{portfolio_status['total_return_pct']:.2f}%")
            
            layout["portfolio"].update(Panel(portfolio_table, title="Portfolio", border_style="green"))
            
            # Positions
            positions_table = Table(title="Positions", show_header=True, header_style="bold blue")
            positions_table.add_column("Symbol", style="cyan")
            positions_table.add_column("Qty", style="white")
            positions_table.add_column("Price", style="white")
            positions_table.add_column("P&L %", style="white")
            
            for symbol, pos in portfolio_status.get('positions', {}).items():
                pnl_color = "green" if pos['unrealized_pnl'] >= 0 else "red"
                positions_table.add_row(
                    symbol,
                    str(pos['quantity']),
                    f"${pos['current_price']:.2f}",
                    f"[{pnl_color}]{pos['unrealized_pnl_pct']:.1f}%[/{pnl_color}]"
                )
            
            layout["positions"].update(Panel(positions_table, title="Positions", border_style="blue"))
            
            # Footer with stats
            stats = self.engine.get_trading_stats()
            footer_text = f"Decisions: {stats['total_decisions']} | Daily Trades: {stats['daily_trades']} | Status: {'ACTIVE' if stats['is_running'] else 'STOPPED'}"
            layout["footer"].update(Panel(footer_text, title="Status", border_style="yellow"))
            
            return layout
        
        try:
            with Live(generate_dashboard(), refresh_per_second=1) as live:
                while True:
                    time.sleep(1)
                    live.update(generate_dashboard())
        except KeyboardInterrupt:
            console.print("\n[STOPPED] Live monitoring stopped", style="bold yellow")


# Click CLI commands
@click.group()
def cli():
    """Live Trading Agent - Real-time autonomous trading system"""
    pass


@cli.command()
@click.option('--symbols', '-s', multiple=True, default=['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL'],
              help='Trading symbols to track')
@click.option('--capital', '-c', default=100000.0, help='Initial capital')
@click.option('--live', is_flag=True, help='Enable live trading (default: paper trading)')
def start(symbols, capital, live):
    """Start live trading"""
    cli_instance = LiveTradingCLI()
    
    if cli_instance.start_trading(list(symbols), paper_trading=not live, initial_capital=capital):
        try:
            # Keep running until interrupted
            console.print("\n📊 Trading is running... Commands available:")
            console.print("  - portfolio: Show portfolio status")
            console.print("  - stats: Show trading statistics")
            console.print("  - risk: Show risk status")
            console.print("  - monitor: Start live dashboard")
            console.print("  - Press Ctrl+C to stop\n")
            
            while True:
                command = input("Enter command (or 'quit' to exit): ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'portfolio':
                    cli_instance.show_portfolio()
                elif command == 'stats':
                    cli_instance.show_stats()
                elif command == 'risk':
                    cli_instance.show_risk_status()
                elif command == 'monitor':
                    cli_instance.live_monitor()
                elif command.startswith('decide '):
                    symbol = command.split(' ')[1].upper()
                    cli_instance.force_decision(symbol)
                elif command.startswith('add '):
                    symbol = command.split(' ')[1].upper()
                    cli_instance.add_symbol(symbol)
                elif command.startswith('remove '):
                    symbol = command.split(' ')[1].upper()
                    cli_instance.remove_symbol(symbol)
                elif command == 'help':
                    console.print("Available commands:")
                    console.print("  portfolio - Show portfolio status")
                    console.print("  stats - Show trading statistics")
                    console.print("  risk - Show risk status")
                    console.print("  monitor - Start live dashboard")
                    console.print("  decide <SYMBOL> - Force decision for symbol")
                    console.print("  add <SYMBOL> - Add symbol to track")
                    console.print("  remove <SYMBOL> - Remove symbol")
                    console.print("  quit - Exit")
                else:
                    console.print("Unknown command. Type 'help' for available commands.")
                    
        except KeyboardInterrupt:
            pass
        finally:
            cli_instance.stop_trading()


@cli.command()
def portfolio():
    """Show portfolio status"""
    cli_instance = LiveTradingCLI()
    cli_instance.show_portfolio()


@cli.command()
def stats():
    """Show trading statistics"""
    cli_instance = LiveTradingCLI()
    cli_instance.show_stats()


@cli.command()
@click.argument('symbol')
def decide(symbol):
    """Force a trading decision for a symbol"""
    cli_instance = LiveTradingCLI()
    cli_instance.force_decision(symbol.upper())


@cli.command()
def monitor():
    """Start live monitoring dashboard"""
    cli_instance = LiveTradingCLI()
    cli_instance.live_monitor()


if __name__ == '__main__':
    cli()