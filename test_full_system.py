#!/usr/bin/env python3
"""
Full System Test - Test the complete AI + Mathematical analysis system
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
from tradingagents.graph.trading_graph import TradingAgentsGraph

console = Console()


class FullSystemTester:
    """Test the complete integrated trading system"""
    
    def __init__(self):
        self.analyzer = MathematicalAnalyzer()
        self.trading_graph = TradingAgentsGraph(debug=False)
        
    def load_test_data(self, symbol):
        """Load test data"""
        try:
            data_file = f"test_data/{symbol}_data.csv"
            if not os.path.exists(data_file):
                console.print(f"[ERROR] Test data not found: {data_file}", style="red")
                return pd.DataFrame()
            
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
            return df
        except Exception as e:
            console.print(f"[ERROR] Failed to load {symbol}: {e}", style="red")
            return pd.DataFrame()
    
    def test_ai_decision(self, symbol, date):
        """Test AI decision making"""
        try:
            final_state, processed_decision = self.trading_graph.propagate(symbol, date)
            
            # Extract action
            decision_upper = processed_decision.upper()
            if 'BUY' in decision_upper:
                action = 'BUY'
            elif 'SELL' in decision_upper:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Calculate confidence
            decision_text = final_state.get('final_trade_decision', '').lower()
            confidence_keywords = {
                'strongly': 0.9, 'confident': 0.8, 'likely': 0.7,
                'probably': 0.6, 'might': 0.4, 'uncertain': 0.3
            }
            
            ai_confidence = 0.5  # Default
            for keyword, score in confidence_keywords.items():
                if keyword in decision_text:
                    ai_confidence = score
                    break
            
            return {
                'action': action,
                'confidence': ai_confidence,
                'reasoning': final_state.get('final_trade_decision', 'No reasoning'),
                'success': True
            }
            
        except Exception as e:
            console.print(f"[WARNING] AI decision failed for {symbol}: {e}", style="yellow")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': f'AI failed: {str(e)}',
                'success': False
            }
    
    def run_integrated_analysis(self, symbol, test_date='2024-01-15'):
        """Run complete integrated analysis"""
        console.print(f"\n[ANALYSIS] Running full system analysis for {symbol}", style="bold blue")
        
        # Load data
        data = self.load_test_data(symbol)
        if data.empty:
            return None
        
        # Get recent data (last 100 days before test date)
        test_dt = pd.to_datetime(test_date)
        recent_data = data[data['Date'] <= test_dt].tail(100)
        
        if len(recent_data) < 50:
            console.print(f"[ERROR] Insufficient data for {symbol}", style="red")
            return None
        
        current_price = recent_data['Close'].iloc[-1]
        
        console.print(f"Current Price: ${current_price:.2f}")
        console.print(f"Analysis Date: {test_date}")
        console.print(f"Data Points: {len(recent_data)} days")
        
        # Mathematical Analysis
        console.print("[INFO] Running mathematical analysis...", style="yellow")
        
        try:
            # Technical indicators
            indicators = self.analyzer.calculate_technical_indicators(recent_data)
            
            # Risk metrics
            returns = recent_data['Close'].pct_change().dropna()
            risk_metrics = self.analyzer.calculate_risk_metrics(returns)
            
            # Quantitative signals
            signals = self.analyzer.generate_quantitative_signals(recent_data, indicators)
            
            # Opportunity scoring
            math_scores = self.analyzer.score_trading_opportunity(indicators, signals, risk_metrics)
            
            console.print("[OK] Mathematical analysis completed", style="green")
            
        except Exception as e:
            console.print(f"[ERROR] Mathematical analysis failed: {e}", style="red")
            return None
        
        # AI Analysis
        console.print("[INFO] Running AI analysis...", style="yellow")
        ai_result = self.test_ai_decision(symbol, test_date)
        
        if ai_result['success']:
            console.print("[OK] AI analysis completed", style="green")
        else:
            console.print("[WARNING] AI analysis failed, using fallback", style="yellow")
        
        # Integrated Decision
        console.print("[INFO] Integrating analysis results...", style="yellow")
        
        # Combine confidences
        math_confidence = math_scores['composite_score'] / 100.0
        ai_confidence = ai_result['confidence']
        probability_confidence = signals.probability_of_profit
        
        # Weighted combination (70% AI, 20% Math, 10% Probability)
        if ai_result['success']:
            combined_confidence = (ai_confidence * 0.7) + (math_confidence * 0.2) + (probability_confidence * 0.1)
        else:
            # Fall back to math-heavy weighting if AI fails
            combined_confidence = (math_confidence * 0.6) + (probability_confidence * 0.4)
        
        # Determine final action
        if ai_result['success'] and ai_result['action'] != 'HOLD':
            final_action = ai_result['action']
        else:
            # Use mathematical signals
            if signals.momentum_score > 0.2 and indicators.rsi < 70:
                final_action = 'BUY'
            elif signals.momentum_score < -0.2 or indicators.rsi > 75:
                final_action = 'SELL'
            else:
                final_action = 'HOLD'
        
        # Position sizing
        portfolio_value = 100000.0
        risk_per_trade = 0.02
        atr_stop_distance = indicators.atr * 2.0
        stop_loss_price = current_price - atr_stop_distance if final_action == 'BUY' else current_price + atr_stop_distance
        
        position_size = self.analyzer.calculate_position_size(
            portfolio_value, risk_per_trade, current_price, stop_loss_price
        )
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.analyzer.calculate_stop_loss_take_profit(
            current_price, indicators.atr, final_action
        )
        
        return {
            'symbol': symbol,
            'test_date': test_date,
            'current_price': current_price,
            'final_action': final_action,
            'combined_confidence': combined_confidence,
            'ai_confidence': ai_confidence,
            'math_confidence': math_confidence,
            'probability_confidence': probability_confidence,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'indicators': indicators,
            'signals': signals,
            'scores': math_scores,
            'ai_result': ai_result,
            'ai_success': ai_result['success']
        }
    
    def display_analysis_results(self, results):
        """Display comprehensive analysis results"""
        if not results:
            console.print("[ERROR] No results to display", style="red")
            return
        
        # Header
        console.print(f"\n{'='*60}")
        console.print(f"INTEGRATED ANALYSIS RESULTS - {results['symbol']}", style="bold blue")
        console.print(f"{'='*60}")
        
        # Overview
        overview_table = Table(title="Analysis Overview", show_header=True, header_style="bold blue")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="white")
        
        overview_table.add_row("Symbol", results['symbol'])
        overview_table.add_row("Analysis Date", results['test_date'])
        overview_table.add_row("Current Price", f"${results['current_price']:.2f}")
        overview_table.add_row("AI System Status", "✓ Working" if results['ai_success'] else "⚠ Fallback Mode")
        
        console.print(overview_table)
        
        # Confidence Analysis
        confidence_table = Table(title="Confidence Analysis", show_header=True, header_style="bold green")
        confidence_table.add_column("Component", style="cyan")
        confidence_table.add_column("Confidence", style="white")
        confidence_table.add_column("Weight", style="white")
        
        if results['ai_success']:
            confidence_table.add_row("AI Analysis", f"{results['ai_confidence']:.3f}", "70%")
        else:
            confidence_table.add_row("AI Analysis", "Failed", "0%")
        
        confidence_table.add_row("Mathematical Analysis", f"{results['math_confidence']:.3f}", "20%" if results['ai_success'] else "60%")
        confidence_table.add_row("Probability Model", f"{results['probability_confidence']:.3f}", "10%" if results['ai_success'] else "40%")
        confidence_table.add_row("Combined Confidence", f"{results['combined_confidence']:.3f}", "100%")
        
        console.print(confidence_table)
        
        # Technical Indicators
        indicators = results['indicators']
        tech_table = Table(title="Technical Indicators", show_header=True, header_style="bold yellow")
        tech_table.add_column("Indicator", style="cyan")
        tech_table.add_column("Value", style="white")
        tech_table.add_column("Signal", style="white")
        
        # RSI signal
        rsi_signal = "Overbought" if indicators.rsi > 70 else "Oversold" if indicators.rsi < 30 else "Neutral"
        rsi_color = "red" if indicators.rsi > 70 else "green" if indicators.rsi < 30 else "yellow"
        
        tech_table.add_row("RSI", f"{indicators.rsi:.2f}", f"[{rsi_color}]{rsi_signal}[/{rsi_color}]")
        
        # MACD signal
        macd_signal = "Bullish" if indicators.macd > indicators.macd_signal else "Bearish"
        macd_color = "green" if indicators.macd > indicators.macd_signal else "red"
        
        tech_table.add_row("MACD", f"{indicators.macd:.4f}", f"[{macd_color}]{macd_signal}[/{macd_color}]")
        tech_table.add_row("MACD Signal", f"{indicators.macd_signal:.4f}", "-")
        tech_table.add_row("ATR", f"{indicators.atr:.3f}", "Volatility Measure")
        tech_table.add_row("SMA 20", f"${indicators.sma_20:.2f}", "-")
        tech_table.add_row("SMA 50", f"${indicators.sma_50:.2f}", "-")
        
        console.print(tech_table)
        
        # Quantitative Signals
        signals = results['signals']
        signals_table = Table(title="Quantitative Signals", show_header=True, header_style="bold magenta")
        signals_table.add_column("Signal", style="cyan")
        signals_table.add_column("Value", style="white")
        signals_table.add_column("Interpretation", style="white")
        
        # Momentum interpretation
        momentum_interp = "Strong Bullish" if signals.momentum_score > 0.5 else "Bullish" if signals.momentum_score > 0.1 else "Bearish" if signals.momentum_score < -0.1 else "Neutral"
        momentum_color = "green" if signals.momentum_score > 0.1 else "red" if signals.momentum_score < -0.1 else "yellow"
        
        signals_table.add_row("Momentum Score", f"{signals.momentum_score:.3f}", f"[{momentum_color}]{momentum_interp}[/{momentum_color}]")
        signals_table.add_row("Mean Reversion", f"{signals.mean_reversion_score:.3f}", "-")
        signals_table.add_row("Trend Strength", f"{signals.trend_strength:.3f}", "Strong" if signals.trend_strength > 0.7 else "Moderate" if signals.trend_strength > 0.4 else "Weak")
        signals_table.add_row("Market Regime", signals.market_regime, "-")
        signals_table.add_row("Volatility Regime", signals.volatility_regime, "-")
        signals_table.add_row("Profit Probability", f"{signals.probability_of_profit:.3f}", f"{signals.probability_of_profit*100:.1f}%")
        
        console.print(signals_table)
        
        # Trading Decision
        action_color = "green" if results['final_action'] == 'BUY' else "red" if results['final_action'] == 'SELL' else "yellow"
        
        decision_table = Table(title="Trading Decision", show_header=True, header_style="bold blue")
        decision_table.add_column("Parameter", style="cyan")
        decision_table.add_column("Value", style="white")
        
        decision_table.add_row("Recommended Action", f"[{action_color}]{results['final_action']}[/{action_color}]")
        decision_table.add_row("Combined Confidence", f"{results['combined_confidence']:.3f}")
        decision_table.add_row("Position Size", f"{results['position_size']} shares")
        decision_table.add_row("Position Value", f"${results['position_size'] * results['current_price']:,.2f}")
        decision_table.add_row("Stop Loss", f"${results['stop_loss']:.2f}")
        decision_table.add_row("Take Profit", f"${results['take_profit']:.2f}")
        decision_table.add_row("Risk Amount", f"${abs(results['position_size'] * (results['current_price'] - results['stop_loss'])):,.2f}")
        
        console.print(decision_table)
        
        # AI Reasoning (if available)
        if results['ai_success']:
            reasoning = results['ai_result']['reasoning']
            console.print(f"\n[AI REASONING]", style="bold cyan")
            console.print(Panel(reasoning[:300] + "..." if len(reasoning) > 300 else reasoning, 
                              title="AI Analysis", border_style="cyan"))
        
        # Overall Assessment
        console.print(f"\n[ASSESSMENT]", style="bold blue")
        
        min_confidence = 0.6
        if results['combined_confidence'] >= min_confidence and results['final_action'] != 'HOLD':
            console.print(f"✅ EXECUTE TRADE: {results['final_action']} {results['position_size']} shares", style="bold green")
            console.print(f"   Confidence: {results['combined_confidence']:.3f} (above threshold {min_confidence})")
        else:
            console.print(f"⏸️ HOLD POSITION: Confidence {results['combined_confidence']:.3f} below threshold {min_confidence}", style="bold yellow")
        
        return results


def main():
    """Main test runner"""
    console.print("=" * 70)
    console.print("FULL INTEGRATED TRADING SYSTEM TEST", style="bold blue")
    console.print("=" * 70)
    
    console.print("\nThis test validates the complete integrated system:", style="yellow")
    console.print("• Mathematical Analysis (Technical + Risk + Quantitative)")
    console.print("• AI Decision Making (Multi-agent trading system)")
    console.print("• Integrated Confidence Scoring")
    console.print("• Position Sizing and Risk Management")
    console.print("• Complete Trading Decision Framework")
    
    tester = FullSystemTester()
    
    # Test symbols
    symbols = ['AAPL', 'NVDA', 'MSFT']
    test_dates = ['2024-01-15', '2024-03-15', '2024-05-15']
    
    all_results = []
    
    for symbol in symbols:
        for test_date in test_dates:
            console.print(f"\n{'='*70}")
            console.print(f"[TEST] {symbol} on {test_date}", style="bold yellow")
            console.print(f"{'='*70}")
            
            results = tester.run_integrated_analysis(symbol, test_date)
            
            if results:
                all_results.append(results)
                tester.display_analysis_results(results)
            else:
                console.print(f"[ERROR] Analysis failed for {symbol} on {test_date}", style="red")
    
    # Summary
    if all_results:
        console.print(f"\n{'='*70}")
        console.print("SYSTEM PERFORMANCE SUMMARY", style="bold blue")
        console.print(f"{'='*70}")
        
        summary_table = Table(title="Full System Test Results", show_header=True, header_style="bold blue")
        summary_table.add_column("Symbol", style="cyan")
        summary_table.add_column("Date", style="white")
        summary_table.add_column("Action", style="white")
        summary_table.add_column("Confidence", style="white")
        summary_table.add_column("AI Status", style="white")
        summary_table.add_column("Position Size", style="white")
        
        ai_success_count = 0
        executable_trades = 0
        avg_confidence = 0
        
        for result in all_results:
            action_color = "green" if result['final_action'] == 'BUY' else "red" if result['final_action'] == 'SELL' else "yellow"
            ai_status = "✓" if result['ai_success'] else "⚠"
            
            if result['ai_success']:
                ai_success_count += 1
            
            if result['combined_confidence'] >= 0.6 and result['final_action'] != 'HOLD':
                executable_trades += 1
            
            avg_confidence += result['combined_confidence']
            
            summary_table.add_row(
                result['symbol'],
                result['test_date'],
                f"[{action_color}]{result['final_action']}[/{action_color}]",
                f"{result['combined_confidence']:.3f}",
                ai_status,
                str(result['position_size'])
            )
        
        console.print(summary_table)
        
        # Overall statistics
        total_tests = len(all_results)
        ai_success_rate = (ai_success_count / total_tests) * 100
        trade_execution_rate = (executable_trades / total_tests) * 100
        avg_confidence = avg_confidence / total_tests
        
        stats_table = Table(title="System Statistics", show_header=True, header_style="bold green")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Total Tests", str(total_tests))
        stats_table.add_row("AI Success Rate", f"{ai_success_rate:.1f}%")
        stats_table.add_row("Trade Execution Rate", f"{trade_execution_rate:.1f}%")
        stats_table.add_row("Average Confidence", f"{avg_confidence:.3f}")
        stats_table.add_row("Executable Trades", str(executable_trades))
        
        console.print(stats_table)
        
        # Final assessment
        console.print(f"\n[FINAL ASSESSMENT]", style="bold blue")
        
        if ai_success_rate >= 80 and avg_confidence >= 0.6:
            console.print("🏆 EXCELLENT: Full system performing exceptionally well!", style="bold green")
        elif ai_success_rate >= 60 and avg_confidence >= 0.5:
            console.print("✅ GOOD: System performing well with good reliability", style="green")
        elif ai_success_rate >= 40:
            console.print("⚠️ MODERATE: System functional but AI reliability could improve", style="yellow")
        else:
            console.print("❌ POOR: System needs improvement, consider fallback modes", style="red")
    
    console.print(f"\n[COMPLETE] Full system test finished!", style="bold blue")
    return 0


if __name__ == '__main__':
    exit(main())