import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field

from ..graph.trading_graph import TradingAgentsGraph
from .portfolio_manager import PortfolioManager
from .trade_executor import TradeExecutor
from .alpaca_executor import AlpacaExecutor
from .market_data_stream import LiveMarketDataStream
from .risk_monitor import RiskMonitor
from .mathematical_analyzer import MathematicalAnalyzer


@dataclass
class TradingDecision:
    """Represents a trading decision made by the system"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: int
    confidence: float
    price: float
    timestamp: datetime
    reasoning: str


@dataclass
class TradingConfig:
    """Configuration for live trading"""
    trading_symbols: List[str] = field(default_factory=lambda: ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL'])
    decision_interval: int = 300  # seconds between decisions
    max_position_size: float = 0.1  # max 10% of portfolio per position
    stop_loss_percent: float = 0.05  # 5% stop loss
    take_profit_percent: float = 0.15  # 15% take profit
    max_daily_trades: int = 50
    max_drawdown_percent: float = 0.2  # 20% max drawdown
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    paper_trading: bool = True  # Start with paper trading
    initial_capital: float = 100000.0
    use_alpaca: bool = False  # Use Alpaca for live trading
    min_confidence_threshold: float = 0.6  # Minimum confidence to execute trades
    use_mathematical_analysis: bool = True  # Enable mathematical analysis
    risk_per_trade: float = 0.02  # Risk 2% per trade for position sizing
    lookback_days: int = 252  # Days of historical data for analysis


class LiveTradingEngine:
    """Real-time trading engine that makes autonomous trading decisions"""
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.is_running = False
        self.decision_thread = None
        
        # Initialize components
        self.trading_graph = TradingAgentsGraph(debug=False)
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.config.initial_capital,
            max_position_size=self.config.max_position_size
        )
        
        # Choose trade executor based on configuration
        if self.config.use_alpaca and not self.config.paper_trading:
            self.trade_executor = AlpacaExecutor(paper_trading=False)
        elif self.config.use_alpaca and self.config.paper_trading:
            self.trade_executor = AlpacaExecutor(paper_trading=True)
        else:
            self.trade_executor = TradeExecutor(paper_trading=self.config.paper_trading)
        
        self.market_data_stream = LiveMarketDataStream(self.config.trading_symbols)
        self.risk_monitor = RiskMonitor(
            max_drawdown=self.config.max_drawdown_percent,
            stop_loss=self.config.stop_loss_percent
        )
        
        # Mathematical analyzer
        self.mathematical_analyzer = MathematicalAnalyzer() if self.config.use_mathematical_analysis else None
        
        # Tracking
        self.daily_trades = 0
        self.last_decision_time = {}
        self.active_positions = {}
        self.decision_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_trading(self):
        """Start the live trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return
            
        self.logger.info("Starting live trading engine...")
        self.is_running = True
        
        # Start market data stream
        self.market_data_stream.start()
        
        # Start decision making thread
        self.decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self.decision_thread.start()
        
        # Start risk monitoring
        self.risk_monitor.start_monitoring(self.portfolio_manager)
        
        self.logger.info("Live trading engine started successfully")

    def stop_trading(self):
        """Stop the live trading engine"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping live trading engine...")
        self.is_running = False
        
        # Stop components
        self.market_data_stream.stop()
        self.risk_monitor.stop_monitoring()
        
        # Wait for threads to finish
        if self.decision_thread:
            self.decision_thread.join(timeout=5)
            
        self.logger.info("Live trading engine stopped")

    def _decision_loop(self):
        """Main decision making loop"""
        while self.is_running:
            try:
                if self._is_trading_hours() and self.daily_trades < self.config.max_daily_trades:
                    self._make_trading_decisions()
                    
                time.sleep(self.config.decision_interval)
                
            except Exception as e:
                self.logger.error(f"Error in decision loop: {e}")
                time.sleep(60)  # Wait before retrying

    def _make_trading_decisions(self):
        """Make trading decisions for all symbols"""
        current_time = datetime.now()
        
        for symbol in self.config.trading_symbols:
            try:
                # Check if enough time has passed since last decision
                if symbol in self.last_decision_time:
                    time_diff = current_time - self.last_decision_time[symbol]
                    if time_diff.total_seconds() < self.config.decision_interval:
                        continue
                
                # Get latest market data
                market_data = self.market_data_stream.get_latest_data(symbol)
                if not market_data:
                    continue
                
                # Make decision using trading graph
                decision = self._analyze_and_decide(symbol, current_time)
                
                if decision and decision.action != 'HOLD':
                    # Execute trade if decision is not HOLD
                    success = self._execute_decision(decision)
                    if success:
                        self.daily_trades += 1
                        self.decision_history.append(decision)
                
                self.last_decision_time[symbol] = current_time
                
            except Exception as e:
                self.logger.error(f"Error making decision for {symbol}: {e}")

    def _analyze_and_decide(self, symbol: str, current_time: datetime) -> Optional[TradingDecision]:
        """Analyze market conditions and make a trading decision"""
        try:
            # Get current price first
            current_price = self.market_data_stream.get_current_price(symbol)
            if not current_price:
                return None
            
            # Get historical data for mathematical analysis
            price_data = None
            technical_indicators = None
            quantitative_signals = None
            math_score = None
            
            if self.mathematical_analyzer:
                try:
                    # Get historical data from yfinance if market data stream doesn't have enough
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(period="1y", interval="1d")
                    
                    if len(hist_data) > 20:
                        # Use the historical data
                        price_data = hist_data.reset_index()
                        price_data = price_data.rename(columns={'Date': 'Date'})
                        
                        # Calculate technical indicators
                        technical_indicators = self.mathematical_analyzer.calculate_technical_indicators(price_data)
                        
                        # Generate quantitative signals
                        quantitative_signals = self.mathematical_analyzer.generate_quantitative_signals(
                            price_data, technical_indicators
                        )
                        
                        # Score the opportunity
                        returns = price_data['Close'].pct_change().dropna()
                        risk_metrics = self.mathematical_analyzer.calculate_risk_metrics(returns)
                        
                        math_score = self.mathematical_analyzer.score_trading_opportunity(
                            technical_indicators, quantitative_signals, risk_metrics
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Mathematical analysis failed for {symbol}: {e}")
            
            # Use the existing trading graph to get AI decision
            final_state, processed_decision = self.trading_graph.propagate(
                symbol, 
                current_time.strftime("%Y-%m-%d")
            )
            
            # Extract decision details
            action = self._extract_action(processed_decision)
            ai_confidence = self._calculate_confidence(final_state)
            
            # Combine AI confidence with mathematical analysis
            if math_score and quantitative_signals:
                # Weight the confidence based on mathematical analysis
                math_confidence = math_score['composite_score'] / 100.0
                probability_weight = quantitative_signals.probability_of_profit
                
                # Combined confidence (70% AI, 20% math analysis, 10% probability)
                confidence = (ai_confidence * 0.7) + (math_confidence * 0.2) + (probability_weight * 0.1)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            else:
                confidence = ai_confidence
            
            # Check confidence threshold
            if confidence < self.config.min_confidence_threshold:
                self.logger.info(f"Decision confidence {confidence:.2f} below threshold {self.config.min_confidence_threshold} for {symbol}")
                return None
            
            # Calculate position size
            if self.mathematical_analyzer and technical_indicators:
                # Use mathematical position sizing with ATR-based stop loss
                atr_stop_distance = technical_indicators.atr * 2.0  # 2x ATR for stop loss
                stop_loss_price = (current_price - atr_stop_distance) if action == 'BUY' else (current_price + atr_stop_distance)
                
                quantity = self.mathematical_analyzer.calculate_position_size(
                    self.portfolio_manager.get_total_value(),
                    self.config.risk_per_trade,
                    current_price,
                    stop_loss_price
                )
            else:
                # Use basic position sizing
                quantity = self._calculate_position_size(symbol, current_price, confidence)
            
            # Create comprehensive reasoning
            reasoning_parts = [final_state.get('final_trade_decision', 'AI analysis')]
            if math_score:
                reasoning_parts.append(f"Math score: {math_score['composite_score']:.1f}/100")
            if quantitative_signals:
                reasoning_parts.append(f"Momentum: {quantitative_signals.momentum_score:.2f}")
                reasoning_parts.append(f"Trend: {quantitative_signals.trend_strength:.2f}")
                reasoning_parts.append(f"Regime: {quantitative_signals.market_regime}")
            if technical_indicators:
                reasoning_parts.append(f"RSI: {technical_indicators.rsi:.1f}")
            
            decision = TradingDecision(
                symbol=symbol,
                action=action,
                quantity=quantity,
                confidence=confidence,
                price=current_price,
                timestamp=current_time,
                reasoning=" | ".join(reasoning_parts)
            )
            
            self.logger.info(f"Decision for {symbol}: {action} {quantity} shares at ${current_price} (confidence: {confidence:.2f})")
            return decision
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _extract_action(self, processed_decision: str) -> str:
        """Extract action from processed decision"""
        decision_upper = processed_decision.upper()
        if 'BUY' in decision_upper:
            return 'BUY'
        elif 'SELL' in decision_upper:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_confidence(self, final_state: Dict) -> float:
        """Calculate confidence score based on final state"""
        # This is a simplified confidence calculation
        # In practice, you might want to analyze the consensus among agents
        try:
            decision_text = final_state.get('final_trade_decision', '').lower()
            
            # Look for confidence indicators in the decision
            confidence_keywords = {
                'strongly': 0.9,
                'confident': 0.8,
                'likely': 0.7,
                'probably': 0.6,
                'might': 0.4,
                'uncertain': 0.3
            }
            
            for keyword, score in confidence_keywords.items():
                if keyword in decision_text:
                    return score
            
            return 0.5  # Default confidence
            
        except:
            return 0.5

    def _calculate_position_size(self, symbol: str, price: float, confidence: float) -> int:
        """Calculate optimal position size"""
        try:
            # Get available capital
            available_capital = self.portfolio_manager.get_available_capital()
            
            # Calculate maximum position value
            max_position_value = available_capital * self.config.max_position_size
            
            # Adjust based on confidence
            adjusted_position_value = max_position_value * confidence
            
            # Calculate quantity
            quantity = int(adjusted_position_value / price)
            
            # Ensure minimum viable quantity
            return max(1, quantity) if quantity > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0

    def _execute_decision(self, decision: TradingDecision) -> bool:
        """Execute a trading decision"""
        try:
            # Check risk limits
            if not self.risk_monitor.check_trade_risk(decision, self.portfolio_manager):
                self.logger.warning(f"Trade blocked by risk monitor: {decision.symbol} {decision.action}")
                return False
            
            # Execute the trade
            order_id = self.trade_executor.execute_trade(
                symbol=decision.symbol,
                action=decision.action,
                quantity=decision.quantity,
                price=decision.price
            )
            
            if order_id:
                # Update portfolio
                self.portfolio_manager.update_position(
                    symbol=decision.symbol,
                    action=decision.action,
                    quantity=decision.quantity,
                    price=decision.price
                )
                
                self.logger.info(f"Successfully executed: {decision.action} {decision.quantity} {decision.symbol} at ${decision.price}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing decision: {e}")
            return False

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now()
        start_time = datetime.strptime(self.config.trading_hours_start, "%H:%M").time()
        end_time = datetime.strptime(self.config.trading_hours_end, "%H:%M").time()
        
        current_time = now.time()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        return start_time <= current_time <= end_time

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return self.portfolio_manager.get_portfolio_summary()

    def get_trading_stats(self) -> Dict:
        """Get trading statistics"""
        return {
            'total_decisions': len(self.decision_history),
            'daily_trades': self.daily_trades,
            'is_running': self.is_running,
            'last_decision_times': self.last_decision_time,
            'portfolio_value': self.portfolio_manager.get_total_value(),
            'available_capital': self.portfolio_manager.get_available_capital()
        }

    def force_decision(self, symbol: str) -> Optional[TradingDecision]:
        """Force a trading decision for a specific symbol (for testing)"""
        if symbol not in self.config.trading_symbols:
            self.config.trading_symbols.append(symbol)
            
        return self._analyze_and_decide(symbol, datetime.now())

    def add_symbol(self, symbol: str):
        """Add a new symbol to track"""
        if symbol not in self.config.trading_symbols:
            self.config.trading_symbols.append(symbol)
            self.market_data_stream.add_symbol(symbol)

    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking"""
        if symbol in self.config.trading_symbols:
            self.config.trading_symbols.remove(symbol)
            self.market_data_stream.remove_symbol(symbol)