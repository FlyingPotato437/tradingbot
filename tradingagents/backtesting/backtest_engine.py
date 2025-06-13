import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.live_trading.mathematical_analyzer import MathematicalAnalyzer
from tradingagents.dataflows.finnhub_utils import get_data_in_range
# from tradingagents.dataflows.yfin_utils import YFinUtils  # Not needed
from tradingagents.dataflows.config import DATA_DIR


@dataclass
class BacktestTrade:
    """Represents a completed backtest trade"""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    action: str  # 'BUY' or 'SELL'
    pnl: float
    pnl_pct: float
    confidence: float
    reasoning: str
    duration_days: int


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)


class BacktestEngine:
    """Comprehensive backtesting engine using historical data"""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 1.0):
        self.initial_capital = initial_capital
        self.commission = commission
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trading_graph = TradingAgentsGraph(debug=False)
        self.mathematical_analyzer = MathematicalAnalyzer()
        
        # Backtest state
        self.cash = initial_capital
        self.positions = {}  # symbol -> {quantity, avg_price, entry_date}
        self.portfolio_history = []
        self.trades = []
        
        # Load environment variables
        self._load_env()
    
    def _load_env(self):
        """Load environment variables from .env file"""
        env_file = os.path.join(os.path.dirname(__file__), '../../.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data using available data sources"""
        try:
            # Try to use YFin data first
            data_file = os.path.join(DATA_DIR, f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv")
            
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter by date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()
                filtered_df = filtered_df.sort_values('Date').reset_index(drop=True)
                
                self.logger.info(f"Loaded {len(filtered_df)} days of data for {symbol}")
                return filtered_df
            else:
                self.logger.warning(f"No historical data file found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str,
                    decision_frequency: int = 5) -> BacktestResults:
        """Run comprehensive backtest"""
        
        self.logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.trades = []
        
        # Load historical data for all symbols
        symbol_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, start_date, end_date)
            if not data.empty:
                symbol_data[symbol] = data
            else:
                self.logger.warning(f"No data available for {symbol}, skipping")
        
        if not symbol_data:
            self.logger.error("No historical data available for any symbols")
            return BacktestResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Get all unique dates and sort them
        all_dates = set()
        for data in symbol_data.values():
            all_dates.update(data['Date'].dt.strftime('%Y-%m-%d'))
        
        trading_dates = sorted(list(all_dates))
        self.logger.info(f"Found {len(trading_dates)} trading dates")
        
        # Run simulation day by day
        for i, current_date in enumerate(trading_dates):
            if i % decision_frequency == 0:  # Make decisions every N days
                self._process_trading_day(current_date, symbol_data)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_date, symbol_data)
            self.portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
        
        # Calculate final results
        results = self._calculate_results()
        
        self.logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
        self.logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
        self.logger.info(f"Max drawdown: {results.max_drawdown:.2%}")
        self.logger.info(f"Win rate: {results.win_rate:.2%}")
        
        return results
    
    def _process_trading_day(self, current_date: str, symbol_data: Dict[str, pd.DataFrame]):
        """Process trading decisions for a specific day"""
        
        for symbol in symbol_data.keys():
            try:
                # Get current price
                symbol_df = symbol_data[symbol]
                current_row = symbol_df[symbol_df['Date'].dt.strftime('%Y-%m-%d') == current_date]
                
                if current_row.empty:
                    continue
                
                current_price = current_row.iloc[0]['Close']
                
                # Get historical data up to current date for analysis
                historical_data = symbol_df[symbol_df['Date'].dt.strftime('%Y-%m-%d') <= current_date].copy()
                
                if len(historical_data) < 20:  # Need minimum data for analysis
                    continue
                
                # Run mathematical analysis
                technical_indicators = self.mathematical_analyzer.calculate_technical_indicators(historical_data)
                returns = historical_data['Close'].pct_change().dropna()
                risk_metrics = self.mathematical_analyzer.calculate_risk_metrics(returns)
                quantitative_signals = self.mathematical_analyzer.generate_quantitative_signals(
                    historical_data, technical_indicators
                )
                
                # Score the opportunity
                math_score = self.mathematical_analyzer.score_trading_opportunity(
                    technical_indicators, quantitative_signals, risk_metrics
                )
                
                # Get AI decision using trading graph
                try:
                    final_state, processed_decision = self.trading_graph.propagate(symbol, current_date)
                    ai_decision = self._extract_action(processed_decision)
                    ai_confidence = self._calculate_confidence(final_state)
                except Exception as e:
                    self.logger.warning(f"AI decision failed for {symbol} on {current_date}: {e}")
                    ai_decision = 'HOLD'
                    ai_confidence = 0.5
                
                # Combine AI and mathematical analysis
                math_confidence = math_score['composite_score'] / 100.0
                combined_confidence = (ai_confidence * 0.7) + (math_confidence * 0.3)
                
                # Make trading decision
                if combined_confidence > 0.65 and ai_decision in ['BUY', 'SELL']:
                    self._execute_backtest_trade(
                        symbol, current_date, current_price, ai_decision, 
                        combined_confidence, final_state.get('final_trade_decision', ''),
                        technical_indicators
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol} on {current_date}: {e}")
    
    def _execute_backtest_trade(self, symbol: str, date: str, price: float, 
                               action: str, confidence: float, reasoning: str,
                               technical_indicators):
        """Execute a backtest trade"""
        
        # Calculate position size based on risk management
        portfolio_value = self._calculate_portfolio_value(date, {})
        risk_per_trade = 0.02  # 2% risk per trade
        
        # Use ATR-based stop loss for position sizing
        stop_distance = technical_indicators.atr * 2.0
        stop_price = price - stop_distance if action == 'BUY' else price + stop_distance
        
        max_position_size = self.mathematical_analyzer.calculate_position_size(
            portfolio_value, risk_per_trade, price, stop_price
        )
        
        # Limit to 10% of portfolio per position
        max_value = portfolio_value * 0.1
        max_shares_by_value = int(max_value / price)
        quantity = min(max_position_size, max_shares_by_value)
        
        if quantity <= 0:
            return
        
        if action == 'BUY':
            cost = quantity * price + self.commission
            
            if cost <= self.cash:
                self.cash -= cost
                
                if symbol in self.positions:
                    # Add to existing position
                    existing = self.positions[symbol]
                    total_quantity = existing['quantity'] + quantity
                    total_cost = (existing['quantity'] * existing['avg_price']) + (quantity * price)
                    avg_price = total_cost / total_quantity
                    
                    self.positions[symbol] = {
                        'quantity': total_quantity,
                        'avg_price': avg_price,
                        'entry_date': existing['entry_date']
                    }
                else:
                    # New position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'entry_date': date
                    }
                
                self.logger.debug(f"BUY {quantity} {symbol} at ${price:.2f} on {date}")
        
        elif action == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            sell_quantity = min(quantity, position['quantity'])
            
            if sell_quantity > 0:
                proceeds = (sell_quantity * price) - self.commission
                self.cash += proceeds
                
                # Calculate P&L for this trade
                cost_basis = sell_quantity * position['avg_price']
                pnl = proceeds - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                
                # Record the trade
                entry_date_obj = datetime.strptime(position['entry_date'], '%Y-%m-%d')
                exit_date_obj = datetime.strptime(date, '%Y-%m-%d')
                duration = (exit_date_obj - entry_date_obj).days
                
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_date=position['entry_date'],
                    exit_date=date,
                    entry_price=position['avg_price'],
                    exit_price=price,
                    quantity=sell_quantity,
                    action='BUY',  # Original action was BUY
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    confidence=confidence,
                    reasoning=reasoning,
                    duration_days=duration
                )
                self.trades.append(trade)
                
                # Update position
                if sell_quantity == position['quantity']:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['quantity'] -= sell_quantity
                
                self.logger.debug(f"SELL {sell_quantity} {symbol} at ${price:.2f} on {date} (P&L: ${pnl:.2f})")
    
    def _calculate_portfolio_value(self, date: str, symbol_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value on a given date"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in symbol_data:
                symbol_df = symbol_data[symbol]
                current_row = symbol_df[symbol_df['Date'].dt.strftime('%Y-%m-%d') == date]
                
                if not current_row.empty:
                    current_price = current_row.iloc[0]['Close']
                    position_value = position['quantity'] * current_price
                    total_value += position_value
                else:
                    # Use last known price if current date not available
                    position_value = position['quantity'] * position['avg_price']
                    total_value += position_value
        
        return total_value
    
    def _extract_action(self, processed_decision: str) -> str:
        """Extract action from AI decision"""
        decision_upper = processed_decision.upper()
        if 'BUY' in decision_upper:
            return 'BUY'
        elif 'SELL' in decision_upper:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_confidence(self, final_state: Dict) -> float:
        """Calculate confidence from AI decision"""
        decision_text = final_state.get('final_trade_decision', '').lower()
        
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
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not self.portfolio_history:
            return BacktestResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate returns
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        daily_returns = []
        
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = len(portfolio_values)
        years = days / 252.0  # Trading days per year
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        if daily_returns and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            daily_returns=daily_returns,
            portfolio_values=portfolio_values,
            dates=[h['date'] for h in self.portfolio_history]
        )
    
    def save_results(self, results: BacktestResults, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        # Convert results to dictionary for JSON serialization
        results_dict = {
            'summary': {
                'total_return': results.total_return,
                'annualized_return': results.annualized_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'profit_factor': results.profit_factor
            },
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'confidence': t.confidence,
                    'duration_days': t.duration_days,
                    'reasoning': t.reasoning[:100]  # Truncate reasoning
                } for t in results.trades
            ],
            'portfolio_history': [
                {'date': date, 'value': value} 
                for date, value in zip(results.dates, results.portfolio_values)
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            self.logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")