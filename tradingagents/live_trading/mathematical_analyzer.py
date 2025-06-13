import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    atr: float
    volume_sma: float
    price_to_sma_ratio: float
    volatility: float


@dataclass
class RiskMetrics:
    """Risk analysis metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk: float
    expected_shortfall: float
    beta: float
    alpha: float
    correlation_to_market: float


@dataclass
class QuantitativeSignals:
    """Quantitative trading signals"""
    momentum_score: float
    mean_reversion_score: float
    trend_strength: float
    volatility_regime: str
    market_regime: str
    confidence_interval: Tuple[float, float]
    probability_of_profit: float


class MathematicalAnalyzer:
    """Mathematical and statistical analysis for trading decisions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.market_data_cache = {}
        self.lookback_period = 252  # 1 year of trading days
        
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators"""
        try:
            close = price_data['Close']
            high = price_data['High']
            low = price_data['Low']
            volume = price_data['Volume']
            
            # RSI calculation
            rsi = self._calculate_rsi(close, 14)
            
            # MACD calculation
            macd, macd_signal = self._calculate_macd(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
            
            # Moving averages
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            sma_50 = close.rolling(window=50).mean().iloc[-1]
            ema_12 = close.ewm(span=12).mean().iloc[-1]
            ema_26 = close.ewm(span=26).mean().iloc[-1]
            
            # ATR
            atr = self._calculate_atr(high, low, close, 14)
            
            # Volume analysis
            volume_sma = volume.rolling(window=20).mean().iloc[-1]
            
            # Price to SMA ratio
            price_to_sma_ratio = close.iloc[-1] / sma_20 if sma_20 > 0 else 1.0
            
            # Volatility
            volatility = self._calculate_volatility(close, 20)
            
            return TechnicalIndicators(
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                bollinger_upper=bb_upper,
                bollinger_lower=bb_lower,
                bollinger_middle=bb_middle,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                atr=atr,
                volume_sma=volume_sma,
                price_to_sma_ratio=price_to_sma_ratio,
                volatility=volatility
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return self._get_default_indicators()
    
    def calculate_risk_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Remove NaN values
            returns = returns.dropna()
            
            if len(returns) < 30:  # Need minimum data
                return self._get_default_risk_metrics()
            
            # Basic statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
            sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)
            
            # Expected Shortfall (Conditional VaR)
            tail_losses = returns[returns <= var_95]
            expected_shortfall = tail_losses.mean() if len(tail_losses) > 0 else var_95
            
            # Beta and Alpha (if benchmark provided)
            beta = 0.0
            alpha = 0.0
            correlation_to_market = 0.0
            
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                # Remove NaN values from both series
                aligned_data = pd.DataFrame({
                    'returns': returns,
                    'benchmark': benchmark_returns
                }).dropna()
                
                if len(aligned_data) > 10:
                    beta = np.cov(aligned_data['returns'], aligned_data['benchmark'])[0,1] / np.var(aligned_data['benchmark'])
                    alpha = aligned_data['returns'].mean() - beta * aligned_data['benchmark'].mean()
                    correlation_to_market = aligned_data['returns'].corr(aligned_data['benchmark'])
            
            return RiskMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                value_at_risk=var_95,
                expected_shortfall=expected_shortfall,
                beta=beta,
                alpha=alpha,
                correlation_to_market=correlation_to_market
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    def generate_quantitative_signals(self, price_data: pd.DataFrame, 
                                    technical_indicators: TechnicalIndicators) -> QuantitativeSignals:
        """Generate quantitative trading signals"""
        try:
            close = price_data['Close']
            returns = close.pct_change().dropna()
            
            # Momentum score (combining multiple momentum indicators)
            rsi_momentum = (technical_indicators.rsi - 50) / 50  # Normalize RSI
            macd_momentum = 1 if technical_indicators.macd > technical_indicators.macd_signal else -1
            price_momentum = (technical_indicators.price_to_sma_ratio - 1) * 2  # Normalize price vs SMA
            momentum_score = (rsi_momentum + macd_momentum + price_momentum) / 3
            
            # Mean reversion score
            bb_position = self._calculate_bollinger_position(
                close.iloc[-1], 
                technical_indicators.bollinger_lower,
                technical_indicators.bollinger_upper
            )
            rsi_reversion = -1 * (technical_indicators.rsi - 50) / 50  # Opposite of momentum
            mean_reversion_score = (bb_position + rsi_reversion) / 2
            
            # Trend strength using ADX-like calculation
            trend_strength = self._calculate_trend_strength(close)
            
            # Volatility regime classification
            volatility_regime = self._classify_volatility_regime(technical_indicators.volatility, close)
            
            # Market regime classification
            market_regime = self._classify_market_regime(returns, trend_strength)
            
            # Confidence interval for next period return
            confidence_interval = self._calculate_confidence_interval(returns)
            
            # Probability of profit estimation
            probability_of_profit = self._estimate_profit_probability(
                momentum_score, mean_reversion_score, trend_strength, volatility_regime
            )
            
            return QuantitativeSignals(
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                trend_strength=trend_strength,
                volatility_regime=volatility_regime,
                market_regime=market_regime,
                confidence_interval=confidence_interval,
                probability_of_profit=probability_of_profit
            )
            
        except Exception as e:
            self.logger.error(f"Error generating quantitative signals: {e}")
            return self._get_default_signals()
    
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float,
                              entry_price: float, stop_loss_price: float) -> int:
        """Calculate optimal position size using Kelly Criterion and risk management"""
        try:
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share <= 0:
                return 0
            
            # Maximum risk amount
            max_risk_amount = portfolio_value * risk_per_trade
            
            # Position size based on risk
            position_size = int(max_risk_amount / risk_per_share)
            
            # Apply additional constraints
            max_position_value = portfolio_value * 0.2  # Max 20% of portfolio per position
            max_shares_by_value = int(max_position_value / entry_price)
            
            # Return minimum of risk-based and value-based position sizes
            return min(position_size, max_shares_by_value)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def calculate_stop_loss_take_profit(self, entry_price: float, atr: float,
                                      trend_direction: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        try:
            # ATR-based stop loss (2x ATR)
            stop_loss_distance = 2.0 * atr
            
            # Take profit at 3:1 risk-reward ratio
            take_profit_distance = 3.0 * stop_loss_distance
            
            if trend_direction.upper() == 'BUY':
                stop_loss = entry_price - stop_loss_distance
                take_profit = entry_price + take_profit_distance
            else:  # SELL
                stop_loss = entry_price + stop_loss_distance
                take_profit = entry_price - take_profit_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss/take profit: {e}")
            return entry_price * 0.95, entry_price * 1.15  # Default 5% stop, 15% profit
    
    def score_trading_opportunity(self, technical_indicators: TechnicalIndicators,
                                quantitative_signals: QuantitativeSignals,
                                risk_metrics: RiskMetrics) -> Dict[str, float]:
        """Score trading opportunity across multiple dimensions"""
        try:
            # Technical score (0-100)
            technical_score = self._calculate_technical_score(technical_indicators)
            
            # Momentum score (0-100)
            momentum_score = (quantitative_signals.momentum_score + 1) * 50  # Convert to 0-100
            
            # Risk-adjusted score (0-100)
            risk_score = self._calculate_risk_score(risk_metrics)
            
            # Probability score (0-100)
            probability_score = quantitative_signals.probability_of_profit * 100
            
            # Overall composite score
            weights = {
                'technical': 0.3,
                'momentum': 0.25,
                'risk_adjusted': 0.25,
                'probability': 0.2
            }
            
            composite_score = (
                technical_score * weights['technical'] +
                momentum_score * weights['momentum'] +
                risk_score * weights['risk_adjusted'] +
                probability_score * weights['probability']
            )
            
            return {
                'technical_score': technical_score,
                'momentum_score': momentum_score,
                'risk_score': risk_score,
                'probability_score': probability_score,
                'composite_score': composite_score
            }
            
        except Exception as e:
            self.logger.error(f"Error scoring trading opportunity: {e}")
            return {
                'technical_score': 50.0,
                'momentum_score': 50.0,
                'risk_score': 50.0,
                'probability_score': 50.0,
                'composite_score': 50.0
            }
    
    # Helper methods
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd.iloc[-1], signal.iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> float:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate historical volatility"""
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        return volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.0
    
    def _calculate_bollinger_position(self, price: float, bb_lower: float, bb_upper: float) -> float:
        """Calculate position within Bollinger Bands (-1 to 1)"""
        if bb_upper == bb_lower:
            return 0.0
        return (price - bb_lower) / (bb_upper - bb_lower) * 2 - 1
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength (0-1)"""
        if len(prices) < 20:
            return 0.5
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Normalize slope and use R-squared as strength
        return abs(r_value)
    
    def _classify_volatility_regime(self, volatility: float, prices: pd.Series) -> str:
        """Classify volatility regime"""
        if len(prices) < 50:
            return "UNKNOWN"
        
        vol_history = prices.pct_change().rolling(window=20).std() * np.sqrt(252)
        vol_percentile = stats.percentileofscore(vol_history.dropna(), volatility)
        
        if vol_percentile > 80:
            return "HIGH"
        elif vol_percentile > 60:
            return "ELEVATED"
        elif vol_percentile > 40:
            return "NORMAL"
        elif vol_percentile > 20:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _classify_market_regime(self, returns: pd.Series, trend_strength: float) -> str:
        """Classify market regime"""
        if len(returns) < 20:
            return "UNKNOWN"
        
        recent_returns = returns.tail(20)
        mean_return = recent_returns.mean()
        
        if trend_strength > 0.7:
            return "TRENDING_UP" if mean_return > 0 else "TRENDING_DOWN"
        elif trend_strength > 0.3:
            return "WEAK_TREND"
        else:
            return "RANGE_BOUND"
    
    def _calculate_confidence_interval(self, returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for next period return"""
        if len(returns) < 10:
            return (-0.05, 0.05)
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Student's t-distribution for small samples
        dof = len(returns) - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, dof)
        
        margin_error = t_critical * std_return / np.sqrt(len(returns))
        
        return (mean_return - margin_error, mean_return + margin_error)
    
    def _estimate_profit_probability(self, momentum_score: float, mean_reversion_score: float,
                                   trend_strength: float, volatility_regime: str) -> float:
        """Estimate probability of profitable trade"""
        # Base probability
        base_prob = 0.5
        
        # Adjust for momentum
        momentum_adjustment = momentum_score * 0.2
        
        # Adjust for trend strength
        trend_adjustment = trend_strength * 0.15
        
        # Adjust for volatility regime
        vol_adjustments = {
            "VERY_LOW": 0.05,
            "LOW": 0.02,
            "NORMAL": 0.0,
            "ELEVATED": -0.05,
            "HIGH": -0.1,
            "UNKNOWN": 0.0
        }
        vol_adjustment = vol_adjustments.get(volatility_regime, 0.0)
        
        # Calculate final probability
        prob = base_prob + momentum_adjustment + trend_adjustment + vol_adjustment
        
        # Clamp between 0.1 and 0.9
        return max(0.1, min(0.9, prob))
    
    def _calculate_technical_score(self, indicators: TechnicalIndicators) -> float:
        """Calculate technical analysis score (0-100)"""
        score = 50.0  # Base score
        
        # RSI contribution
        if indicators.rsi > 70:
            score -= 20  # Overbought
        elif indicators.rsi < 30:
            score += 20  # Oversold
        
        # MACD contribution
        if indicators.macd > indicators.macd_signal:
            score += 15
        else:
            score -= 15
        
        # Moving average contribution
        if indicators.price_to_sma_ratio > 1.02:
            score += 10
        elif indicators.price_to_sma_ratio < 0.98:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate risk-adjusted score (0-100)"""
        score = 50.0  # Base score
        
        # Sharpe ratio contribution
        if metrics.sharpe_ratio > 1.0:
            score += 20
        elif metrics.sharpe_ratio > 0.5:
            score += 10
        elif metrics.sharpe_ratio < 0:
            score -= 20
        
        # Max drawdown contribution
        if abs(metrics.max_drawdown) < 0.05:
            score += 15
        elif abs(metrics.max_drawdown) > 0.2:
            score -= 20
        
        return max(0, min(100, score))
    
    def _get_default_indicators(self) -> TechnicalIndicators:
        """Return default technical indicators"""
        return TechnicalIndicators(
            rsi=50.0, macd=0.0, macd_signal=0.0,
            bollinger_upper=100.0, bollinger_lower=100.0, bollinger_middle=100.0,
            sma_20=100.0, sma_50=100.0, ema_12=100.0, ema_26=100.0,
            atr=1.0, volume_sma=1000000, price_to_sma_ratio=1.0, volatility=0.2
        )
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics"""
        return RiskMetrics(
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
            value_at_risk=0.0, expected_shortfall=0.0, beta=1.0,
            alpha=0.0, correlation_to_market=0.0
        )
    
    def _get_default_signals(self) -> QuantitativeSignals:
        """Return default quantitative signals"""
        return QuantitativeSignals(
            momentum_score=0.0, mean_reversion_score=0.0, trend_strength=0.5,
            volatility_regime="NORMAL", market_regime="UNKNOWN",
            confidence_interval=(-0.05, 0.05), probability_of_profit=0.5
        )