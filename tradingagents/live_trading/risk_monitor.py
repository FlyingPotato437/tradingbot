import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """Represents a risk alert"""
    alert_id: str
    risk_type: str
    symbol: str
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False


class RiskMonitor:
    """Monitors portfolio risk and generates alerts"""
    
    def __init__(self, max_drawdown: float = 0.2, stop_loss: float = 0.05, 
                 position_size_limit: float = 0.1, correlation_limit: float = 0.7):
        # Risk thresholds
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.position_size_limit = position_size_limit
        self.correlation_limit = correlation_limit
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.portfolio_manager = None
        
        # Risk tracking
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.risk_callbacks: List[Callable] = []
        
        # Performance tracking
        self.peak_portfolio_value = 0.0
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.daily_trade_limit = 100
        
        # Emergency stops
        self.emergency_stop_triggered = False
        self.trading_halted = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self, portfolio_manager):
        """Start risk monitoring"""
        if self.is_monitoring:
            self.logger.warning("Risk monitoring is already running")
            return
        
        self.portfolio_manager = portfolio_manager
        self.peak_portfolio_value = portfolio_manager.get_total_value()
        
        self.logger.info("Starting risk monitoring...")
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Risk monitoring started")

    def stop_monitoring(self):
        """Stop risk monitoring"""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping risk monitoring...")
        self.is_monitoring = False
        
        # Wait for thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Risk monitoring stopped")

    def _monitoring_loop(self):
        """Main risk monitoring loop"""
        while self.is_monitoring:
            try:
                self._check_all_risks()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(60)

    def _check_all_risks(self):
        """Check all risk metrics"""
        if not self.portfolio_manager:
            return
        
        # Update peak portfolio value
        current_value = self.portfolio_manager.get_total_value()
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        # Check various risks
        self._check_drawdown_risk()
        self._check_position_size_risk()
        self._check_daily_loss_risk()
        self._check_stop_loss_risk()
        self._check_concentration_risk()
        self._check_liquidity_risk()

    def _check_drawdown_risk(self):
        """Check portfolio drawdown risk"""
        if not self.portfolio_manager:
            return
        
        current_value = self.portfolio_manager.get_total_value()
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        if drawdown > self.max_drawdown:
            self._create_alert(
                "MAX_DRAWDOWN",
                "PORTFOLIO",
                RiskLevel.CRITICAL,
                f"Portfolio drawdown {drawdown:.2%} exceeds maximum allowed {self.max_drawdown:.2%}",
                drawdown,
                self.max_drawdown
            )
            
            # Trigger emergency stop
            if drawdown > self.max_drawdown * 1.5:
                self._trigger_emergency_stop("Excessive drawdown")

    def _check_position_size_risk(self):
        """Check individual position size risk"""
        if not self.portfolio_manager:
            return
        
        total_value = self.portfolio_manager.get_total_value()
        positions = self.portfolio_manager.get_all_positions()
        
        for symbol, position in positions.items():
            position_pct = position.market_value / total_value
            
            if position_pct > self.position_size_limit:
                self._create_alert(
                    "POSITION_SIZE",
                    symbol,
                    RiskLevel.HIGH,
                    f"Position size {position_pct:.2%} exceeds limit {self.position_size_limit:.2%}",
                    position_pct,
                    self.position_size_limit
                )

    def _check_daily_loss_risk(self):
        """Check daily loss limits"""
        if not self.portfolio_manager:
            return
        
        # Get today's trades
        today_trades = self.portfolio_manager.get_trade_history(days=1)
        
        if len(today_trades) > self.daily_trade_limit:
            self._create_alert(
                "DAILY_TRADES",
                "PORTFOLIO", 
                RiskLevel.HIGH,
                f"Daily trades {len(today_trades)} exceed limit {self.daily_trade_limit}",
                len(today_trades),
                self.daily_trade_limit
            )
        
        # Calculate daily P&L
        daily_pnl = sum(self._estimate_trade_pnl(trade) for trade in today_trades)
        portfolio_value = self.portfolio_manager.get_total_value()
        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        
        if daily_loss_pct > self.daily_loss_limit:
            self._create_alert(
                "DAILY_LOSS",
                "PORTFOLIO",
                RiskLevel.CRITICAL,
                f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.daily_loss_limit:.2%}",
                daily_loss_pct,
                self.daily_loss_limit
            )

    def _check_stop_loss_risk(self):
        """Check stop loss violations"""
        if not self.portfolio_manager:
            return
        
        positions = self.portfolio_manager.get_all_positions()
        
        for symbol, position in positions.items():
            if position.current_price <= 0:
                continue
                
            loss_pct = (position.avg_cost - position.current_price) / position.avg_cost
            
            if loss_pct > self.stop_loss:
                self._create_alert(
                    "STOP_LOSS",
                    symbol,
                    RiskLevel.HIGH,
                    f"Position loss {loss_pct:.2%} exceeds stop loss {self.stop_loss:.2%}",
                    loss_pct,
                    self.stop_loss
                )

    def _check_concentration_risk(self):
        """Check sector/industry concentration risk"""
        # This is a simplified check - in practice you'd get sector data
        positions = self.portfolio_manager.get_all_positions()
        total_value = self.portfolio_manager.get_total_value()
        
        # Check if any single position is too large
        for symbol, position in positions.items():
            concentration = position.market_value / total_value
            if concentration > 0.3:  # 30% concentration limit
                self._create_alert(
                    "CONCENTRATION",
                    symbol,
                    RiskLevel.MEDIUM,
                    f"Position concentration {concentration:.2%} is high",
                    concentration,
                    0.3
                )

    def _check_liquidity_risk(self):
        """Check liquidity risk based on position sizes"""
        # Simplified liquidity check
        positions = self.portfolio_manager.get_all_positions()
        
        for symbol, position in positions.items():
            # Positions over $10k might have liquidity issues for some stocks
            if position.market_value > 10000:
                self._create_alert(
                    "LIQUIDITY",
                    symbol,
                    RiskLevel.LOW,
                    f"Large position {position.market_value:.0f} may have liquidity risk",
                    position.market_value,
                    10000
                )

    def _create_alert(self, risk_type: str, symbol: str, risk_level: RiskLevel, 
                     message: str, value: float, threshold: float):
        """Create a risk alert"""
        alert_id = f"{risk_type}_{symbol}_{int(time.time())}"
        
        # Don't create duplicate alerts
        existing_key = f"{risk_type}_{symbol}"
        if existing_key in self.active_alerts:
            return
        
        alert = RiskAlert(
            alert_id=alert_id,
            risk_type=risk_type,
            symbol=symbol,
            risk_level=risk_level,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.active_alerts[existing_key] = alert
        self.alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self.risk_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in risk callback: {e}")
        
        self.logger.warning(f"Risk Alert: {message}")

    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        if self.emergency_stop_triggered:
            return
        
        self.emergency_stop_triggered = True
        self.trading_halted = True
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Create critical alert
        self._create_alert(
            "EMERGENCY_STOP",
            "SYSTEM",
            RiskLevel.CRITICAL,
            f"Emergency stop triggered: {reason}",
            1.0,
            0.0
        )

    def _estimate_trade_pnl(self, trade) -> float:
        """Estimate P&L for a trade (simplified)"""
        # This is a simplified calculation
        # In practice, you'd need more sophisticated P&L calculation
        return 0.0

    def check_trade_risk(self, decision, portfolio_manager) -> bool:
        """Check if a trade passes risk checks"""
        if self.trading_halted:
            return False
        
        # Check position size limit
        if decision.action == 'BUY':
            total_value = portfolio_manager.get_total_value()
            position_value = decision.quantity * decision.price
            position_pct = position_value / total_value
            
            if position_pct > self.position_size_limit:
                self.logger.warning(f"Trade blocked: position size {position_pct:.2%} exceeds limit")
                return False
        
        # Check available cash
        if decision.action == 'BUY':
            required_cash = decision.quantity * decision.price + 1.0  # Include commission
            available_cash = portfolio_manager.get_available_capital()
            
            if required_cash > available_cash:
                self.logger.warning(f"Trade blocked: insufficient cash")
                return False
        
        # Check if position exists for sell orders
        if decision.action == 'SELL':
            position = portfolio_manager.get_position(decision.symbol)
            if not position or position.quantity < decision.quantity:
                self.logger.warning(f"Trade blocked: insufficient position to sell")
                return False
        
        return True

    def acknowledge_alert(self, alert_key: str):
        """Acknowledge a risk alert"""
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].acknowledged = True
            del self.active_alerts[alert_key]

    def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        if not self.portfolio_manager:
            return {}
        
        current_value = self.portfolio_manager.get_total_value()
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        active_alerts_by_level = {}
        for alert in self.active_alerts.values():
            level = alert.risk_level.value
            if level not in active_alerts_by_level:
                active_alerts_by_level[level] = 0
            active_alerts_by_level[level] += 1
        
        return {
            'current_drawdown_pct': drawdown * 100,
            'max_drawdown_limit_pct': self.max_drawdown * 100,
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'trading_halted': self.trading_halted,
            'active_alerts_count': len(self.active_alerts),
            'active_alerts_by_level': active_alerts_by_level,
            'total_alerts_24h': len(self.get_alert_history(24)),
            'peak_portfolio_value': self.peak_portfolio_value,
            'current_portfolio_value': current_value,
            'risk_limits': {
                'max_drawdown_pct': self.max_drawdown * 100,
                'stop_loss_pct': self.stop_loss * 100,
                'position_size_limit_pct': self.position_size_limit * 100,
                'daily_loss_limit_pct': self.daily_loss_limit * 100
            }
        }

    def add_risk_callback(self, callback: Callable):
        """Add a callback for risk alerts"""
        self.risk_callbacks.append(callback)

    def remove_risk_callback(self, callback: Callable):
        """Remove a risk callback"""
        if callback in self.risk_callbacks:
            self.risk_callbacks.remove(callback)

    def reset_emergency_stop(self):
        """Reset emergency stop (use with caution)"""
        self.emergency_stop_triggered = False
        self.trading_halted = False
        self.logger.info("Emergency stop reset - trading enabled")

    def update_risk_limits(self, max_drawdown: float = None, stop_loss: float = None,
                          position_size_limit: float = None, daily_loss_limit: float = None):
        """Update risk limits"""
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
        if stop_loss is not None:
            self.stop_loss = stop_loss
        if position_size_limit is not None:
            self.position_size_limit = position_size_limit
        if daily_loss_limit is not None:
            self.daily_loss_limit = daily_loss_limit
        
        self.logger.info("Risk limits updated")

    def simulate_risk_scenario(self, scenario_type: str):
        """Simulate risk scenarios for testing"""
        if scenario_type == "drawdown":
            self._create_alert(
                "TEST_DRAWDOWN",
                "PORTFOLIO",
                RiskLevel.HIGH,
                "Simulated drawdown risk",
                0.15,
                0.1
            )
        elif scenario_type == "position_size":
            self._create_alert(
                "TEST_POSITION",
                "AAPL",
                RiskLevel.MEDIUM,
                "Simulated position size risk",
                0.25,
                0.2
            )
        
        self.logger.info(f"Simulated risk scenario: {scenario_type}")