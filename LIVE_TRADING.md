# Live Trading System

**Professional real-time autonomous stock trading system using TradingAgents**

## IMPORTANT WARNING

This system can execute real trades and lose real money. Always start with paper trading to test your strategies. Use live trading only after thorough testing and with money you can afford to lose. 

This system is designed specifically for stock trading and integrates with Alpaca for live execution.

## Features

- **Real-time market data streaming** - Live price feeds from Yahoo Finance
- **AI-powered trading decisions** - Multi-agent analysis (news, technical, fundamental)
- **Mathematical analysis framework** - Technical indicators, risk metrics, quantitative signals
- **Alpaca integration** - Professional broker API for live trading
- **Automated risk management** - Stop-loss, take-profit, position sizing
- **Portfolio management & tracking** - Real-time P&L, positions, and performance
- **Paper trading simulation** - Test strategies without risk
- **Live monitoring dashboard** - Professional command-line interface
- **Advanced position sizing** - Kelly criterion and ATR-based sizing
- **Risk monitoring** - Drawdown limits, position size controls, emergency stops

## Quick Start

### Prerequisites

For live trading with Alpaca, you'll need:
1. Alpaca account (paper or live)
2. API keys from Alpaca dashboard
3. Environment variables set:
   ```bash
   export ALPACA_API_KEY="your_api_key"
   export ALPACA_SECRET_KEY="your_secret_key"
   ```

### Option 1: Interactive Main Script
```bash
python live_trading_main.py
```

### Option 2: CLI Interface
```bash
# Start with default symbols (paper trading)
python -m tradingagents.live_trading.cli start

# Start with custom symbols and capital
python -m tradingagents.live_trading.cli start --symbols AAPL NVDA TSLA --capital 50000

# Enable live trading (USE WITH EXTREME CAUTION)
python -m tradingagents.live_trading.cli start --live
```

## Configuration

### Trading Parameters
- **Trading Symbols**: Default `['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']`
- **Decision Interval**: 5 minutes (300 seconds)
- **Max Position Size**: 10% of portfolio per position
- **Stop Loss**: 5% per position
- **Take Profit**: 15% per position
- **Max Daily Trades**: 50
- **Max Drawdown**: 20% portfolio-wide

### Mathematical Analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, moving averages
- **Risk Metrics**: Sharpe ratio, Sortino ratio, VaR, expected shortfall
- **Quantitative Signals**: Momentum, mean reversion, trend strength
- **Confidence Scoring**: Multi-dimensional opportunity scoring
- **Position Sizing**: Kelly criterion and risk-based sizing

### Risk Controls
- **Emergency Stop**: Triggered on excessive drawdown
- **Position Size Limits**: Prevents over-concentration  
- **Daily Loss Limits**: Stops trading on large daily losses
- **Confidence Thresholds**: Minimum confidence required for trades
- **ATR-based Stops**: Dynamic stop losses based on volatility

## Usage Commands

### Interactive Commands
When running the system, you can use these commands:

- `portfolio` - Show current portfolio status
- `stats` - Show trading statistics
- `risk` - Show risk monitor status
- `monitor` - Start live dashboard
- `decide <SYMBOL>` - Force trading decision for symbol
- `add <SYMBOL>` - Add symbol to tracking list
- `remove <SYMBOL>` - Remove symbol from tracking
- `quit` - Stop trading

### Live Dashboard
The live dashboard provides real-time monitoring with:
- Portfolio value and P&L
- Current positions and performance
- Trading statistics
- Risk status
- Recent decisions

## System Architecture

### Core Components

1. **LiveTradingEngine**: Main orchestrator with mathematical analysis integration
2. **PortfolioManager**: Tracks positions, cash, P&L with persistence
3. **TradeExecutor/AlpacaExecutor**: Handles order execution (paper/live)
4. **MarketDataStream**: Real-time market data feed from Yahoo Finance
5. **RiskMonitor**: Advanced risk management and alerts
6. **MathematicalAnalyzer**: Technical analysis and quantitative modeling
7. **TradingGraph**: AI decision making (existing multi-agent system)

### Data Flow
```
Market Data → AI Analysis → Trading Decision → Risk Check → Order Execution → Portfolio Update
```

### Decision Making Process
1. **Market Data Collection**: Real-time price, volume, historical data
2. **Mathematical Analysis**: 
   - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
   - Risk metrics (Sharpe, Sortino, VaR, drawdown)
   - Quantitative signals (momentum, mean reversion, trend)
   - Confidence scoring across multiple dimensions
3. **Multi-Agent AI Analysis**: 
   - Market Analyst (technical analysis)
   - News Analyst (sentiment analysis) 
   - Fundamentals Analyst (financial metrics)
   - Social Media Analyst (social sentiment)
4. **Investment Debate**: Bull vs Bear researchers debate
5. **Confidence Integration**: Combine AI (70%) + Math (20%) + Probability (10%)
6. **Risk Assessment**: Risk management review and position sizing
7. **Final Decision**: BUY/SELL/HOLD with optimal quantity
8. **Execution**: Professional order placement via Alpaca or simulation

## Example Usage

### Basic Paper Trading
```python
from tradingagents.live_trading import LiveTradingEngine, TradingConfig

# Create configuration
config = TradingConfig(
    trading_symbols=['AAPL', 'NVDA'],
    paper_trading=True,
    initial_capital=100000.0
)

# Start trading
engine = LiveTradingEngine(config)
engine.start_trading()

# Monitor portfolio
portfolio = engine.get_portfolio_status()
print(f"Portfolio Value: ${portfolio['total_value']:,.2f}")

# Force a decision
decision = engine.force_decision('AAPL')
print(f"Decision: {decision.action} {decision.quantity} shares")

# Stop trading
engine.stop_trading()
```

### Advanced Configuration
```python
config = TradingConfig(
    trading_symbols=['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL'],
    decision_interval=300,  # 5 minutes
    max_position_size=0.15,  # 15% max position
    stop_loss_percent=0.03,  # 3% stop loss
    take_profit_percent=0.20,  # 20% take profit
    max_daily_trades=25,
    max_drawdown_percent=0.15,  # 15% max drawdown
    paper_trading=True,
    initial_capital=250000.0,
    use_alpaca=True,  # Enable Alpaca integration
    min_confidence_threshold=0.65,  # Require 65% confidence
    use_mathematical_analysis=True,  # Enable math analysis
    risk_per_trade=0.02,  # Risk 2% per trade
    lookback_days=252  # 1 year of data
)
```

## Risk Management

### Automatic Risk Controls
- **Position Size Limits**: Maximum 10% of portfolio per position
- **Stop Loss Orders**: Automatic 5% stop loss per position
- **Drawdown Monitoring**: Emergency stop at 20% portfolio drawdown
- **Daily Trade Limits**: Maximum 50 trades per day
- **Liquidity Checks**: Prevents oversized positions

### Risk Alerts
The system generates alerts for:
- Excessive drawdown
- Large position sizes
- Stop loss violations
- High daily losses
- Liquidity concerns

### Emergency Stops
Automatic trading halt triggers:
- Portfolio drawdown > 30%
- System errors
- Risk limit violations
- Manual emergency stop

## Performance Tracking

### Portfolio Metrics
- Total portfolio value
- Cash vs positions allocation
- Realized and unrealized P&L
- Total return percentage
- Maximum drawdown
- Sharpe ratio (calculated over time)

### Trading Metrics
- Total number of decisions
- Win rate percentage
- Average trade P&L
- Best and worst trades
- Trading frequency
- Commission costs

### Risk Metrics
- Current drawdown
- Position concentration
- Daily loss limits
- Risk-adjusted returns

## Trading Modes

### Paper Trading (Default)
- Simulates real trading without risk
- Uses actual market prices with slippage
- Perfect for strategy testing
- No real money involved
- Immediate order fills

### Live Trading with Alpaca
- **WARNING: USE WITH EXTREME CAUTION**
- Requires Alpaca API keys
- Real money at risk
- Professional-grade execution
- Subject to market conditions and regulations
- Pattern day trader rules apply
- Commission-free stock trading

## Troubleshooting

### Common Issues

**Market Data Connection**
- Check internet connection
- Verify symbol exists and is tradeable
- Check market hours (9:30 AM - 4:00 PM ET)

**AI Decision Errors**
- Verify OpenAI API key is set
- Check rate limits and API credits
- Ensure sufficient historical data available

**Alpaca Integration Issues**
- Verify API keys are correctly set
- Check account status and trading permissions
- Ensure market is open for live trading
- Verify sufficient buying power

**Performance Issues**
- Reduce number of tracked symbols
- Increase decision interval
- Check system resources

**Risk Alerts**
- Review risk parameters
- Check portfolio allocation
- Monitor drawdown levels

### Logs and Debugging
- Log files: `live_trading.log`
- Enable debug mode in TradingConfig
- Check console output for errors
- Monitor system resource usage

## Legal Disclaimer

This software is for educational and research purposes only. The authors and contributors are not responsible for any financial losses incurred through the use of this system. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## Contributing

To contribute to the live trading system:
1. Test thoroughly with paper trading
2. Add comprehensive unit tests
3. Follow existing code patterns
4. Document new features
5. Submit pull requests with detailed descriptions

## Support

For issues or questions:
1. Check existing documentation
2. Review troubleshooting section
3. Search existing issues
4. Create new issue with detailed description