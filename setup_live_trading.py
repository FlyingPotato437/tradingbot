#!/usr/bin/env python3
"""
Setup script for Live Trading System
Installs dependencies and helps configure the system
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        return False
    print(f"[OK] Python version: {sys.version}")
    return True


def install_requirements():
    """Install required packages"""
    print("[INSTALL] Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install requirements: {e}")
        return False


def setup_alpaca_credentials():
    """Help user set up Alpaca API credentials"""
    print("\n[SETUP] Alpaca API Credentials")
    print("You need Alpaca API credentials for live trading.")
    print("Get them from: https://app.alpaca.markets/")
    
    setup_alpaca = input("Do you want to set up Alpaca credentials now? (y/N): ").lower()
    
    if setup_alpaca != 'y':
        print("[SKIP] Alpaca setup skipped")
        return True
    
    # Get credentials
    api_key = input("Enter your Alpaca API Key: ").strip()
    secret_key = getpass.getpass("Enter your Alpaca Secret Key: ").strip()
    
    if not api_key or not secret_key:
        print("[ERROR] Both API key and secret key are required")
        return False
    
    # Create .env file
    env_file = Path(".env")
    env_content = f"""# Alpaca API Credentials
ALPACA_API_KEY={api_key}
ALPACA_SECRET_KEY={secret_key}

# OpenAI API Key (if not already set)
# OPENAI_API_KEY=your_openai_key_here
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        
        print(f"[OK] Credentials saved to {env_file}")
        print("[INFO] Make sure to source these environment variables:")
        print("       source .env")
        print("       OR")
        print("       export ALPACA_API_KEY='your_key'")
        print("       export ALPACA_SECRET_KEY='your_secret'")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save credentials: {e}")
        return False


def setup_openai_api():
    """Help user set up OpenAI API key"""
    print("\n[SETUP] OpenAI API Key")
    print("You need an OpenAI API key for AI-powered trading decisions.")
    
    # Check if already set
    if os.getenv("OPENAI_API_KEY"):
        print("[OK] OpenAI API key already set in environment")
        return True
    
    setup_openai = input("Do you want to set up OpenAI API key now? (y/N): ").lower()
    
    if setup_openai != 'y':
        print("[SKIP] OpenAI setup skipped")
        return True
    
    api_key = getpass.getpass("Enter your OpenAI API Key: ").strip()
    
    if not api_key:
        print("[ERROR] OpenAI API key is required")
        return False
    
    # Append to .env file
    env_file = Path(".env")
    
    try:
        # Read existing content
        existing_content = ""
        if env_file.exists():
            with open(env_file, "r") as f:
                existing_content = f.read()
        
        # Add OpenAI key
        if "OPENAI_API_KEY" not in existing_content:
            with open(env_file, "a") as f:
                f.write(f"\n# OpenAI API Key\nOPENAI_API_KEY={api_key}\n")
        
        print("[OK] OpenAI API key saved")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save OpenAI API key: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file"""
    print("\n[SETUP] Creating sample configuration...")
    
    config_content = '''#!/usr/bin/env python3
"""
Sample Live Trading Configuration
Customize this file for your trading strategy
"""

from tradingagents.live_trading.live_trading_engine import TradingConfig

# Conservative configuration for beginners
CONSERVATIVE_CONFIG = TradingConfig(
    trading_symbols=['AAPL', 'MSFT', 'GOOGL'],
    decision_interval=600,  # 10 minutes
    max_position_size=0.05,  # 5% max position
    stop_loss_percent=0.02,  # 2% stop loss
    take_profit_percent=0.06,  # 6% take profit
    max_daily_trades=10,
    max_drawdown_percent=0.10,  # 10% max drawdown
    paper_trading=True,
    initial_capital=50000.0,
    use_alpaca=False,
    min_confidence_threshold=0.70,  # High confidence required
    risk_per_trade=0.01,  # Risk 1% per trade
)

# Aggressive configuration for experienced traders
AGGRESSIVE_CONFIG = TradingConfig(
    trading_symbols=['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    decision_interval=300,  # 5 minutes
    max_position_size=0.15,  # 15% max position
    stop_loss_percent=0.05,  # 5% stop loss
    take_profit_percent=0.15,  # 15% take profit
    max_daily_trades=50,
    max_drawdown_percent=0.20,  # 20% max drawdown
    paper_trading=True,
    initial_capital=250000.0,
    use_alpaca=True,
    min_confidence_threshold=0.60,
    risk_per_trade=0.025,  # Risk 2.5% per trade
)

# Live trading configuration (USE WITH EXTREME CAUTION)
LIVE_CONFIG = TradingConfig(
    trading_symbols=['AAPL', 'MSFT'],  # Start with just 2 stocks
    decision_interval=900,  # 15 minutes
    max_position_size=0.08,  # 8% max position
    stop_loss_percent=0.03,  # 3% stop loss
    take_profit_percent=0.10,  # 10% take profit
    max_daily_trades=5,  # Very conservative
    max_drawdown_percent=0.05,  # 5% max drawdown
    paper_trading=False,  # LIVE TRADING
    initial_capital=10000.0,  # Start small
    use_alpaca=True,
    min_confidence_threshold=0.80,  # Very high confidence
    risk_per_trade=0.01,  # Risk only 1% per trade
)
'''
    
    config_file = Path("trading_configs.py")
    
    try:
        with open(config_file, "w") as f:
            f.write(config_content)
        
        print(f"[OK] Sample configuration saved to {config_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create sample config: {e}")
        return False


def run_test():
    """Run a quick test to verify installation"""
    print("\n[TEST] Running installation test...")
    
    try:
        # Test imports
        print("[TEST] Testing imports...")
        
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from tradingagents.live_trading.live_trading_engine import LiveTradingEngine, TradingConfig
        
        print("[OK] All imports successful")
        
        # Test basic functionality
        print("[TEST] Testing basic functionality...")
        
        config = TradingConfig(
            trading_symbols=['AAPL'],
            paper_trading=True,
            initial_capital=10000.0,
            use_mathematical_analysis=True
        )
        
        engine = LiveTradingEngine(config)
        print("[OK] Trading engine created successfully")
        
        # Test portfolio
        portfolio_status = engine.get_portfolio_status()
        print(f"[OK] Portfolio initialized with ${portfolio_status['total_value']:,.2f}")
        
        print("[SUCCESS] All tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("LIVE TRADING SYSTEM SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        print("[ERROR] Failed to install requirements")
        return 1
    
    # Setup credentials
    setup_alpaca_credentials()
    setup_openai_api()
    
    # Create sample config
    create_sample_config()
    
    # Run test
    if not run_test():
        print("[ERROR] Installation test failed")
        return 1
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Source environment variables: source .env")
    print("2. Review trading_configs.py")
    print("3. Start with paper trading: python live_trading_main.py")
    print("4. Test the system thoroughly before live trading")
    print("\nWARNING: Never trade with money you cannot afford to lose!")
    
    return 0


if __name__ == "__main__":
    exit(main())