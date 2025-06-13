#!/usr/bin/env python3
"""
Test script to demonstrate the portfolio advisory feature
"""
import subprocess
import sys

def main():
    print("=" * 70)
    print("PORTFOLIO ADVISORY FEATURE TEST")
    print("=" * 70)
    
    print("\nThis demonstrates the new portfolio advisory feature that:")
    print("1. Allows you to ask AI what stocks to buy")
    print("2. Analyze your current portfolio and get advice")
    print("3. Compare multiple stocks using AI analysis")
    
    print("\nTo run the portfolio advisory:")
    print("1. python3 -m cli.main advisory")
    print("   OR")
    print("2. python3 -m cli.main (and select option 2)")
    
    print("\nThe system uses these AI agents:")
    print("- Market Analyst (technical analysis)")
    print("- News Analyst (latest events)")
    print("- Fundamentals Analyst (financials)")
    print("- Social Analyst (sentiment)")
    print("- Research Team (bull vs bear debate)")
    print("- Risk Management Team")
    print("- Portfolio Manager (final decision)")
    
    print("\nWould you like to run it now? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        subprocess.run([sys.executable, "-m", "cli.main", "advisory"])

if __name__ == "__main__":
    main()