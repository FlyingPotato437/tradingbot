#!/usr/bin/env python3
"""
Example showing how to use TradingAgents for portfolio decisions
"""

print("""
===================================================================
EXAMPLE: Using TradingAgents Portfolio Advisory
===================================================================

SCENARIO 1: "I have $10,000 to invest, what stocks should I buy?"
-----------------------------------------------------------------
1. Run: python3 -m cli.main advisory
2. Select option 1: "Ask AI what stocks to buy"
3. Enter your investment amount: $10,000
4. Enter risk tolerance (1-3)
5. Enter time horizon (1-3)
6. The AI will recommend stocks based on your criteria
7. You can then run detailed AI analysis on any recommended stock

SCENARIO 2: "I own AAPL and NVDA, should I hold or sell?"
-----------------------------------------------------------------
1. Run: python3 -m cli.main advisory
2. Select option 2: "Give AI your portfolio"
3. Enter your holdings:
   - AAPL: 100 shares at $150 avg cost
   - NVDA: 50 shares at $400 avg cost
4. The AI will analyze each holding and advise what to do

SCENARIO 3: "Should I buy TSLA or GOOGL?"
-----------------------------------------------------------------
1. Run: python3 -m cli.main advisory
2. Select option 3: "Compare multiple stocks"
3. Enter stocks to compare: TSLA, GOOGL
4. The AI will analyze both and help you decide

The system uses multiple AI agents:
- Market Analyst: Technical indicators, price trends
- News Analyst: Latest news impact
- Fundamentals Analyst: Company financials
- Social Analyst: Market sentiment
- Research Team: Bull vs Bear debate
- Risk Management: Risk assessment
- Portfolio Manager: Final recommendation

Each analysis provides:
- BUY/SELL/HOLD recommendation
- Key insights from each analyst
- Investment reasoning
- Risk considerations

To start: python3 -m cli.main advisory
===================================================================
""")