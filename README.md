
# HaruPyQuant Trading Bot

A Python-based algorithmic trading bot for MetaTrader 5 platform.

## Features

- Real-time market data processing
- Multiple trading strategies
- Risk management system
- Performance analysis and reporting
- Backtesting capabilities
- Web dashboard for monitoring
- Automated trading execution

## Requirements

- Python 3.13.2 or higher
- MetaTrader 5 platform
- MetaTrader 5 Python package
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your settings
5. Run the bot:
   ```bash
   python main.py
   ```

## Project Structure

```
haru_trader/
│
├── README.md                 # Project overview and documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
├── .env.example              # Template for environment variables
├── .gitignore                # Git ignore file
├── config.ini                # Configuration file
├── main.py                   # Main entry point script
│
├── app/                      # Main application package
│   ├── __init__.py           # Package initializer
│   │
│   ├── core/                 # Core system components
│   │   ├── __init__.py
│   │   ├── bot.py            # Main bot class
│   │   ├── constants.py      # System-wide constants
│   │   ├── exceptions.py     # Custom exception classes
│   │   └── event_system.py   # Event handling framework
│   │
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py       # Application settings manager
│   │   └── credentials.py    # Secure credential handling
│   │
│   ├── mt5/                  # MT5 integration module
│   │   ├── __init__.py
│   │   ├── client.py         # MT5 connection client
│   │   ├── data_feed.py      # Market data operations
│   │   ├── symbols.py        # Symbol management
│   │   └── timeframes.py     # Timeframe handling
│   │
│   ├── trading/              # Trading operations
│   │   ├── __init__.py
│   │   ├── order.py          # Order placement and management
│   │   ├── position.py       # Position tracking
│   │   ├── risk.py           # Risk calculation and management
│   │   └── execution.py      # Trade execution optimization
│   │
│   ├── strategy/             # Trading strategies
│   │   ├── __init__.py
│   │   ├── base.py           # Strategy base class
│   │   ├── indicators/       # Technical indicators
│   │   │   ├── __init__.py
│   │   │   ├── standard.py   # Standard indicators
│   │   │   └── custom.py     # Custom indicators
│   │   │
│   │   ├── signal.py         # Signal generation framework
│   │   ├── screener.py       # Symbol screening
│   │   ├── risk_management.py # Strategy-specific risk rules
│   │   │
│   │   └── strategies/       # Strategy implementations
│   │       ├── __init__.py
│   │       ├── trend_following.py
│   │       ├── mean_reversion.py
│   │       ├── breakout.py
│   │       └── scalping.py
│   │
│   ├── analysis/             # Performance analysis
│   │   ├── __init__.py
│   │   ├── performance.py    # Performance metrics
│   │   ├── trade_stats.py    # Trade statistics
│   │   ├── drawdown.py       # Drawdown analysis
│   │   └── reporting.py      # Report generation
│   │
│   ├── backtest/             # Backtesting system
│   │   ├── __init__.py
│   │   ├── engine.py         # Backtesting engine
│   │   ├── data_loader.py    # Historical data loading
│   │   ├── simulator.py      # Market simulation
│   │   ├── results.py        # Results processing
│   │   └── visualization.py  # Backtest visualization
│   │
│   ├── optimization/         # Strategy optimization
│   │   ├── __init__.py
│   │   ├── optimizer.py      # Optimization framework
│   │   ├── walk_forward.py   # Walk-forward analysis
│   │   ├── monte_carlo.py    # Monte Carlo simulation
│   │   ├── cross_validation.py # Cross-validation
│   │   ├── genetic.py        # Genetic algorithm
│   │   ├── bayesian.py       # Bayesian optimization
│   │   └── metrics.py        # Optimization metrics
│   │
│   ├── dashboard/            # Web dashboard
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI application
│   │   ├── auth.py           # Authentication
│   │   ├── routes/           # API routes
│   │   │   ├── __init__.py
│   │   │   ├── account.py
│   │   │   ├── performance.py
│   │   │   └── strategy.py
│   │   ├── static/           # Static assets
│   │   └── templates/        # HTML templates
│   │
│   ├── database/             # Database operations
│   │   ├── __init__.py
│   │   ├── connection.py     # Database connection
│   │   ├── models.py         # Data models
│   │   ├── queries.py        # SQL queries
│   │   └── migrations/       # Schema migrations
│   │       ├── __init__.py
│   │       └── versions/     # Migration versions
│   │
│   ├── notification/         # Alert and notification system
│   │   ├── __init__.py
│   │   ├── telegram.py       # Telegram integration
│   │   ├── email.py          # Email notifications
│   │   └── formatter.py      # Message formatting
│   │
│   ├── integration/          # External integrations
│   │   ├── __init__.py
│   │   ├── news.py           # News API integration
│   │   ├── economic.py       # Economic calendar
│   │   └── sentiment.py      # Market sentiment analysis
│   │
│   ├── live_trading/         # Live trading module
│   │   ├── __init__.py
│   │   ├── executor.py       # Real-time execution
│   │   ├── monitor.py        # Performance monitoring
│   │   └── recovery.py       # Failover mechanisms
│   │
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── logger.py         # Logging configuration
│       ├── validation.py     # Input validation
│       ├── timeutils.py      # Time-related utilities
│       ├── math.py           # Mathematical helpers
│       └── serialization.py  # Data serialization
│
├── models/                   # Data models
│   ├── __init__.py
│   ├── market_data.py        # Market data models
│   ├── trade.py              # Trade and position models
│   ├── account.py            # Account and balance models
│   └── performance.py        # Performance metrics models
│
├── scripts/                  # Utility scripts
│   ├── __init__.py
│   ├── setup_database.py     # Database initialization
│   ├── backfill_data.py      # Historical data importer
│   └── health_check.py       # System health check
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Test configuration
│   ├── unit/                 # Unit tests
│   │   ├── __init__.py
│   │   ├── test_core.py
│   │   ├── test_mt5.py
│   │   ├── test_trading.py
│   │   └── test_strategy.py
│   │
│   ├── integration/          # Integration tests
│   │   ├── __init__.py
│   │   ├── test_data_feed.py
│   │   ├── test_execution.py
│   │   └── test_database.py
│   │
│   └── fixtures/             # Test fixtures
│       ├── __init__.py
│       ├── market_data.json
│       └── account_data.json
│
├── logs/                     # Log files directory
│   └── .gitkeep
│
├── data/                     # Data storage
│   ├── historical/           # Historical price data
│   ├── backtest_results/     # Backtest results
│   └── optimization_results/ # Optimization results
│
└── docs/                     # Documentation
    ├── architecture.md       # System architecture
    ├── installation.md       # Installation guide
    ├── configuration.md      # Configuration guide
    ├── api.md                # API documentation
    ├── strategies.md         # Strategy documentation
    └── user_guide.md         # User guide
```

## Documentation

Detailed documentation is available in the `docs/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 