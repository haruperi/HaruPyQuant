# Trading Module Documentation

## 1. Overview

The Trading Module is the core component of the HaruPyQuant trading system, responsible for all aspects of trade execution, management, and risk control. It is designed with a modular architecture that separates the core trading logic from the specifics of any particular brokerage, allowing for greater flexibility and easier integration with different brokers.

The module's key responsibilities include:
- Creating and managing market and pending orders.
- Tracking open positions and their profit/loss.
- Enforcing risk management rules, such as position sizing.
- Abstracting broker-specific implementations through a unified interface.

## 2. Core Components

The Trading Module is composed of several key classes, each with a distinct responsibility.

### `Trader`
- **File:** `app/trading/trader.py`
- **Purpose:** The central orchestrator of the module. The `Trader` class coordinates all trading activities, using the `Broker` interface to interact with the market and the `RiskManager` to control exposure. It maintains the current state of all open orders and positions.

### `Broker` (Abstract Base Class)
- **File:** `app/trading/broker.py`
- **Purpose:** Defines the standard interface that all broker implementations must adhere to. This abstraction ensures that the core trading logic remains independent of any specific broker's API. It outlines essential methods for sending orders, fetching account data, and managing trades.

### `Order`
- **File:** `app/trading/order.py`
- **Purpose:** A data class that represents a trading order, whether it's a market, limit, or stop order. It contains all relevant information, such as the symbol, direction (buy/sell), volume, and status (pending, filled, etc.).

### `Position`
- **File:** `app/trading/position.py`
- **Purpose:** A data class that represents a currently held asset. It tracks the entry price, volume, direction (long/short), and real-time profit or loss of an open trade.

### `RiskManager`
- **File:** `app/trading/risk_manager.py`
- **Purpose:** A crucial component for managing risk. The `RiskManager` is responsible for calculating appropriate position sizes based on the account balance and a predefined risk percentage. This helps to ensure that no single trade exposes the account to excessive loss.

## 3. Workflow

The following steps outline the typical workflow for using the Trading Module:

1.  **Initialization:**
    -   An instance of a concrete `Broker` class (e.g., `MT5Broker`) is created and connected to the brokerage.
    -   A `RiskManager` is initialized with the account balance and desired risk parameters.
    -   The `Trader` is instantiated with the `Broker` and `RiskManager` instances.

2.  **Creating Orders:**
    -   To enter a trade immediately, use `trader.create_market_order()`.
    -   To place a pending order (e.g., a limit or stop order), use `trader.create_pending_order()`.

3.  **Managing Trades:**
    -   To modify a pending order, use `trader.modify_order()`.
    -   To cancel a pending order, use `trader.cancel_order()`.
    -   To modify the stop-loss or take-profit of an open position, use `trader.modify_position()`.
    -   To close an open position, use `trader.close_position()`.

4.  **State Updates:**
    -   The `trader.update()` method should be called periodically to synchronize the `Trader`'s internal state with the latest information from the broker.

## 4. Live Trading Example

A practical, live demonstration of the Trading Module is available in the `scripts/trade_executor_example.py` file. This script connects to a MetaTrader 5 account and executes a series of common trading operations.

### How to Run the Example

1.  **Configure Your MT5 Credentials:** Ensure your MT5 login, password, server, and terminal path are correctly configured in your project's configuration file.

2.  **Activate Your Virtual Environment:**
    ```bash
    source venv/bin/activate  # On Linux/macOS
    venv\\Scripts\\activate    # On Windows
    ```

3.  **Run the Script:**
    ```bash
    python scripts/trade_executor_example.py
    ```

**⚠️ Warning:** This script executes **real** trades. For safety, always run it on a **demo account** to avoid financial risk.

The script will:
- Connect to your MT5 account.
- Clean up any previous test trades.
- Create, modify, and cancel a pending order.
- Open a market position, modify its stop-loss, and then close it. 