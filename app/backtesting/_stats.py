from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, cast

import numpy as np
import pandas as pd

from ._util import _data_period, _indicator_warmup_nbars

if TYPE_CHECKING:
    from .backtesting import Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(np.int64)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def compute_stats(
        trades: Union[List['Trade'], pd.DataFrame],
        equity: np.ndarray,
        ohlc_data: pd.DataFrame,
        strategy_instance: Strategy | None,
        risk_free_rate: float = 0,
) -> pd.Series:
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur},
        index=index)

    if isinstance(trades, pd.DataFrame):
        trades_df: pd.DataFrame = trades
        commissions = None  # Not shown
    else:
        # Came straight from Backtest.run()
        trades_df = pd.DataFrame({
            'Size': [t.size for t in trades],
            'EntryBar': [t.entry_bar for t in trades],
            'ExitBar': [t.exit_bar for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price for t in trades],
            'SL': [t.sl for t in trades],
            'TP': [t.tp for t in trades],
            'PnL': [t.pl for t in trades],
            'ReturnPct': [t.pl_pct for t in trades],
            'PnLPips': [t.pl_pips for t in trades],  # Add PnL in Pips
            'EntryTime': [t.entry_time for t in trades],
            'ExitTime': [t.exit_time for t in trades],
        })
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
        trades_df['Tag'] = [t.tag for t in trades]

        # Add indicator values
        if len(trades_df) and strategy_instance:
            for ind in strategy_instance._indicators:
                ind = np.atleast_2d(ind)
                for i, values in enumerate(ind):  # multi-d indicators
                    suffix = f'_{i}' if len(ind) > 1 else ''
                    trades_df[f'Entry_{ind.name}{suffix}'] = values[trades_df['EntryBar'].values]
                    trades_df[f'Exit_{ind.name}{suffix}'] = values[trades_df['ExitBar'].values]

        commissions = sum(t._commissions for t in trades)
    del trades

    pl = trades_df['PnL']
    pl_pips = trades_df['PnLPips']  # Get the Pips PnL series
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    # --- Calculate all statistics ---
    stats_dict = {}
    start_time = index[0]
    end_time = index[-1]
    duration = end_time - start_time

    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1
    exposure_time_pct = have_position.mean() * 100

    stats_dict['Exposure Time [%]'] = exposure_time_pct
    stats_dict['Equity Final [$]'] = equity[-1]
    stats_dict['Equity Peak [$]'] = equity.max()
    if commissions:
        stats_dict['Commissions [$]'] = commissions
    stats_dict['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100
    first_trading_bar = _indicator_warmup_nbars(strategy_instance)
    c = ohlc_data.Close.values
    buy_and_hold_return_pct = (c[-1] - c[first_trading_bar]) / c[first_trading_bar] * 100

    gmean_day_return: float = 0
    day_returns = np.array(np.nan)
    annual_trading_days = np.nan
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if is_datetime_index:
        freq_days = cast(pd.Timedelta, _data_period(index)).days
        have_weekends = index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6
        annual_trading_days = (
            52 if freq_days == 7 else
            12 if freq_days == 31 else
            1 if freq_days == 365 else
            (365 if have_weekends else 252))
        freq = {7: 'W', 31: 'ME', 365: 'YE'}.get(freq_days, 'D')
        day_returns = equity_df['Equity'].resample(freq).last().dropna().pct_change()
        gmean_day_return = geometric_mean(day_returns)

    # Calculate trade direction counts and percentages
    n_trades = len(trades_df)
    buy_trades = (trades_df['Size'] > 0).sum()
    sell_trades = (trades_df['Size'] < 0).sum()
    buy_trades_pct = (buy_trades / n_trades * 100) if n_trades else 0
    sell_trades_pct = (sell_trades / n_trades * 100) if n_trades else 0

    # Calculate Trade Durations
    shortest_trade_length = durations.min() if n_trades > 0 else pd.Timedelta(0)
    avg_trade_length = durations.mean() if n_trades > 0 else pd.Timedelta(0)
    longest_trade_length = durations.max() if n_trades > 0 else pd.Timedelta(0)

    # Calculate Avg Trades Per Day
    total_days = duration.days
    avg_trades_per_day = (n_trades / total_days) if total_days > 0 else 0

    # Calculate consecutive wins/losses
    max_consecutive_winners = 0
    max_consecutive_losers = 0
    current_consecutive_winners = 0
    current_consecutive_losers = 0
    for p in pl:
        if p > 0:
            current_consecutive_winners += 1
            current_consecutive_losers = 0
            max_consecutive_winners = max(max_consecutive_winners, current_consecutive_winners)
        elif p < 0:
            current_consecutive_losers += 1
            current_consecutive_winners = 0
            max_consecutive_losers = max(max_consecutive_losers, current_consecutive_losers)
        else: # p == 0 or NaN
            current_consecutive_winners = 0
            current_consecutive_losers = 0

    # Calculate Net Profit
    net_profit = equity[-1] - equity[0]
    net_profit_pips = pl_pips.sum()

    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    stats_dict['Return (Ann.) [%]'] = annualized_return * 100
    volatility_ann = np.sqrt((day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return)**2)**annual_trading_days - (1 + gmean_day_return)**(2 * annual_trading_days)) * 100  # noqa: E501
    stats_dict['Volatility (Ann.) [%]'] = volatility_ann
    if is_datetime_index:
        time_in_years = (duration.days + duration.seconds / 86400) / annual_trading_days
        stats_dict['CAGR [%]'] = ((equity[0] + net_profit) / equity[0])**(1 / time_in_years) - 1 * 100 if time_in_years else np.nan # Recalculated using net_profit

    stats_dict['Sharpe Ratio'] = (stats_dict['Return (Ann.) [%]'] - risk_free_rate * 100) / (volatility_ann or np.nan)  # noqa: E501
    with np.errstate(divide='ignore'):
        sortino_denominator = np.sqrt(np.mean(day_returns.clip(-np.inf, 0)**2)) * np.sqrt(annual_trading_days)
        stats_dict['Sortino Ratio'] = (annualized_return - risk_free_rate) / (sortino_denominator or np.nan)  # noqa: E501
    max_dd = -np.nan_to_num(dd.max())
    stats_dict['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)
    equity_log_returns = np.log(equity[1:] / equity[:-1])
    market_log_returns = np.log(c[1:] / c[:-1])
    beta = np.nan
    if len(equity_log_returns) > 1 and len(market_log_returns) > 1:
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    stats_dict['Alpha [%]'] = stats_dict['Return [%]'] - risk_free_rate * 100 - beta * (buy_and_hold_return_pct - risk_free_rate * 100)  # noqa: E501
    stats_dict['Beta'] = beta
    stats_dict['Max. Drawdown [%]'] = max_dd * 100
    stats_dict['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    stats_dict['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    stats_dict['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
    stats_dict['# Trades'] = n_trades
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    stats_dict['Win Rate [%]'] = win_rate * 100
    stats_dict['Worst Trade [%]'] = returns.min() * 100
    stats_dict['Max. Trade Duration'] = _round_timedelta(longest_trade_length)
    stats_dict['Avg. Trade Duration'] = _round_timedelta(avg_trade_length)
    stats_dict['Net Profit (Pips)'] = net_profit_pips

    # --- Calculate Winning Trade Stats ---
    winners_df = trades_df[trades_df['PnL'] > 0]
    n_winners = len(winners_df)
    win_rate = (pl > 0).mean() if n_trades > 0 else np.nan # Ensure win_rate is calculated here
    gross_profit_usd = winners_df['PnL'].sum()
    gross_profit_pips = winners_df['PnLPips'].sum()
    avg_profit_usd = winners_df['PnL'].mean() # Returns NaN if n_winners is 0
    avg_profit_pips = winners_df['PnLPips'].mean() # Returns NaN if n_winners is 0
    largest_win_usd = winners_df['PnL'].max() # Returns NaN if n_winners is 0
    largest_win_pips = winners_df['PnLPips'].max() # Returns NaN if n_winners is 0
    largest_win_pct = returns[returns > 0].max() # Returns NaN if no positive returns
    largest_win_datetime = pd.NaT
    if n_winners > 0:
        # Find index of largest PnL win to get datetime
        largest_win_idx = winners_df['PnL'].idxmax()
        largest_win_datetime = trades_df.loc[largest_win_idx, 'EntryTime']

    # --- Calculate Losing Trade Stats ---
    losers_df = trades_df[trades_df['PnL'] < 0]
    n_losers = len(losers_df)
    loss_rate = (pl < 0).mean() if n_trades > 0 else np.nan
    gross_loss_usd = losers_df['PnL'].sum()
    gross_loss_pips = losers_df['PnLPips'].sum()
    avg_loss_usd = losers_df['PnL'].mean() # Returns NaN if n_losers is 0
    avg_loss_pips = losers_df['PnLPips'].mean() # Returns NaN if n_losers is 0
    largest_loss_usd = losers_df['PnL'].min() # Returns NaN if n_losers is 0
    largest_loss_pips = losers_df['PnLPips'].min() # Returns NaN if n_losers is 0
    largest_loss_pct = returns[returns < 0].min() # Returns NaN if no negative returns
    largest_loss_datetime = pd.NaT
    if n_losers > 0:
        # Find index of largest PnL loss to get datetime
        largest_loss_idx = losers_df['PnL'].idxmin()
        largest_loss_datetime = trades_df.loc[largest_loss_idx, 'EntryTime']

    # --- Calculate Ratio Stats ---
    profit_factor_pips = gross_profit_pips / abs(gross_loss_pips or np.nan)
    expectancy_usd = pl.mean()
    expectancy_pips = pl_pips.mean()
    expectancy_pct = returns.mean()
    absolute_max_drawdown = (np.maximum.accumulate(equity) - equity).max()
    recovery_factor = net_profit / (absolute_max_drawdown or np.nan)
    avg_wl_ratio_usd = avg_profit_usd / abs(avg_loss_usd or np.nan)
    avg_wl_ratio_pips = avg_profit_pips / abs(avg_loss_pips or np.nan)

    # --- Calculate Remaining General Stats ---
    # These might be moved later or might not fit into specific categories
    if is_datetime_index:
        time_in_years = (duration.days + duration.seconds / 86400) / annual_trading_days
        stats_dict['CAGR [%]'] = ((equity[0] + net_profit) / equity[0])**(1 / time_in_years) - 1 * 100 if time_in_years else np.nan # Recalculated using net_profit

    # --- Arrange statistics ---
    ordered_stats = {}
    ordered_stats['--- Main Stats ---'] = '---'
    ordered_stats['Trading Period'] = f"{start_time} - {end_time}"
    ordered_stats['Duration'] = duration
    ordered_stats['Number of Bars'] = len(index)
    ordered_stats['Exposure Time [%]'] = stats_dict.pop('Exposure Time [%]')
    ordered_stats['Strategy'] = strategy_instance.__class__.__name__ if strategy_instance else 'N/A'
    ordered_stats['Parameters'] = str(strategy_instance.params) if strategy_instance and hasattr(strategy_instance, 'params') else 'N/A'
    ordered_stats['Equity Initial [$]'] = equity[0]
    ordered_stats['Equity Final [$]'] = stats_dict.pop('Equity Final [$]')
    ordered_stats['Equity Peak [$]'] = stats_dict.pop('Equity Peak [$]')
    ordered_stats['Net Profit [$]'] = net_profit
    ordered_stats['Net Profit [Pips]'] = net_profit_pips
    ordered_stats['Return [%]'] = stats_dict.pop('Return [%]')
    ordered_stats['Buy & Hold Return [%]'] = buy_and_hold_return_pct
    ordered_stats['Total Trades'] = n_trades
    ordered_stats['Buy Trades'] = buy_trades
    ordered_stats['Sell Trades'] = sell_trades
    ordered_stats['Buy Trades [%]'] = buy_trades_pct
    ordered_stats['Sell Trades [%]'] = sell_trades_pct
    ordered_stats['Average Trade Length'] = _round_timedelta(avg_trade_length)
    ordered_stats['Longest Trade Length'] = _round_timedelta(longest_trade_length)
    ordered_stats['Shortest Trade Length'] = _round_timedelta(shortest_trade_length)
    ordered_stats['Average Trades Per Day'] = avg_trades_per_day

    ordered_stats['--- Winning Stats ---'] = '---'
    ordered_stats['Winners'] = n_winners
    ordered_stats['Winners [%]'] = stats_dict.pop('Win Rate [%]') # Move Win Rate here
    ordered_stats['Gross Profit [$]'] = gross_profit_usd
    ordered_stats['Gross Profit [Pips]'] = gross_profit_pips
    ordered_stats['Average Profit [$]'] = avg_profit_usd
    ordered_stats['Average Profit [Pips]'] = avg_profit_pips
    ordered_stats['Largest Winning Trade [$]'] = largest_win_usd
    ordered_stats['Largest Winning Trade [Pips]'] = largest_win_pips
    ordered_stats['Largest Winning Trade [%]'] = (largest_win_pct * 100) if pd.notna(largest_win_pct) else np.nan
    ordered_stats['Largest Winning Trade [Date]'] = largest_win_datetime
    ordered_stats['Max Consecutive Winners'] = max_consecutive_winners

    ordered_stats['--- Losing Stats ---'] = '---'
    ordered_stats['Losers'] = n_losers
    ordered_stats['Losers [%]'] = (loss_rate * 100) if pd.notna(loss_rate) else np.nan
    ordered_stats['Gross Loss [$]'] = gross_loss_usd
    ordered_stats['Gross Loss [Pips]'] = gross_loss_pips
    ordered_stats['Average Loss [$]'] = avg_loss_usd
    ordered_stats['Average Loss [Pips]'] = avg_loss_pips
    ordered_stats['Largest Losing Trade [$]'] = largest_loss_usd
    ordered_stats['Largest Losing Trade [Pips]'] = largest_loss_pips
    ordered_stats['Largest Losing Trade [%]'] = (largest_loss_pct * 100) if pd.notna(largest_loss_pct) else np.nan
    ordered_stats['Largest Losing Trade [Date]'] = largest_loss_datetime
    ordered_stats['Max Consecutive Losers'] = max_consecutive_losers

    ordered_stats['--- Ratio Stats ---'] = '---'
    ordered_stats['Profit Factor'] = stats_dict.pop('Profit Factor', gross_profit_usd / abs(gross_loss_usd or np.nan)) # Use calc directly
    ordered_stats['Profit Factor [Pips]'] = profit_factor_pips
    ordered_stats['Expectancy'] = expectancy_usd
    ordered_stats['Expectancy [%]'] = expectancy_pct * 100
    ordered_stats['Expectancy [Pips]'] = expectancy_pips
    ordered_stats['Max. Drawdown [%]'] = stats_dict.pop('Max. Drawdown [%]')
    ordered_stats['Avg. Drawdown [%]'] = stats_dict.pop('Avg. Drawdown [%]')
    ordered_stats['Max. Drawdown Duration'] = stats_dict.pop('Max. Drawdown Duration')
    ordered_stats['Avg. Drawdown Duration'] = stats_dict.pop('Avg. Drawdown Duration')
    ordered_stats['Recovery Factor'] = recovery_factor
    ordered_stats['Average W/L Ratio ($)'] = avg_wl_ratio_usd
    ordered_stats['Average W/L Ratio (Pips)'] = avg_wl_ratio_pips
    ordered_stats['Average R:R'] = avg_wl_ratio_usd # Using W/L Ratio ($) as Avg R:R
    ordered_stats['Return (Ann.) [%]'] = stats_dict.pop('Return (Ann.) [%]')
    ordered_stats['Volatility (Ann.) [%]'] = stats_dict.pop('Volatility (Ann.) [%]')
    ordered_stats['CAGR [%]'] = stats_dict.pop('CAGR [%]', np.nan) # Use pop with default
    ordered_stats['Sharpe Ratio'] = stats_dict.pop('Sharpe Ratio')
    ordered_stats['Sortino Ratio'] = stats_dict.pop('Sortino Ratio')
    ordered_stats['Calmar Ratio'] = stats_dict.pop('Calmar Ratio')
    ordered_stats['Alpha [%]'] = stats_dict.pop('Alpha [%]')
    ordered_stats['Beta'] = stats_dict.pop('Beta')
    sqn_val = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    kelly_val = np.nan
    if pd.notna(win_rate) and avg_wl_ratio_usd is not None and avg_wl_ratio_usd != 0:
        kelly_val = win_rate - (1 - win_rate) / avg_wl_ratio_usd
    ordered_stats['SQN'] = stats_dict.pop('SQN', sqn_val)
    ordered_stats['Kelly Criterion'] = stats_dict.pop('Kelly Criterion', kelly_val)

    # Add remaining stats (should be few, if any)
    ordered_stats.update(stats_dict)

    # Create final Series
    s = pd.Series(ordered_stats, dtype=object)

    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df

    s = _Stats(s)
    return s


class _Stats(pd.Series):
    def __repr__(self):
        with pd.option_context(
            'display.max_colwidth', 20,  # Prevent expansion due to _equity and _trades dfs
            'display.max_rows', len(self),  # Reveal self whole
            'display.precision', 5,  # Enough for my eyes at least
            # 'format.na_rep', '--',  # TODO: Enable once it works
        ):
            return super().__repr__()


def dummy_stats():
    from .backtesting import Trade, _Broker
    index = pd.DatetimeIndex(['2025'])
    data = pd.DataFrame({col: [np.nan] for col in ('Close',)}, index=index)
    trade = Trade(_Broker(data=data, cash=10000, spread=.01, commission=.01, margin=.1,
                          trade_on_close=True, hedging=True, exclusive_orders=False, index=index),
                  1, 1, 0, None)
    trade._replace(exit_price=1, exit_bar=0)
    trade._commissions = np.nan
    return compute_stats([trade], np.r_[[np.nan]], data, None, 0)
