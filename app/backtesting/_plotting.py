from __future__ import annotations

import os
import re
import sys
import warnings
from colorsys import hls_to_rgb, rgb_to_hls
from itertools import cycle, combinations
from functools import partial
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from bokeh.colors import RGB
from bokeh.colors.named import (
    lime as BULL_COLOR,
    tomato as BEAR_COLOR
)
from bokeh.events import DocumentReady
from bokeh.plotting import figure as _figure
from bokeh.models import (  # type: ignore
    CrosshairTool,
    CustomJS,
    ColumnDataSource,
    CustomJSTransform,
    Label, NumeralTickFormatter,
    Span,
    HoverTool,
    Range1d,
    DatetimeTickFormatter,
    WheelZoomTool,
    LinearColorMapper,
    DataTable,
    TableColumn,
    NumberFormatter,
)
try:
    from bokeh.models import CustomJSTickFormatter
except ImportError:  # Bokeh < 3.0
    from bokeh.models import FuncTickFormatter as CustomJSTickFormatter  # type: ignore
from bokeh.io import curdoc, output_notebook, output_file, show
from bokeh.io.state import curstate
from bokeh.layouts import gridplot, column
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap, transform

from ._util import _data_period, _as_list, _Indicator, try_

with open(os.path.join(os.path.dirname(__file__), 'autoscale_cb.js'),
          encoding='utf-8') as _f:
    _AUTOSCALE_JS_CALLBACK = _f.read()

IS_JUPYTER_NOTEBOOK = ('JPY_PARENT_PID' in os.environ or
                       'inline' in os.environ.get('MPLBACKEND', ''))

if IS_JUPYTER_NOTEBOOK:
    warnings.warn('Jupyter Notebook detected. '
                  'Setting Bokeh output to notebook. '
                  'This may not work in Jupyter clients without JavaScript '
                  'support, such as old IDEs. '
                  'Reset with `backtesting.set_bokeh_output(notebook=False)`.')
    output_notebook()


def set_bokeh_output(notebook=False):
    """
    Set Bokeh to output either to a file or Jupyter notebook.
    By default, Bokeh outputs to notebook if running from within
    notebook was detected.
    """
    global IS_JUPYTER_NOTEBOOK
    IS_JUPYTER_NOTEBOOK = notebook


def _windos_safe_filename(filename):
    if sys.platform.startswith('win'):
        return re.sub(r'[^a-zA-Z0-9,_-]', '_', filename.replace('=', '-'))
    return filename


def _bokeh_reset(filename=None):
    curstate().reset()
    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        output_file(filename, title=filename)
    elif IS_JUPYTER_NOTEBOOK:
        curstate().output_notebook()
    _add_popcon()


def _add_popcon():
    curdoc().js_on_event(DocumentReady, CustomJS(code='''(function() { var i = document.createElement('iframe'); i.style.display='none';i.width=i.height=1;i.loading='eager';i.src='https://kernc.github.io/backtesting.py/plx.gif.html?utm_source='+location.origin;document.body.appendChild(i);})();'''))  # noqa: E501


def _watermark(fig: _figure):
    fig.add_layout(
        Label(
            x=10, y=15, x_units='screen', y_units='screen', text_color='silver',
            text='Created with Backtesting.py: http://kernc.github.io/backtesting.py',
            text_alpha=.09))


def colorgen():
    yield from cycle(Category10[10])


def lightness(color, lightness=.94):
    rgb = np.array([color.r, color.g, color.b]) / 255
    h, _, s = rgb_to_hls(*rgb)
    rgb = (np.array(hls_to_rgb(h, lightness, s)) * 255).astype(int)
    return RGB(*rgb)


_MAX_CANDLES = 10_000
_INDICATOR_HEIGHT = 50


def _maybe_resample_data(resample_rule, df, indicators, equity_data, trades):
    if isinstance(resample_rule, str):
        freq = resample_rule
    else:
        if resample_rule is False or len(df) <= _MAX_CANDLES:
            return df, indicators, equity_data, trades

        freq_minutes = pd.Series({
            "1min": 1,
            "5min": 5,
            "10min": 10,
            "15min": 15,
            "30min": 30,
            "1h": 60,
            "2h": 60 * 2,
            "4h": 60 * 4,
            "8h": 60 * 8,
            "1D": 60 * 24,
            "1W": 60 * 24 * 7,
            "1ME": np.inf,
        })
        timespan = df.index[-1] - df.index[0]
        require_minutes = (timespan / _MAX_CANDLES).total_seconds() // 60
        freq = freq_minutes.where(freq_minutes >= require_minutes).first_valid_index()
        warnings.warn(f"Data contains too many candlesticks to plot; downsampling to {freq!r}. "
                      "See `Backtest.plot(resample=...)`")

    from .lib import OHLCV_AGG, TRADES_AGG, _EQUITY_AGG
    df = df.resample(freq, label='right').agg(OHLCV_AGG).dropna()

    def try_mean_first(indicator):
        nonlocal freq
        resampled = indicator.df.fillna(np.nan).resample(freq, label='right')
        try:
            return resampled.mean()
        except Exception:
            return resampled.first()

    indicators = [_Indicator(try_mean_first(i).dropna().reindex(df.index).values.T,
                             **dict(i._opts, name=i.name,
                                    # Replace saved index with the resampled one
                                    index=df.index))
                  for i in indicators]
    assert not indicators or indicators[0].df.index.equals(df.index)

    equity_data = equity_data.resample(freq, label='right').agg(_EQUITY_AGG).dropna(how='all')
    assert equity_data.index.equals(df.index)

    def _weighted_returns(s, trades=trades):
        df = trades.loc[s.index]
        return ((df['Size'].abs() * df['ReturnPct']) / df['Size'].abs().sum()).sum()

    def _group_trades(column):
        def f(s, new_index=pd.Index(df.index.astype(np.int64)), bars=trades[column]):
            if s.size:
                # Via int64 because on pandas recently broken datetime
                mean_time = int(bars.loc[s.index].astype(np.int64).mean())
                new_bar_idx = new_index.get_indexer([mean_time], method='nearest')[0]
                return new_bar_idx
        return f

    if len(trades):  # Avoid pandas "resampling on Int64 index" error
        trades = trades.assign(count=1).resample(freq, on='ExitTime', label='right').agg(dict(
            TRADES_AGG,
            ReturnPct=_weighted_returns,
            count='sum',
            EntryBar=_group_trades('EntryTime'),
            ExitBar=_group_trades('ExitTime'),
        )).dropna()

    return df, indicators, equity_data, trades


def plot(*, results: pd.Series,
         df: pd.DataFrame,
         indicators: List[_Indicator],
         filename='', plot_width=None,
         plot_equity=True, plot_return=False, plot_pl=True,
         plot_volume=False, plot_drawdown=True, plot_trades=True,
         plot_trade_table=True,
         smooth_equity=False, relative_equity=True,
         superimpose=True, resample=True,
         reverse_indicators=True,
         show_legend=True, open_browser=True):
    """
    Like much of GUI code everywhere, this is a mess.
    """
    # We need to reset global Bokeh state, otherwise subsequent runs of
    # plot() contain some previous run's cruft data (was noticed when
    # TestPlot.test_file_size() test was failing).
    if not filename and not IS_JUPYTER_NOTEBOOK:
        filename = _windos_safe_filename(str(results._strategy))
    _bokeh_reset(filename)

    COLORS = [BEAR_COLOR, BULL_COLOR]
    BAR_WIDTH = .8

    assert df.index.equals(results['_equity_curve'].index)
    equity_data = results['_equity_curve'].copy(deep=False)
    trades = results['_trades']

    plot_volume = plot_volume and not df.Volume.isnull().all()
    plot_equity = plot_equity and not trades.empty
    plot_return = plot_return and not trades.empty
    plot_pl = plot_pl and not trades.empty
    plot_trades = plot_trades and not trades.empty
    plot_trade_table = plot_trade_table and not trades.empty
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)

    from .lib import OHLCV_AGG
    # ohlc df may contain many columns. We're only interested in, and pass on to Bokeh, these
    df = df[list(OHLCV_AGG.keys())].copy(deep=False)

    # Limit data to max_candles
    if is_datetime_index:
        df, indicators, equity_data, trades = _maybe_resample_data(
            resample, df, indicators, equity_data, trades)

    df.index.name = None  # Provides source name @index
    df['datetime'] = df.index  # Save original, maybe datetime index
    df = df.reset_index(drop=True)
    equity_data = equity_data.reset_index(drop=True)
    index = df.index

    new_bokeh_figure = partial(  # type: ignore[call-arg]
        _figure,
        x_axis_type='linear',
        width=plot_width,
        height=400,
        # TODO: xwheel_pan on horizontal after https://github.com/bokeh/bokeh/issues/14363
        tools="xpan,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save",
        active_drag='xpan',
        active_scroll='xwheel_zoom')

    pad = (index[-1] - index[0]) / 20

    _kwargs = dict(x_range=Range1d(index[0], index[-1],  # type: ignore[call-arg]
                                   min_interval=10,
                                   bounds=(index[0] - pad,
                                           index[-1] + pad))) if index.size > 1 else {}
    fig_ohlc = new_bokeh_figure(**_kwargs)  # type: ignore[arg-type]
    figs_above_ohlc, figs_below_ohlc = [], []

    source = ColumnDataSource(df)
    source.add((df.Close >= df.Open).values.astype(np.uint8).astype(str), 'inc')

    trade_source = ColumnDataSource(dict(
        index=trades['ExitBar'],
        datetime=trades['ExitTime'],
        size=trades['Size'],
        returns_positive=(trades['ReturnPct'] > 0).astype(int).astype(str),
    ))

    inc_cmap = factor_cmap('inc', COLORS, ['0', '1'])
    cmap = factor_cmap('returns_positive', COLORS, ['0', '1'])
    colors_darker = [lightness(BEAR_COLOR, .35),
                     lightness(BULL_COLOR, .35)]
    trades_cmap = factor_cmap('returns_positive', colors_darker, ['0', '1'])

    if is_datetime_index:
        fig_ohlc.xaxis.formatter = CustomJSTickFormatter(  # type: ignore[attr-defined]
            args=dict(axis=fig_ohlc.xaxis[0],
                      formatter=DatetimeTickFormatter(days='%a, %d %b',
                                                      months='%m/%Y'),
                      source=source),
            code='''
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
        ''')

    NBSP = '\N{NBSP}' * 4  # noqa: E999
    ohlc_extreme_values = df[['High', 'Low']].copy(deep=False)
    ohlc_tooltips = [
        ('x, y', NBSP.join(('$index',
                            '$y{0,0.0[0000]}'))),
        ('OHLC', NBSP.join(('@Open{0,0.0[0000]}',
                            '@High{0,0.0[0000]}',
                            '@Low{0,0.0[0000]}',
                            '@Close{0,0.0[0000]}'))),
        ('Volume', '@Volume{0,0}')]

    def new_indicator_figure(**kwargs):
        kwargs.setdefault('height', _INDICATOR_HEIGHT)
        fig = new_bokeh_figure(x_range=fig_ohlc.x_range,
                               active_scroll='xwheel_zoom',
                               active_drag='xpan',
                               **kwargs)
        fig.xaxis.visible = False
        fig.yaxis.minor_tick_line_color = None
        fig.yaxis.ticker.desired_num_ticks = 3
        return fig

    def set_tooltips(fig, tooltips=(), vline=True, renderers=()):
        tooltips = list(tooltips)
        renderers = list(renderers)

        if is_datetime_index:
            formatters = {'@datetime': 'datetime'}
            tooltips = [("Date", "@datetime{%c}")] + tooltips
        else:
            formatters = {}
            tooltips = [("#", "@index")] + tooltips
        fig.add_tools(HoverTool(
            point_policy='follow_mouse',
            renderers=renderers, formatters=formatters,
            tooltips=tooltips, mode='vline' if vline else 'mouse'))

    def _plot_equity_section(is_return=False):
        """Equity section"""
        # Max DD Dur. line
        equity = equity_data['Equity'].copy()
        dd_end = equity_data['DrawdownDuration'].idxmax()
        if np.isnan(dd_end):
            dd_start = dd_end = equity.index[0]
        else:
            dd_start = equity[:dd_end].idxmax()
            # If DD not extending into the future, get exact point of intersection with equity
            if dd_end != equity.index[-1]:
                dd_end = np.interp(equity[dd_start],
                                   (equity[dd_end - 1], equity[dd_end]),
                                   (dd_end - 1, dd_end))

        if smooth_equity:
            interest_points = pd.Index([
                # Beginning and end
                equity.index[0], equity.index[-1],
                # Peak equity and peak DD
                equity.idxmax(), equity_data['DrawdownPct'].idxmax(),
                # Include max dd end points. Otherwise the MaxDD line looks amiss.
                dd_start, int(dd_end), min(int(dd_end + 1), equity.size - 1),
            ])
            select = pd.Index(trades['ExitBar']).union(interest_points)
            select = select.unique().dropna()
            equity = equity.iloc[select].reindex(equity.index)
            equity.interpolate(inplace=True)

        assert equity.index.equals(equity_data.index)

        if relative_equity:
            equity /= equity.iloc[0]
        if is_return:
            equity -= equity.iloc[0]

        yaxis_label = 'Return' if is_return else 'Equity'
        source_key = 'eq_return' if is_return else 'equity'
        source.add(equity, source_key)
        fig = new_indicator_figure(
            y_axis_label=yaxis_label,
            **(dict(height=80) if plot_drawdown else dict(height=100)))

        # High-watermark drawdown dents
        fig.patch('index', 'equity_dd',
                  source=ColumnDataSource(dict(
                      index=np.r_[index, index[::-1]],
                      equity_dd=np.r_[equity, equity.cummax()[::-1]]
                  )),
                  fill_color='#ffffea', line_color='#ffcb66')

        # Equity line
        r = fig.line('index', source_key, source=source, line_width=1.5, line_alpha=1)
        if relative_equity:
            tooltip_format = f'@{source_key}{{+0,0.[000]%}}'
            tick_format = '0,0.[00]%'
            legend_format = '{:,.0f}%'
        else:
            tooltip_format = f'@{source_key}{{$ 0,0}}'
            tick_format = '$ 0.0 a'
            legend_format = '${:,.0f}'
        set_tooltips(fig, [(yaxis_label, tooltip_format)], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)

        # Peaks
        argmax = equity.idxmax()
        fig.scatter(argmax, equity[argmax],
                    legend_label='Peak ({})'.format(
                        legend_format.format(equity[argmax] * (100 if relative_equity else 1))),
                    color='cyan', size=8)
        fig.scatter(index[-1], equity.values[-1],
                    legend_label='Final ({})'.format(
                        legend_format.format(equity.iloc[-1] * (100 if relative_equity else 1))),
                    color='blue', size=8)

        if not plot_drawdown:
            drawdown = equity_data['DrawdownPct']
            argmax = drawdown.idxmax()
            fig.scatter(argmax, equity[argmax],
                        legend_label='Max Drawdown (-{:.1f}%)'.format(100 * drawdown[argmax]),
                        color='red', size=8)
        dd_timedelta_label = df['datetime'].iloc[int(round(dd_end))] - df['datetime'].iloc[dd_start]
        fig.line([dd_start, dd_end], equity.iloc[dd_start],
                 line_color='red', line_width=2,
                 legend_label=f'Max Dd Dur. ({dd_timedelta_label})'
                 .replace(' 00:00:00', '')
                 .replace('(0 days ', '('))

        figs_above_ohlc.append(fig)

    def _plot_drawdown_section():
        """Drawdown section"""
        fig = new_indicator_figure(y_axis_label="Drawdown", height=80)
        drawdown = equity_data['DrawdownPct']
        argmax = drawdown.idxmax()
        source.add(drawdown, 'drawdown')
        r = fig.line('index', 'drawdown', source=source, line_width=1.3)
        fig.scatter(argmax, drawdown[argmax],
                    legend_label='Peak (-{:.1f}%)'.format(100 * drawdown[argmax]),
                    color='red', size=8)
        set_tooltips(fig, [('Drawdown', '@drawdown{-0.[0]%}')], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format="-0.[0]%")
        return fig

    def _plot_pl_section():
        """Profit/Loss markers section"""
        fig = new_indicator_figure(y_axis_label="Profit / Loss", height=80)
        fig.add_layout(Span(location=0, dimension='width', line_color='#666666',
                            line_dash='dashed', level='underlay', line_width=1))
        trade_source.add(trades['ReturnPct'], 'returns')
        size = trades['Size'].abs()
        size = np.interp(size, (size.min(), size.max()), (8, 20))
        trade_source.add(size, 'marker_size')
        if 'count' in trades:
            trade_source.add(trades['count'], 'count')
        trade_source.add(trades[['EntryBar', 'ExitBar']].values.tolist(), 'lines')
        fig.multi_line(xs='lines',
                       ys=transform('returns', CustomJSTransform(v_func='return [...xs].map(i => [0, i]);')),
                       source=trade_source, color='#999', line_width=1)
        r1 = fig.scatter('index', 'returns', source=trade_source, fill_color=cmap,
                         marker='circle', line_color='black', size='marker_size')
        tooltips = [("Size", "@size{0,0}")]
        if 'count' in trades:
            tooltips.append(("Count", "@count{0,0}"))
        set_tooltips(fig, tooltips + [("P/L", "@returns{+0.[000]%}")],
                     vline=False, renderers=[r1])
        fig.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
        return fig

    def _plot_volume_section():
        """Volume section"""
        fig = new_indicator_figure(height=70, y_axis_label="Volume")
        fig.yaxis.ticker.desired_num_ticks = 3
        fig.xaxis.formatter = fig_ohlc.xaxis[0].formatter
        fig.xaxis.visible = True
        fig_ohlc.xaxis.visible = False  # Show only Volume's xaxis
        r = fig.vbar('index', BAR_WIDTH, 'Volume', source=source, color=inc_cmap)
        set_tooltips(fig, [('Volume', '@Volume{0.00 a}')], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format="0 a")
        return fig

    def _plot_superimposed_ohlc():
        """Superimposed, downsampled vbars"""
        time_resolution = pd.DatetimeIndex(df['datetime']).resolution
        resample_rule = (superimpose if isinstance(superimpose, str) else
                         dict(day='ME',
                              hour='D',
                              minute='h',
                              second='min',
                              millisecond='s').get(time_resolution))
        if not resample_rule:
            warnings.warn(
                f"'Can't superimpose OHLC data with rule '{resample_rule}'"
                f"(index datetime resolution: '{time_resolution}'). Skipping.",
                stacklevel=4)
            return

        df2 = (df.assign(_width=1).set_index('datetime')
               .resample(resample_rule, label='left')
               .agg(dict(OHLCV_AGG, _width='count')))

        # Check if resampling was downsampling; error on upsampling
        orig_freq = _data_period(df['datetime'])
        resample_freq = _data_period(df2.index)
        if resample_freq < orig_freq:
            raise ValueError('Invalid value for `superimpose`: Upsampling not supported.')
        if resample_freq == orig_freq:
            warnings.warn('Superimposed OHLC plot matches the original plot. Skipping.',
                          stacklevel=4)
            return

        df2.index = df2['_width'].cumsum().shift(1).fillna(0)
        df2.index += df2['_width'] / 2 - .5
        df2['_width'] -= .1  # Candles don't touch

        df2['inc'] = (df2.Close >= df2.Open).astype(int).astype(str)
        df2.index.name = None
        source2 = ColumnDataSource(df2)
        fig_ohlc.segment('index', 'High', 'index', 'Low', source=source2, color='#bbbbbb')
        colors_lighter = [lightness(BEAR_COLOR, .92),
                          lightness(BULL_COLOR, .92)]
        fig_ohlc.vbar('index', '_width', 'Open', 'Close', source=source2, line_color=None,
                      fill_color=factor_cmap('inc', colors_lighter, ['0', '1']))

    def _plot_ohlc():
        """Main OHLC bars"""
        fig_ohlc.segment('index', 'High', 'index', 'Low', source=source, color="black",
                         legend_label='OHLC')
        r = fig_ohlc.vbar('index', BAR_WIDTH, 'Open', 'Close', source=source,
                          line_color="black", fill_color=inc_cmap, legend_label='OHLC')
        return r

    def _plot_ohlc_trades():
        """Trade entry / exit markers on OHLC plot"""
        trade_source.add(trades[['EntryBar', 'ExitBar']].values.tolist(), 'position_lines_xs')
        trade_source.add(trades[['EntryPrice', 'ExitPrice']].values.tolist(), 'position_lines_ys')
        fig_ohlc.multi_line(xs='position_lines_xs', ys='position_lines_ys',
                            source=trade_source, line_color=trades_cmap,
                            legend_label=f'Trades ({len(trades)})',
                            line_width=8, line_alpha=1, line_dash='dotted')

    def _plot_indicators():
        """Strategy indicators"""

        def _too_many_dims(value):
            assert value.ndim >= 2
            if value.ndim > 2:
                warnings.warn(f"Can't plot indicators with >2D ('{value.name}')",
                              stacklevel=5)
                return True
            return False

        class LegendStr(str):
            # The legend string is such a string that only matches
            # itself if it's the exact same object. This ensures
            # legend items are listed separately even when they have the
            # same string contents. Otherwise, Bokeh would always consider
            # equal strings as one and the same legend item.
            def __eq__(self, other):
                return self is other

        ohlc_colors = colorgen()
        indicator_figs = []

        for i, value in enumerate(indicators):
            value = np.atleast_2d(value)

            # Use .get()! A user might have assigned a Strategy.data-evolved
            # _Array without Strategy.I()
            if not value._opts.get('plot') or _too_many_dims(value):
                continue

            is_overlay = value._opts['overlay']
            is_scatter = value._opts['scatter']
            if is_overlay:
                fig = fig_ohlc
            else:
                fig = new_indicator_figure()
                indicator_figs.append(fig)
            tooltips = []
            colors = value._opts['color']
            colors = colors and cycle(_as_list(colors)) or (
                cycle([next(ohlc_colors)]) if is_overlay else colorgen())

            if isinstance(value.name, str):
                tooltip_label = value.name
                legend_labels = [LegendStr(value.name)] * len(value)
            else:
                tooltip_label = ", ".join(value.name)
                legend_labels = [LegendStr(item) for item in value.name]

            for j, arr in enumerate(value):
                color = next(colors)
                source_name = f'{legend_labels[j]}_{i}_{j}'
                if arr.dtype == bool:
                    arr = arr.astype(int)
                source.add(arr, source_name)
                tooltips.append(f'@{{{source_name}}}{{0,0.0[0000]}}')
                if is_overlay:
                    ohlc_extreme_values[source_name] = arr
                    if is_scatter:
                        fig.circle(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], color=color,
                            line_color='black', fill_alpha=.8,
                            radius=BAR_WIDTH / 2 * .9)
                    else:
                        fig.line(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], line_color=color,
                            line_width=1.3)
                else:
                    if is_scatter:
                        r = fig.circle(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], color=color,
                            radius=BAR_WIDTH / 2 * .6)
                    else:
                        r = fig.line(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], line_color=color,
                            line_width=1.3)
                    # Add dashed centerline just because
                    mean = try_(lambda: float(pd.Series(arr).mean()), default=np.nan)
                    if not np.isnan(mean) and (abs(mean) < .1 or
                                               round(abs(mean), 1) == .5 or
                                               round(abs(mean), -1) in (50, 100, 200)):
                        fig.add_layout(Span(location=float(mean), dimension='width',
                                            line_color='#666666', line_dash='dashed',
                                            level='underlay', line_width=.5))
            if is_overlay:
                ohlc_tooltips.append((tooltip_label, NBSP.join(tooltips)))
            else:
                set_tooltips(fig, [(tooltip_label, NBSP.join(tooltips))], vline=True, renderers=[r])
                # If the sole indicator line on this figure,
                # have the legend only contain text without the glyph
                if len(value) == 1:
                    fig.legend.glyph_width = 0
        return indicator_figs

    def _plot_trades_section():
        """Trades table section showing all trades"""
        if trades.empty:
            return None
            
        # Make a copy of the trades dataframe for display
        display_trades = trades.copy()
        
        # Format datetime columns for better display
        if 'EntryTime' in display_trades.columns and isinstance(display_trades['EntryTime'], pd.Series):
            display_trades['EntryTime'] = display_trades['EntryTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'ExitTime' in display_trades.columns and isinstance(display_trades['ExitTime'], pd.Series):
            display_trades['ExitTime'] = display_trades['ExitTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        # Format return percentage for display
        if 'ReturnPct' in display_trades.columns:
            display_trades['ReturnPct'] = display_trades['ReturnPct'] * 100
        
        # Format Duration for better readability
        if 'Duration' in display_trades.columns:
            # Convert duration to a readable format (handling pandas Timedelta objects)
            def format_duration(td):
                # For pandas Timedelta objects
                if hasattr(td, 'total_seconds'):
                    seconds = td.total_seconds()
                else:
                    # Fallback for numeric values (assuming microseconds)
                    seconds = float(td) / 1_000_000
                
                # Format duration based on length, with more detailed representation
                # Minimum time is 1 minute, so no need to handle seconds
                if seconds < 3600:
                    minutes = int(seconds / 60)
                    return f"{minutes}m"
                elif seconds < 86400:
                    hours = int(seconds / 3600)
                    remaining_minutes = int((seconds % 3600) / 60)
                    if remaining_minutes > 0:
                        return f"{hours}h {remaining_minutes}m"
                    else:
                        return f"{hours}h"
                else:
                    days = int(seconds / 86400)
                    remaining_hours = int((seconds % 86400) / 3600)
                    if remaining_hours > 0:
                        return f"{days}d {remaining_hours}h"
                    else:
                        return f"{days}d"
            
            # Apply formatting to Duration column
            display_trades['Duration_Formatted'] = display_trades['Duration'].apply(format_duration)
            # Save original duration for tooltip/sorting
            display_trades['Duration_Raw'] = display_trades['Duration']
            # Use the formatted duration for display
            display_trades['Duration'] = display_trades['Duration_Formatted']
            
        # Create a ColumnDataSource with the trades data
        trade_table_source = ColumnDataSource(display_trades)
        
        # Define the columns for the DataTable
        columns = []
        
        # Add columns based on what's available in the trades DataFrame
        # Common trade columns
        if 'EntryTime' in display_trades.columns:
            columns.append(TableColumn(field='EntryTime', title='Entry Time'))
        if 'ExitTime' in display_trades.columns:
            columns.append(TableColumn(field='ExitTime', title='Exit Time'))
        if 'EntryPrice' in display_trades.columns:
            columns.append(TableColumn(field='EntryPrice', title='Entry Price', 
                                      formatter=NumberFormatter(format='0.0000')))
        if 'ExitPrice' in display_trades.columns:
            columns.append(TableColumn(field='ExitPrice', title='Exit Price', 
                                      formatter=NumberFormatter(format='0.0000')))
        if 'Size' in display_trades.columns:
            columns.append(TableColumn(field='Size', title='Size',
                                      formatter=NumberFormatter(format='0.00')))
        if 'ReturnPct' in display_trades.columns:
            columns.append(TableColumn(field='ReturnPct', title='Return %',
                                      formatter=NumberFormatter(format='0.00')))
        if 'PnL' in display_trades.columns:
            columns.append(TableColumn(field='PnL', title='Profit/Loss',
                                      formatter=NumberFormatter(format='0.00')))
        if 'Duration' in display_trades.columns:
            columns.append(TableColumn(field='Duration', title='Duration'))
            
        # Create the data table
        data_table = DataTable(
            source=trade_table_source,
            columns=columns,
            width=plot_width if plot_width is not None else 1200,
            height=280,
            index_position=None,
            sortable=True,
            sizing_mode='stretch_width',
        )
        
        return data_table

    # Construct figure ...

    if plot_equity:
        _plot_equity_section()

    if plot_return:
        _plot_equity_section(is_return=True)

    if plot_drawdown:
        figs_above_ohlc.append(_plot_drawdown_section())

    if plot_pl:
        figs_above_ohlc.append(_plot_pl_section())

    if plot_volume:
        fig_volume = _plot_volume_section()
        figs_below_ohlc.append(fig_volume)

    if superimpose and is_datetime_index:
        _plot_superimposed_ohlc()

    ohlc_bars = _plot_ohlc()
    if plot_trades:
        _plot_ohlc_trades()
    indicator_figs = _plot_indicators()
    if reverse_indicators:
        indicator_figs = indicator_figs[::-1]
    figs_below_ohlc.extend(indicator_figs)
    
    # Add trade table if requested
    trade_table = None
    if plot_trade_table:
        trade_table = _plot_trades_section()
    
    _watermark(fig_ohlc)

    set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[ohlc_bars])

    source.add(ohlc_extreme_values.min(1), 'ohlc_low')
    source.add(ohlc_extreme_values.max(1), 'ohlc_high')

    custom_js_args = dict(ohlc_range=fig_ohlc.y_range,
                          source=source)
    if plot_volume:
        custom_js_args.update(volume_range=fig_volume.y_range)

    fig_ohlc.x_range.js_on_change('end', CustomJS(args=custom_js_args,
                                                  code=_AUTOSCALE_JS_CALLBACK))

    figs = figs_above_ohlc + [fig_ohlc] + figs_below_ohlc
    linked_crosshair = CrosshairTool(
        dimensions='both', line_color='lightgrey',
        overlay=(Span(dimension="width", line_dash="dotted", line_width=1),
                 Span(dimension="height", line_dash="dotted", line_width=1)),
    )

    for f in figs:
        if f.legend:
            f.legend.visible = show_legend
            f.legend.location = 'top_left'
            f.legend.border_line_width = 1
            f.legend.border_line_color = '#333333'
            f.legend.padding = 5
            f.legend.spacing = 0
            f.legend.margin = 0
            f.legend.label_text_font_size = '8pt'
            f.legend.click_policy = "hide"
            f.legend.background_fill_alpha = .9
        f.min_border_left = 0
        f.min_border_top = 3
        f.min_border_bottom = 6
        f.min_border_right = 10
        f.outline_line_color = '#666666'

        f.add_tools(linked_crosshair)
        wheelzoom_tool = next(wz for wz in f.tools if isinstance(wz, WheelZoomTool))
        wheelzoom_tool.maintain_focus = False

    kwargs = {}
    if plot_width is None:
        kwargs['sizing_mode'] = 'stretch_width'

    fig = gridplot(
        figs,
        ncols=1,
        toolbar_location='right',
        toolbar_options=dict(logo=None),
        merge_tools=True,
        **kwargs  # type: ignore
    )
    
    # If we have a trade table, create a combined layout
    if trade_table is not None:
        # Create a combined layout with the figure and trade table
        fig = column(fig, trade_table, sizing_mode='stretch_width')
    
    show(fig, browser=None if open_browser else 'none')
    return fig


def plot_heatmaps(heatmap: pd.Series, agg: Union[Callable, str], ncols: int,
                  filename: str = '', plot_width: int = 1200, open_browser: bool = True):
    if not (isinstance(heatmap, pd.Series) and
            isinstance(heatmap.index, pd.MultiIndex)):
        raise ValueError('heatmap must be heatmap Series as returned by '
                         '`Backtest.optimize(..., return_heatmap=True)`')
    if len(heatmap.index.levels) < 2:
        raise ValueError('`plot_heatmap()` requires at least two optimization '
                         'variables to plot')

    _bokeh_reset(filename)

    param_combinations = combinations(heatmap.index.names, 2)
    dfs = [heatmap.groupby(list(dims)).agg(agg).to_frame(name='_Value')
           for dims in param_combinations]
    figs: list[_figure] = []
    cmap = LinearColorMapper(palette='Viridis256',
                             low=min(df.min().min() for df in dfs),
                             high=max(df.max().max() for df in dfs),
                             nan_color='white')
    for df in dfs:
        name1, name2 = df.index.names
        level1 = df.index.levels[0].astype(str).tolist()
        level2 = df.index.levels[1].astype(str).tolist()
        df = df.reset_index()
        df[name1] = df[name1].astype('str')
        df[name2] = df[name2].astype('str')

        fig = _figure(x_range=level1,  # type: ignore[call-arg]
                      y_range=level2,
                      x_axis_label=name1,
                      y_axis_label=name2,
                      width=plot_width // ncols,
                      height=plot_width // ncols,
                      tools='box_zoom,reset,save',
                      tooltips=[(name1, '@' + name1),
                                (name2, '@' + name2),
                                ('Value', '@_Value{0.[000]}')])
        fig.grid.grid_line_color = None        # type: ignore[attr-defined]
        fig.axis.axis_line_color = None        # type: ignore[attr-defined]
        fig.axis.major_tick_line_color = None  # type: ignore[attr-defined]
        fig.axis.major_label_standoff = 0      # type: ignore[attr-defined]

        if not len(figs):
            _watermark(fig)

        fig.rect(x=name1,
                 y=name2,
                 width=1,
                 height=1,
                 source=df,
                 line_color=None,
                 fill_color=dict(field='_Value',
                                 transform=cmap))
        figs.append(fig)

    fig = gridplot(
        figs,  # type: ignore
        ncols=ncols,
        toolbar_options=dict(logo=None),
        toolbar_location='above',
        merge_tools=True,
    )
    show(fig, browser=None if open_browser else 'none')
    return fig
