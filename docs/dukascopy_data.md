# 🏦 dukascopy-python

Download and stream historical price data for variety of financial instruments (e.g. Forex, Commodities and Indices) from Dukascopy Bank SA. , including support for tick-level and aggregated intervals.

---

## 📦 Installation

```bash
pip install dukascopy-python
```

---

## 🛠️ Usage

### Importing

```python
from datetime import datetime, timedelta
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD
```

---

## 🧠 Key Concepts

Both `fetch` and `live_fetch` share similar parameters:

| Parameter      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `start`        | `datetime`, required. The start time of the data.                           |
| `end`          | `datetime`, optional. If `None`, fetches data up to "now".                  |
| `instrument`   | e.g., `INSTRUMENT_FX_MAJORS_GBP_USD`.                                       |
| `offer_side`   | `OFFER_SIDE_BID` or `OFFER_SIDE_ASK`.                                       |
| `max_retries`  | Optional. If `None`, keeps retrying on failure.                             |
| `debug`        | Optional. If `True`, prints debug logs.                                     |

### 🧊 `fetch()` only:

| Parameter    | Description                  |
|--------------|------------------------------|
| `interval`   | e.g., `INTERVAL_HOUR_1`      |

### 🔥 `live_fetch()` only:

| Parameter         | Description                                 |
|-------------------|---------------------------------------------|
| `interval_value`  | e.g., `1`                                    |
| `time_unit`       | e.g., `dukascopy_python.TIME_UNIT_HOUR`           |

---

## 📝 Notes

- **fetch**: Fetches static historical data. Returns **one** `DataFrame`.
- **live_fetch**: Continuously fetches live updates. Returns a **generator** that yields the **same `DataFrame`** with updated data.

When using intervals not based on ticks eg: `1HOUR`, `fetch()` will return delayed data. For up-to-date values, use `live_fetch()` which fetches tick data under the hood and reshapes it based on the `interval_value` and `time_unit`.

---

## 📊 DataFrame Columns

### When interval/time_unit is based on tick:
ie:

`interval = INTERVAL_TICK`

or

`interval_value = 1`
`time_units = TIME_UNIT_TICK`

| Column      | Description            |
|-------------|------------------------|
| `timestamp` | UTC datetime, Dataframe Index |
| `bidPrice`  | Bid price              |
| `askPrice`  | Ask price              |
| `bidVolume` | Bid volume             |
| `askVolume` | Ask volume             |

### When interval/time_unit is NOT based on tick
eg: 5 minutes OHLC candle data

`interval_value = 5`
`time_units = TIME_UNIT_MIN`

| Column      | Description            |
|-------------|------------------------|
| `timestamp` | UTC datetime, Dataframe Index |
| `open`      | Opening price          |
| `high`      | Highest price          |
| `low`       | Lowest price           |
| `close`     | Closing price          |
| `volume`    | Volume (in units)      |

---

## 💾 Saving Results

Use built-in `pandas` methods to export:

```python
df.to_csv("data.csv")
df.to_excel("data.xlsx")
df.to_json("data.json")
```

---

## 🚀 Examples

### Example 1: Fetch Historical Data

```python
start = datetime(2025, 1, 1)
end = datetime(2025, 2, 1)
instrument = INSTRUMENT_FX_MAJORS_GBP_USD
interval = dukascopy_python.INTERVAL_HOUR_1
offer_side = dukascopy_python.OFFER_SIDE_BID

df = dukascopy_python.fetch(
    instrument,
    interval,
    offer_side,
    start,
    end,
)

df.to_json("output.json")
```

---

### Example 2: Live Fetch with End Time

```python
now = datetime.now()
start = datetime(now.year, now.month, now.day)
end = start + timedelta(hours=24)
instrument = INSTRUMENT_FX_MAJORS_GBP_USD
offer_side = dukascopy_python.OFFER_SIDE_BID

iterator = dukascopy_python.live_fetch(
    instrument,
    1,
    dukascopy_python.TIME_UNIT_HOUR,
    offer_side,
    start,
    end,
)

for df in iterator:
    pass

df.to_csv("output.csv")
```

---

### Example 3: Live Fetch Indefinitely (End = None)

```python
now = datetime.now()
start = datetime(now.year, now.month, now.day)
end = None
instrument = INSTRUMENT_FX_MAJORS_GBP_USD
offer_side = dukascopy_python.OFFER_SIDE_BID

df_iterator = dukascopy_python.live_fetch(
    instrument,
    1,
    dukascopy_python.TIME_UNIT_HOUR,
    offer_side,
    start,
    end,
)

for df in df_iterator:
    # Do something with latest data
    pass
```

---

## 📄 License

MIT

---

## 👋 Contributing

Pull requests and suggestions are highly welcome!

