# Data-Quality Memo (auto-generated)

- Membership snapshots: 2712, 1996-01-02 to 2026-06-02
- Unique tickers 2022-01-01..2026-06-30: 596; no-Yahoo-data (delisted/renamed): 45
  - ['ABC', 'ABMD', 'ANSS', 'ANTM', 'ATVI', 'BLL', 'CDAY', 'CERN', 'CMA', 'CTLT', 'CTXS', 'DAY', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DRE', 'FBHS', 'FI', 'FLT', 'FRC', 'GPS', 'HES', 'HOLX', 'IPG', 'JNPR', 'K', 'MMC', 'MRO', 'NLOK', 'NLSN', 'PARA', 'PBCT', 'PEAK', 'PKI', 'PXD', 'RE', 'SEE', 'SIVB', 'TWTR', 'VIAC', 'WBA', 'WLTW', 'WRK', 'XLNX']
- Price matrix: 1126 days x 552 tickers; median coverage 100.0%
- LLM sample eval (30, seed 42, as of 2025-09-01): ['ARE', 'ATO', 'AVY', 'BLDR', 'CI', 'CMG', 'FDS', 'GEN', 'GM', 'GOOG', 'HII', 'HLT', 'IRM', 'IVZ', 'JBL', 'LDOS', 'MO', 'MSI', 'NOW', 'OMC', 'PAYX', 'PH', 'PLTR', 'PTC', 'REG', 'SO', 'SOLV', 'SPG', 'UNH', 'VRTX']
- LLM sample inwindow (as of 2023-07-01): ['APH', 'APTV', 'AVB', 'BG', 'CEG', 'CI', 'FDX', 'GEN', 'GNRC', 'GOOGL', 'HII', 'HLT', 'IRM', 'IVZ', 'JCI', 'LHX', 'MOS', 'MTCH', 'NRG', 'OKE', 'PANW', 'PG', 'PKG', 'PTC', 'RCL', 'SJM', 'SNA', 'UNH', 'VRSN', 'ZBRA']

## Survivorship caveat
Tickers with no Yahoo data are predominantly delisted/renamed names. Their absence
biases the *point-in-time* universe slightly toward survivors; documented as a
limitation (thesis Ch. 3 & 6). CRSP would resolve this in a university setting.

## Market-series cross-check
{
  "base": "Yahoo SPY (adjusted)",
  "IVV": {
    "n_days": 1124,
    "corr": 0.9990573864686273,
    "median_abs_diff_bps": 1.4698553668479608,
    "p95_abs_diff_bps": 5.758742847070616,
    "share_gt_50bps": 0.0017793594306049821
  },
  "VOO": {
    "n_days": 1124,
    "corr": 0.9989046246808275,
    "median_abs_diff_bps": 1.348673021261959,
    "p95_abs_diff_bps": 5.503729085607906,
    "share_gt_50bps": 0.0017793594306049821
  },
  "^GSPC": {
    "n_days": 1124,
    "corr": 0.998440699868745,
    "median_abs_diff_bps": 2.5201984224537677,
    "p95_abs_diff_bps": 8.563389016140944,
    "share_gt_50bps": 0.0026690391459074734
  }
}

## Wikipedia reconciliation
{
  "asof": "2026-06-30 00:00:00",
  "n_primary": 503,
  "n_wikipedia": 503,
  "only_primary": [
    "CAG",
    "CPB",
    "POOL",
    "SATS"
  ],
  "only_wikipedia": [
    "ECHO",
    "FLEX",
    "HONA",
    "MRVL"
  ],
  "overlap": 499
}