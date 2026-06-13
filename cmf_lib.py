"""
cmf_lib — общая инфраструктура для Task 2 / Task 3.

Реализует РОВНО формулы из description.md:
  * markout PnL мейкера в bps:  pnl_i(tau) = -s_i * (mid(t_i+tau) - p_i)/p_i * 1e4 + 0.5
  * mid = forward-fill BBO mid на момент t_i+tau (последний bbo с ts <= t_i+tau)
  * вес w_i = min(notional_i, 100_000)
  * Score(tau) = PnL_kept(tau) - PnL_all(tau),  где
        PnL_all  = sum w*pnl / sum w
        PnL_kept = sum (1-f)*w*pnl / sum (1-f)*w
        PnL_filt = sum f*w*pnl / sum f*w
  * KeptTurnoverPerDay = sum (1-f)*w / num_days  (>= 500k)

Загрузка через polars (как требует Task 2), тяжёлые операции — numpy/searchsorted,
чтобы держать память под контролем на 400M+ строк.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl

# ---- константы времени / сплита -------------------------------------------------
US = 1_000_000                      # микросекунд в секунде
ONE_DAY_US = 86_400 * US
BYBIT_SHIFT_US = 200_000            # +200мс задержка Bybit -> Binance

def _us(date_str: str) -> int:
    """UTC-полночь даты -> микросекунды от epoch."""
    import datetime as dt
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()) * US

TRAIN_START = _us("2025-12-01")
TRAIN_END   = _us("2026-02-01")     # train: Dec+Jan  [TRAIN_START, TRAIN_END)
VAL_END     = _us("2026-03-01")     # val:   Feb      [TRAIN_END,   VAL_END)
TAUS = (30, 120, 300)
CLIP = 100_000.0                    # клип нотионала для веса
TURNOVER_MIN = 500_000.0

# ---- расположение данных --------------------------------------------------------
_CANDIDATES = [
    Path("data"),
    Path("../liquidation_task/data"),
    Path("/Users/a1234/PycharmProjects/liquidation_task/data"),
]
def data_dir() -> Path:
    for c in _CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError("data/ не найдена: " + ", ".join(map(str, _CANDIDATES)))

_FILE = {
    "trades":      lambda d, s: d / "binance_trades"       / f"perp_{s}.parquet",
    "bbo":         lambda d, s: d / "binance_booktickers"  / f"perp_{s}.parquet",
    "liq_binance": lambda d, s: d / "binance_liquidations" / f"perp_{s}.parquet",
    "liq_bybit":   lambda d, s: d / "bybit_liquidations"   / f"{s}.parquet",
}

# ---- загрузка -------------------------------------------------------------------
def load(kind: str, sym: str, t0: int | None = None, t1: int | None = None,
         every_nth: int | None = None) -> pl.DataFrame:
    """Ленивая загрузка одного источника в окне [t0, t1) (микросекунды).

    every_nth: если задан — взять каждую N-ю строку (детерминированный сэмпл для ETH).
    Дропает ticker и битые строки timestamp<=0.
    """
    d = data_dir()
    lf = pl.scan_parquet(_FILE[kind](d, sym))
    lf = lf.filter(pl.col("timestamp") > 0)
    if t0 is not None:
        lf = lf.filter(pl.col("timestamp") >= t0)
    if t1 is not None:
        lf = lf.filter(pl.col("timestamp") < t1)
    if "ticker" in lf.collect_schema().names():
        lf = lf.drop("ticker")
    if every_nth and every_nth > 1:
        lf = lf.with_row_index("_ri").filter(pl.col("_ri") % every_nth == 0).drop("_ri")
    return lf.collect(engine="streaming")

def num_days(t0: int, t1: int) -> float:
    return (t1 - t0) / ONE_DAY_US

# ---- markout --------------------------------------------------------------------
def trade_arrays(trades: pl.DataFrame):
    """timestamp(int64), s_taker(+1 buy / -1 sell), price(f64), w=min(notional,100k)."""
    ts = trades["timestamp"].to_numpy()
    s = np.where(trades["side"].to_numpy() == "buy", 1.0, -1.0)
    p = trades["price"].to_numpy().astype(np.float64)
    a = trades["amount"].to_numpy().astype(np.float64)
    w = np.minimum(p * a, CLIP)
    return ts, s, p, w

def bbo_arrays(bbo: pl.DataFrame):
    ts = bbo["timestamp"].to_numpy()
    mid = (bbo["bid_price"].to_numpy().astype(np.float64)
           + bbo["ask_price"].to_numpy().astype(np.float64)) / 2.0
    return ts, mid

def markout_pnl(ts_tr, s_tr, p_tr, ts_bbo, mid_bbo, tau_sec: int) -> np.ndarray:
    """pnl_i(tau) в bps; NaN если t_i+tau за пределами BBO. Forward-fill mid."""
    q = ts_tr + tau_sec * US
    idx = np.searchsorted(ts_bbo, q, side="right") - 1
    mid_fwd = np.full(ts_tr.shape, np.nan)
    ok = (idx >= 0) & (q <= int(ts_bbo[-1]))
    mid_fwd[ok] = mid_bbo[idx[ok]]
    return -s_tr * (mid_fwd - p_tr) / p_tr * 1e4 + 0.5

# ---- метрики --------------------------------------------------------------------
# ---- потоковый движок (chunk по дням, ограниченная память) ----------------------
def iter_days(t0: int, t1: int):
    """Генерит окна [d0, d1) по календарным дням внутри [t0, t1)."""
    d = t0
    while d < t1:
        yield d, min(d + ONE_DAY_US, t1)
        d += ONE_DAY_US

def load_day(sym: str, d0: int, d1: int, fwd_pad_us: int = 300 * US,
             liq_pad_us: int = 600 * US, every_nth: int | None = None):
    """Загрузка одного дня: trades(+сэмпл), bbo(+pad вперёд под маркаут),
    liq binance/bybit(+pad назад под lookback)."""
    trades = load("trades", sym, d0, d1, every_nth=every_nth)
    bbo    = load("bbo", sym, d0 - liq_pad_us, d1 + fwd_pad_us)   # назад чуть и вперёд под t+tau
    lbn    = load("liq_binance", sym, d0 - liq_pad_us, d1)
    lby    = load("liq_bybit",   sym, d0 - liq_pad_us, d1)
    return trades, bbo, lbn, lby

# ---- знаковый экспоненциально-затухающий индекс ликвидационного давления ---------
def merged_liq_stream(lbn, lby):
    """Слить Binance + Bybit(+200ms) в один отсортированный поток (ts, signed_notional).
    signed_notional = (+1 buy / -1 sell) * price*amount  — направление давления."""
    def arrs(df, shift):
        ts = df["timestamp"].to_numpy().astype(np.int64) + shift
        sgn = np.where(df["side"].to_numpy() == "buy", 1.0, -1.0)
        notl = df["price"].to_numpy().astype(np.float64) * df["amount"].to_numpy().astype(np.float64)
        return ts, sgn * notl
    a_ts, a_v = arrs(lbn, 0)
    b_ts, b_v = arrs(lby, BYBIT_SHIFT_US)
    ts = np.concatenate([a_ts, b_ts]); v = np.concatenate([a_v, b_v])
    o = np.argsort(ts, kind="mergesort")
    return ts[o], v[o]

def liq_decay_state(L_ts: np.ndarray, V: np.ndarray, lam_us: float) -> np.ndarray:
    """G_k = EWMA знакового ноушнла СРАЗУ ПОСЛЕ k-го liq-события:
        G_k = V_k + G_{k-1} * exp(-(L_k - L_{k-1}) / lam).
    Возвращает массив G длины len(L_ts). Поток ликвидаций мал (сотни тыс.),
    поэтому рекуррентный проход дёшев."""
    n = len(L_ts)
    G = np.empty(n, dtype=np.float64)
    if n == 0:
        return G
    G[0] = V[0]
    dt = np.diff(L_ts).astype(np.float64)
    decay = np.exp(-dt / lam_us)
    g = V[0]
    for k in range(1, n):
        g = V[k] + g * decay[k - 1]
        G[k] = g
    return G

def decay_index(L_ts: np.ndarray, G: np.ndarray, query_ts: np.ndarray, lam_us: float) -> np.ndarray:
    """D_i = знаковое экспоненциально-затухающее ликвидационное давление в момент query_ts:
        D_i = G_{k(i)} * exp(-(t_i - L_{k(i)}) / lam),  k(i) = последнее liq-событие <= t_i."""
    if len(L_ts) == 0:
        return np.zeros(len(query_ts))
    k = np.searchsorted(L_ts, query_ts, side="right") - 1
    out = np.zeros(len(query_ts))
    ok = k >= 0
    out[ok] = G[k[ok]] * np.exp(-(query_ts[ok] - L_ts[k[ok]]).astype(np.float64) / lam_us)
    return out

def microprice_skew(bbo: pl.DataFrame, ts_tr: np.ndarray, s_tr: np.ndarray):
    """Side-aware перекос микроцены к моменту сделки (forward-fill BBO):
        mp_skew = s_taker * (microprice - mid)/mid * 1e4  (bps).
    microprice = (bid*ask_amt + ask*bid_amt)/(bid_amt+ask_amt)."""
    ts_b = bbo["timestamp"].to_numpy()
    bidp = bbo["bid_price"].to_numpy().astype(np.float64); askp = bbo["ask_price"].to_numpy().astype(np.float64)
    bidq = bbo["bid_amount"].to_numpy().astype(np.float64); askq = bbo["ask_amount"].to_numpy().astype(np.float64)
    mid = (bidp + askp) / 2.0
    denom = bidq + askq
    micro = np.where(denom > 0, (bidp * askq + askp * bidq) / np.where(denom > 0, denom, 1.0), mid)
    idx = np.searchsorted(ts_b, ts_tr, side="right") - 1
    out = np.full(len(ts_tr), np.nan)
    ok = idx >= 0
    out[ok] = s_tr[ok] * (micro[idx[ok]] - mid[idx[ok]]) / mid[idx[ok]] * 1e4
    return out

def queue_imbalance(bbo: pl.DataFrame, ts_tr: np.ndarray, s_tr: np.ndarray):
    """Side-aware дисбаланс очереди топа книги (forward-fill BBO):
        qi = s_taker * (bid_amount - ask_amount) / (bid_amount + ask_amount),  в [-1, 1]."""
    ts_b = bbo["timestamp"].to_numpy()
    bq = bbo["bid_amount"].to_numpy().astype(np.float64)
    aq = bbo["ask_amount"].to_numpy().astype(np.float64)
    denom = bq + aq
    imb = np.where(denom > 0, (bq - aq) / np.where(denom > 0, denom, 1.0), 0.0)
    idx = np.searchsorted(ts_b, ts_tr, side="right") - 1
    out = np.full(len(ts_tr), np.nan)
    ok = idx >= 0
    out[ok] = s_tr[ok] * imb[idx[ok]]
    return out

class ScoreAccumulator:
    """Потоковая агрегация числителей/знаменателей Score по чанкам.
    Score(tau) = PnL_kept - PnL_all — отношение сумм, аддитивно по чанкам."""
    def __init__(self, taus=TAUS):
        self.t = {tau: dict(sw=0.0, swp=0.0, kw=0.0, kwp=0.0, fw=0.0, fwp=0.0) for tau in taus}
        self.days = 0.0
    def add(self, tau, f, pnl, w):
        valid = ~np.isnan(pnl)
        pnl, w, f = pnl[valid], w[valid], f[valid].astype(np.float64)
        a = self.t[tau]
        a["sw"]  += w.sum();           a["swp"] += (w * pnl).sum()
        kw = w * (1.0 - f); fw = w * f
        a["kw"]  += kw.sum();          a["kwp"] += (kw * pnl).sum()
        a["fw"]  += fw.sum();          a["fwp"] += (fw * pnl).sum()
    def add_days(self, n):  self.days += n
    def result(self, tau, turnover_scale=1.0):
        a = self.t[tau]
        pnl_all  = a["swp"] / a["sw"] if a["sw"] else float("nan")
        pnl_kept = a["kwp"] / a["kw"] if a["kw"] else float("nan")
        pnl_filt = a["fwp"] / a["fw"] if a["fw"] else float("nan")
        kept_to  = a["kw"] * turnover_scale / self.days if self.days else float("nan")
        return {
            "tau": tau,
            "PnL_all":  round(pnl_all, 3),
            "PnL_kept": round(pnl_kept, 3),
            "PnL_filt": round(pnl_filt, 3),
            "Score":    round(pnl_kept - pnl_all, 3),
            "kept_turnover_per_day": int(round(kept_to)) if kept_to == kept_to else None,
            "turnover_OK": bool(kept_to >= TURNOVER_MIN),
            "pct_w_filt": round(a["fw"] / a["sw"] * 100, 3) if a["sw"] else float("nan"),
        }

def score(f: np.ndarray, pnl: np.ndarray, w: np.ndarray, n_days: float) -> dict:
    """Score(tau) + сопутствующие метрики для уже посчитанного pnl(tau)."""
    f = f.astype(np.float64)
    valid = ~np.isnan(pnl)
    pnl, w, f = pnl[valid], w[valid], f[valid]
    keep_w = w * (1.0 - f)
    filt_w = w * f
    pnl_all  = float((w * pnl).sum() / w.sum())
    pnl_kept = float((keep_w * pnl).sum() / max(keep_w.sum(), 1.0))
    pnl_filt = float((filt_w * pnl).sum() / max(filt_w.sum(), 1.0)) if filt_w.sum() > 0 else float("nan")
    kept_turnover = float(keep_w.sum() / n_days)
    return {
        "PnL_all":   round(pnl_all, 3),
        "PnL_kept":  round(pnl_kept, 3),
        "PnL_filt":  round(pnl_filt, 3),
        "Score":     round(pnl_kept - pnl_all, 3),
        "kept_turnover_per_day": int(round(kept_turnover)),
        "turnover_OK": bool(kept_turnover >= TURNOVER_MIN),
        "pct_w_filt": round(float(filt_w.sum() / w.sum() * 100), 3),
    }
