"""US Session & Gold — backend.

Lancer :  .venv/bin/python server.py   puis ouvrir http://localhost:8501
Sources : Yahoo Finance (prix), FRED (macro US), ForexFactory (calendrier).
"""
from __future__ import annotations

import io
import math
import re
import threading
import time as _t
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).parent
NY, LDN, PARIS = ZoneInfo("America/New_York"), ZoneInfo("Europe/London"), ZoneInfo("Europe/Paris")

# ticker -> (clé, libellé, type)
ASSETS = {
    "^DJI": ("dow", "Dow Jones", "price"),
    "^NDX": ("ndx", "Nasdaq 100", "price"),
    "^GSPC": ("spx", "S&P 500", "price"),
    "GC=F": ("gold", "Or (XAUUSD)", "price"),
    "DX-Y.NYB": ("dxy", "Dollar (DXY)", "price"),
    "^TNX": ("us10y", "Taux 10 ans", "yield"),
    "^VIX": ("vix", "VIX", "price"),
    "CL=F": ("oil", "Pétrole WTI", "price"),
    "EURUSD=X": ("eur", "EUR/USD", "price"),
}
LABEL = {v[0]: v[1] for v in ASSETS.values()}
KIND = {v[0]: v[2] for v in ASSETS.values()}
MAIN = ["dow", "ndx", "spx", "gold", "oil", "eur"]
FUTURES = {"dow": "YM=F", "ndx": "NQ=F", "spx": "ES=F", "gold": "GC=F", "oil": "CL=F", "eur": "6E=F"}
FRED_IDS = ["DFF", "DGS2", "DFII10", "T10YIE", "T10Y2Y", "CPIAUCSL", "CPILFESL", "PCEPILFE", "UNRATE",
            "PAYEMS", "ICSA", "A191RL1Q225SBEA", "RSAFS", "INDPRO", "UMCSENT"]

# ----------------------------------------------------------------------------- cache
_cache: dict = {}
_lock = threading.Lock()


def cached(key, ttl, fn, force=False):
    with _lock:
        hit = _cache.get(key)
        if hit and not force and _t.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (_t.time(), val)
    return val


# ----------------------------------------------------------------------------- chargement
def load_prices() -> pd.DataFrame:
    df = yf.download(list(ASSETS), period="2y", interval="1d", progress=False, auto_adjust=True)["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.rename(columns={k: v[0] for k, v in ASSETS.items()})


def load_intraday() -> pd.DataFrame:
    df = yf.download(list(FUTURES.values()), period="60d", interval="1h", progress=False, auto_adjust=True)
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(NY)
    return df


def _fred_one(sid):
    txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=20).text
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.dropna().set_index("date")["v"]


def load_fred() -> dict:
    def safe(s):
        try:
            return _fred_one(s)
        except Exception:
            return None
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(FRED_IDS, ex.map(safe, FRED_IDS)))
    return {k: v for k, v in res.items() if v is not None and len(v)}


CAL_BACKUP = BASE / "calendar_backup.json"


def load_calendar() -> list:
    """ForexFactory limite le nombre de requêtes : on garde une copie locale de la dernière réponse valide."""
    import json
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = [e for e in r.json() if e.get("country") == "USD"]
        CAL_BACKUP.write_text(json.dumps(data))
        return data
    except Exception:
        if CAL_BACKUP.exists():
            return json.loads(CAL_BACKUP.read_text())
        raise


# ----------------------------------------------------------------------------- utilitaires
def pct(a, b):
    return (a / b - 1) * 100


def ago(s, n):
    return s.iloc[-1 - n] if len(s) > n else s.iloc[0]


def chg(s, n, kind="price"):
    return s.iloc[-1] - ago(s, n) if kind == "yield" else pct(s.iloc[-1], ago(s, n))


def at_or_before(s, days):
    sub = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return sub.iloc[-1] if len(sub) else s.iloc[0]


def yoy(s):
    return (s.pct_change(12) * 100).dropna()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def verdict(total):
    return "porteur" if total >= 3 else "favorable" if total >= 1 else "défavorable" if total <= -3 else "fragile" if total <= -1 else "neutre"


# ----------------------------------------------------------------------------- session
def session_info(now):
    wd, t = now.weekday(), now.time()
    ldn = now.astimezone(LDN)
    london = ldn.weekday() < 5 and time(8) <= ldn.time() < time(16, 30)
    closed = wd == 5 or (wd == 6 and t < time(18)) or (wd == 4 and t >= time(17))
    if closed:
        phase, label = "closed", "Marchés fermés · réouverture dimanche 18:00 NY (minuit Paris)"
    elif time(17) <= t < time(18):
        phase, label = "closed", "Pause quotidienne des futures (17:00–18:00 NY)"
    elif wd < 5 and time(9, 30) <= t < time(16):
        phase, label = "us", "Session US ouverte" + (" · chevauchement avec Londres" if london else "")
    elif wd < 5 and time(8) <= t < time(9, 30):
        phase, label = "pre", "Pré-ouverture US · fenêtre des statistiques de 8:30"
    elif london:
        phase, label = "london", "Session de Londres"
    else:
        phase, label = "asia", "Session asiatique"

    def until(h, m):
        tg = now.replace(hour=h, minute=m, second=0, microsecond=0)
        while tg <= now or tg.weekday() >= 5:
            tg = (tg + timedelta(days=1)).replace(hour=h, minute=m)
        return int((tg - now).total_seconds())

    return {"phase": phase, "label": label, "ny": now.strftime("%H:%M"),
            "paris": now.astimezone(PARIS).strftime("%H:%M"), "day": now.strftime("%A"),
            "minute_of_day": now.hour * 60 + now.minute,
            "countdowns": [{"label": "Statistiques de 8:30", "s": until(8, 30)},
                           {"label": "Ouverture de Wall Street", "s": until(9, 30)},
                           {"label": "Clôture de Wall Street", "s": until(16, 0)}]}


# ----------------------------------------------------------------------------- blocs
def asset_cards(px):
    out = []
    for k in MAIN:
        s = px[k].dropna()
        last = s.iloc[-1]
        prev_year = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
        sma50, sma200 = s.rolling(50).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
        trend = "haussière" if last > sma50 > sma200 else "baissière" if last < sma50 < sma200 else "indécise"
        hi = s.tail(252).max()
        sp = s.tail(120)
        out.append({"key": k, "label": LABEL[k], "last": last, "date": s.index[-1].strftime("%d/%m/%Y"),
                    "d1": chg(s, 1), "w1": chg(s, 5), "m1": chg(s, 21),
                    "ytd": pct(last, prev_year.iloc[-1]) if len(prev_year) else None,
                    "trend": trend, "sma50": sma50, "sma200": sma200, "from_hi": pct(last, hi),
                    "spark": {"t": [d.strftime("%d/%m") for d in sp.index], "v": sp.round(4).tolist()}})
    return out


def drivers(px, fred):
    out = []
    dxy = px["dxy"].dropna()
    c = chg(dxy, 20)
    out.append({"label": "Dollar (DXY)", "value": f"{dxy.iloc[-1]:.2f}", "change": f"{c:+.2f} % sur 20 j", "dir": np.sign(round(c, 1)),
                "note": "Dollar en hausse : pèse sur l'or et les multinationales." if c > 0.5 else
                "Dollar en baisse : soutient l'or et les bénéfices à l'export." if c < -0.5 else "Dollar stable."})
    t10 = px["us10y"].dropna()
    c = chg(t10, 20, "yield")
    out.append({"label": "Taux US 10 ans", "value": f"{t10.iloc[-1]:.2f} %", "change": f"{c:+.2f} pt sur 20 j", "dir": np.sign(round(c, 2)),
                "note": "Hausse des taux : pression sur le Nasdaq et l'or." if c > 0.1 else
                "Détente des taux : favorable au Nasdaq et à l'or." if c < -0.1 else "Taux stables."})
    if "DFII10" in fred:
        r = fred["DFII10"]
        c = r.iloc[-1] - at_or_before(r, 30)
        out.append({"label": "Taux réel 10 ans", "value": f"{r.iloc[-1]:.2f} %", "change": f"{c:+.2f} pt sur 1 mois", "dir": np.sign(round(c, 2)),
                    "note": "Principal moteur de l'or : quand il monte, l'or baisse généralement."})
    if "DFF" in fred and "DGS2" in fred:
        ff, two = fred["DFF"].iloc[-1], fred["DGS2"].iloc[-1]
        gap = two - ff
        exp = "des baisses de taux" if gap < -0.25 else "des hausses de taux" if gap > 0.25 else "un statu quo"
        out.append({"label": "Anticipations Fed", "value": f"Fed {ff:.2f} %", "change": f"taux 2 ans {two:.2f} %", "dir": np.sign(round(gap, 1)),
                    "note": f"Le marché anticipe {exp} sur 12–24 mois (écart 2 ans − Fed : {gap:+.2f} pt)."})
    v = px["vix"].dropna()
    out.append({"label": "VIX (peur)", "value": f"{v.iloc[-1]:.1f}", "change": f"{chg(v, 1):+.1f} % sur 1 j", "dir": np.sign(round(chg(v, 1), 0)),
                "note": "Stress élevé sur les actions." if v.iloc[-1] > 25 else "Nervosité modérée." if v.iloc[-1] > 18
                else "Marché serein." if v.iloc[-1] > 13 else "Grande complaisance."})
    if "T10YIE" in fred:
        b = fred["T10YIE"]
        c = b.iloc[-1] - at_or_before(b, 30)
        out.append({"label": "Inflation anticipée 10 ans", "value": f"{b.iloc[-1]:.2f} %", "change": f"{c:+.2f} pt sur 1 mois", "dir": np.sign(round(c, 2)),
                    "note": "Ce que le marché obligataire attend comme inflation. Liée au pétrole, elle soutient l'or quand elle monte."})
    return out


# Effet de chaque situation sur le pétrole et l'EUR/USD (ajouté à l'explication)
EXTRA = {
    ("infl", "neg"): "Le dollar en profite : l'EUR/USD recule.",
    ("infl", "pos"): "Le dollar perd de son attrait : l'EUR/USD en profite.",
    ("infl", "warn"): "Des taux US élevés plus longtemps soutiennent le dollar face à l'euro.",
    ("job", "neg"): "La demande d'énergie ralentit (pétrole ▼) et le dollar faiblit face à l'euro.",
    ("job", "pos"): "Demande d'énergie soutenue et dollar ferme face à l'euro.",
    ("gdp", "pos"): "Plus de demande de pétrole. L'avance américaine attire les capitaux vers le dollar (EUR/USD ▼).",
    ("gdp", "neg"): "La demande de pétrole recule et le dollar perd son avantage face à l'euro.",
    ("rates", "neg"): "L'écart de taux avec la zone euro se creuse en faveur du dollar (EUR/USD ▼).",
    ("rates", "pos"): "L'écart de taux avec la zone euro se resserre : favorable à l'EUR/USD.",
    ("rates", "warn"): "Le rendement élevé du dollar reste un frein pour l'EUR/USD.",
    ("usd", "neg"): "Le pétrole, coté en dollars, devient plus cher pour le reste du monde. L'EUR/USD baisse mécaniquement : l'euro pèse 57 % du DXY.",
    ("usd", "pos"): "Le pétrole est soutenu et l'EUR/USD monte mécaniquement.",
    ("risk", "neg"): "Le pétrole chute sur les craintes de récession, et le dollar sert de refuge face à l'euro.",
    ("risk", "warn"): "Le pétrole reste fragile face aux craintes sur la demande.",
    ("risk", "pos"): "L'appétit pour le risque profite aussi au pétrole et à l'euro.",
    ("oil", "neg"): "La zone euro, importatrice d'énergie, en souffre : l'EUR/USD recule.",
    ("oil", "pos"): "Bonne nouvelle pour la zone euro, importatrice d'énergie : l'EUR/USD en profite.",
}


def themes(px, fred):
    T = []

    def add(key, name, icon, status, tone, expl, figs, sc):
        extra = EXTRA.get((key, tone))
        if extra:
            expl = f"{expl} {extra}"
        T.append({"key": key, "name": name, "icon": icon, "status": status, "tone": tone, "explain": expl,
                  "figures": figs, "impacts": dict(zip(MAIN, sc))})

    def fig(label, s, unit, digits=1):
        return {"label": label, "value": f"{s.iloc[-1]:.{digits}f}{unit}", "prev": f"{s.iloc[-2]:.{digits}f}{unit}",
                "date": s.index[-1].strftime("%m/%Y")}

    # Inflation
    if "CPILFESL" in fred:
        core = yoy(fred["CPILFESL"])
        tr = core.iloc[-1] - ago(core, 3)
        figs = [fig("CPI core a/a", core, " %")]
        if "CPIAUCSL" in fred:
            figs.insert(0, fig("CPI a/a", yoy(fred["CPIAUCSL"]), " %"))
        if "PCEPILFE" in fred:
            figs.append(fig("PCE core a/a", yoy(fred["PCEPILFE"]), " %"))
        if tr > 0.15:
            add("infl", "Inflation", "🔥", "En accélération", "neg",
                "L'inflation repart : la Fed garde des taux élevés. Le Nasdaq, sensible aux taux, est le plus exposé. Pour l'or, l'effet est partagé entre couverture contre l'inflation et hausse des taux.",
                figs, [-1, -2, -1, 0, 0, -1])
        elif tr < -0.15:
            add("infl", "Inflation", "🔥", "En ralentissement", "pos",
                "L'inflation ralentit : la Fed peut assouplir sa politique. Favorable aux valeurs de croissance (Nasdaq) et à l'or.",
                figs, [1, 2, 1, 1, 0, 1])
        elif core.iloc[-1] > 2.5:
            add("infl", "Inflation", "🔥", f"Stable, au-dessus de la cible ({core.iloc[-1]:.1f} %)", "warn",
                "L'inflation core reste au-dessus de la cible de 2 % : la Fed ne peut pas baisser ses taux rapidement, ce qui freine le Nasdaq.",
                figs, [0, -1, 0, 0, 0, -1])
        else:
            add("infl", "Inflation", "🔥", "Proche de la cible", "neutral",
                "Inflation stable et proche de 2 % : facteur neutre pour les marchés.", figs, [0, 0, 0, 0, 0, 0])

    # Emploi
    if "UNRATE" in fred and "PAYEMS" in fred:
        u = fred["UNRATE"]
        nfp = fred["PAYEMS"].diff().dropna()
        du = u.iloc[-1] - ago(u, 3)
        nfp3 = nfp.tail(3).mean()
        figs = [fig("Chômage", u, " %"), {"label": "Emplois créés (moy. 3 mois)", "value": f"{nfp3:+.0f} k",
                                          "prev": f"{nfp.iloc[-1]:+.0f} k dernier mois", "date": nfp.index[-1].strftime("%m/%Y")}]
        if "ICSA" in fred:
            figs.append({"label": "Inscriptions hebdo", "value": f"{fred['ICSA'].iloc[-1] / 1000:.0f} k",
                         "prev": f"{fred['ICSA'].iloc[-2] / 1000:.0f} k", "date": fred["ICSA"].index[-1].strftime("%d/%m")})
        if du >= 0.2 or nfp3 < 50:
            add("job", "Emploi", "👷", "En dégradation", "neg",
                "Le marché du travail ralentit : risque pour les valeurs cycliques du Dow, mais le marché anticipe des baisses de taux, ce qui soutient fortement l'or.",
                figs, [-1, 0, -1, 2, -1, 1])
        elif du <= 0 and nfp3 > 150:
            add("job", "Emploi", "👷", "Robuste", "pos",
                "Emploi solide : bon pour la consommation et les bénéfices, mais la Fed n'a pas de raison de baisser ses taux, ce qui freine l'or.",
                figs, [1, 0, 1, -1, 1, -1])
        else:
            add("job", "Emploi", "👷", "En normalisation", "neutral",
                "Ni surchauffe ni dégradation nette : facteur neutre.", figs, [0, 0, 0, 0, 0, 0])

    # Croissance
    if "A191RL1Q225SBEA" in fred:
        g = fred["A191RL1Q225SBEA"]
        figs = [{"label": "PIB (trim. annualisé)", "value": f"{g.iloc[-1]:.1f} %", "prev": f"{g.iloc[-2]:.1f} %",
                 "date": f"T{(g.index[-1].month - 1) // 3 + 1} {g.index[-1].year}"}]
        if "RSAFS" in fred:
            figs.append(fig("Ventes au détail a/a", yoy(fred["RSAFS"]), " %"))
        if "INDPRO" in fred:
            figs.append(fig("Production industrielle a/a", yoy(fred["INDPRO"]), " %"))
        if g.iloc[-1] >= 2.5:
            add("gdp", "Croissance", "📈", "Solide", "pos",
                "Croissance forte : les industriels et financières, très présents dans le Dow, en profitent le plus. L'or est moins recherché.",
                figs, [2, 1, 1, -1, 1, -1])
        elif g.iloc[-1] < 1:
            add("gdp", "Croissance", "📈", "Faible", "neg",
                "Croissance molle : les bénéfices sont menacés, l'or attire comme valeur refuge.", figs, [-2, -1, -1, 1, -2, 1])
        else:
            add("gdp", "Croissance", "📈", "Modérée", "neutral", "Croissance proche de son potentiel (~2 %) : facteur neutre.",
                figs, [0, 0, 0, 0, 0, 0])

    # Fed & taux
    if "DFII10" in fred:
        r = fred["DFII10"]
        dr = r.iloc[-1] - at_or_before(r, 30)
        t10 = px["us10y"].dropna()
        figs = [{"label": "Taux réel 10 ans", "value": f"{r.iloc[-1]:.2f} %", "prev": f"{at_or_before(r, 30):.2f} % il y a 1 mois", "date": r.index[-1].strftime("%d/%m")},
                {"label": "Taux 10 ans", "value": f"{t10.iloc[-1]:.2f} %", "prev": f"{ago(t10, 20):.2f} % il y a 20 j", "date": t10.index[-1].strftime("%d/%m")}]
        if "DFF" in fred:
            figs.insert(0, {"label": "Taux Fed", "value": f"{fred['DFF'].iloc[-1]:.2f} %", "prev": "", "date": fred["DFF"].index[-1].strftime("%d/%m")})
        if dr > 0.10:
            add("rates", "Fed & taux", "🏛️", "Taux réels en hausse", "neg",
                "La hausse des taux réels pénalise d'abord les valeurs à forte valorisation (Nasdaq) et l'or, qui ne rapporte aucun intérêt.",
                figs, [-1, -2, -1, -2, 0, -1])
        elif dr < -0.10:
            add("rates", "Fed & taux", "🏛️", "Taux réels en baisse", "pos",
                "La baisse des taux réels profite d'abord au Nasdaq et à l'or.", figs, [1, 2, 1, 2, 0, 1])
        elif r.iloc[-1] > 2:
            add("rates", "Fed & taux", "🏛️", f"Stables mais élevés ({r.iloc[-1]:.1f} %)", "warn",
                "Des taux réels au-dessus de 2 % restent un frein pour la tech et pour l'or.", figs, [0, -1, 0, -1, 0, -1])
        else:
            add("rates", "Fed & taux", "🏛️", "Stables", "neutral", "Taux stables : facteur neutre.", figs, [0, 0, 0, 0, 0, 0])

    # Dollar
    dxy = px["dxy"].dropna()
    c = chg(dxy, 20)
    figs = [{"label": "DXY", "value": f"{dxy.iloc[-1]:.2f}", "prev": f"{c:+.2f} % sur 20 j", "date": dxy.index[-1].strftime("%d/%m")}]
    if c > 1.5:
        add("usd", "Dollar", "💵", "En forte hausse", "neg",
            "Un dollar fort réduit les bénéfices réalisés à l'étranger par les grandes entreprises US et renchérit l'or pour le reste du monde.",
            figs, [-1, -1, -1, -2, -1, -2])
    elif c < -1.5:
        add("usd", "Dollar", "💵", "En forte baisse", "pos",
            "Un dollar faible gonfle les bénéfices à l'export et soutient mécaniquement l'or.", figs, [1, 1, 1, 2, 1, 2])
    else:
        add("usd", "Dollar", "💵", "Sans tendance nette", "neutral", "Dollar stable : facteur neutre.", figs, [0, 0, 0, 0, 0, 0])

    # Risque
    v = px["vix"].dropna().iloc[-1]
    figs = [{"label": "VIX", "value": f"{v:.1f}", "prev": "moyenne historique ≈ 19", "date": px["vix"].dropna().index[-1].strftime("%d/%m")}]
    if "UMCSENT" in fred:
        figs.append(fig("Confiance consommateurs", fred["UMCSENT"], ""))
    if v > 25:
        add("risk", "Sentiment de marché", "⚠️", "Stress élevé", "neg",
            "Les investisseurs vendent les actions et cherchent des valeurs refuges comme l'or.", figs, [-2, -2, -2, 1, -2, -1])
    elif v > 18:
        add("risk", "Sentiment de marché", "⚠️", "Nervosité", "warn",
            "Marché nerveux : prudence sur les indices, légère demande de valeur refuge.", figs, [-1, -1, -1, 1, -1, 0])
    elif v < 14:
        add("risk", "Sentiment de marché", "⚠️", "Euphorie", "pos",
            "Fort appétit pour le risque : bon pour les indices, peu de demande pour l'or.", figs, [1, 1, 1, -1, 1, 1])
    else:
        add("risk", "Sentiment de marché", "⚠️", "Calme", "neutral", "Sentiment équilibré : facteur neutre.", figs, [0, 0, 0, 0, 0, 0])

    # Énergie
    o = px["oil"].dropna()
    c = chg(o, 20)
    figs = [{"label": "WTI", "value": f"{o.iloc[-1]:.2f} $", "prev": f"{c:+.1f} % sur 20 j", "date": o.index[-1].strftime("%d/%m")}]
    if c > 10:
        add("oil", "Énergie", "🛢️", "Pétrole en flambée", "neg",
            "Le pétrole alimente l'inflation et pèse sur la consommation. Le Dow, avec ses pétrolières, résiste mieux que le Nasdaq.",
            figs, [0, -1, -1, 1, 2, -1])
    elif c < -10:
        add("oil", "Énergie", "🛢️", "Pétrole en chute", "pos",
            "La baisse de l'énergie réduit l'inflation et soutient la consommation.", figs, [0, 1, 1, -1, -2, 1])
    else:
        add("oil", "Énergie", "🛢️", "Stable", "neutral", "Prix de l'énergie sans effet marqué.", figs, [0, 0, 0, 0, 0, 0])

    totals = {k: sum(t["impacts"][k] for t in T) for k in MAIN}
    return T, {k: {"score": v, "verdict": verdict(v)} for k, v in totals.items()}


def correlations(px):
    al = px.dropna(subset=["spx"]).ffill()
    rets = pd.DataFrame({k: (al[k].diff() if KIND[k] == "yield" else np.log(al[k]).diff()) for k in al.columns}).dropna()
    order = ["dow", "ndx", "spx", "gold", "oil", "eur", "dxy", "us10y", "vix"]
    W = (20, 60, 120)
    mats = {w: rets.tail(w)[order].corr().values.tolist() for w in W}
    gold_vs = {k: {w: rets.tail(w)["gold"].corr(rets.tail(w)[k]) for w in W} for k in order if k != "gold"}
    roll_keys = ["spx", "ndx", "dow", "dxy", "us10y", "oil", "eur"]
    rolling = {w: {k: rets["gold"].rolling(w).corr(rets[k]).tail(250).tolist() for k in roll_keys} for w in W}
    dates = rets.tail(250).index.strftime("%d/%m/%y").tolist()

    c = lambda a, b, w=60: rets.tail(w)[a].corr(rets.tail(w)[b])
    ins = []
    gs = c("gold", "spx")
    if gs > 0.3:
        ins.append(("warn", "L'or monte et baisse avec les actions",
                    f"Corrélation or / S&P 500 de {gs:+.2f}. Les deux actifs suivent la liquidité : l'or ne protège pas d'une baisse des indices en ce moment."))
    elif gs < -0.3:
        ins.append(("pos", "L'or joue son rôle de valeur refuge",
                    f"Corrélation or / S&P 500 de {gs:+.2f}. Quand les indices baissent, l'or tend à monter."))
    else:
        ins.append(("neutral", "L'or évolue indépendamment des actions",
                    f"Corrélation or / S&P 500 de {gs:+.2f} : les deux marchés suivent des moteurs différents."))
    gd = c("gold", "dxy")
    if gd > -0.15:
        ins.append(("warn", "Le lien or / dollar est rompu",
                    f"Corrélation de {gd:+.2f} alors qu'elle est normalement négative. Un autre moteur domine : achats des banques centrales, géopolitique ou demande d'Asie."))
    else:
        ins.append(("pos", "Relation or / dollar classique",
                    f"Corrélation de {gd:+.2f} : un dollar qui monte fait baisser l'or. Surveillez le DXY pour anticiper l'or."))
    nr = c("ndx", "us10y")
    if nr < -0.25:
        ins.append(("neg", "Le Nasdaq est piloté par les taux",
                    f"Corrélation Nasdaq / taux 10 ans de {nr:+.2f} : chaque hausse des taux fait baisser la tech. Les chiffres d'inflation sont décisifs."))
    elif nr > 0.25:
        ins.append(("pos", "Le Nasdaq suit la croissance",
                    f"Corrélation Nasdaq / taux 10 ans de {nr:+.2f} : taux et actions montent ensemble, signe de bonnes nouvelles économiques."))
    rot = chg(px["ndx"].dropna(), 21) - chg(px["dow"].dropna(), 21)
    if abs(rot) > 2:
        ins.append(("neutral", "Rotation sectorielle" + (" vers la tech" if rot > 0 else " vers la valeur"),
                    f"Sur 1 mois, le Nasdaq {'surperforme' if rot > 0 else 'sous-performe'} le Dow de {abs(rot):.1f} points. "
                    + ("Les investisseurs privilégient la croissance." if rot > 0 else "Les investisseurs privilégient les valeurs industrielles et financières.")))
    ge = c("gold", "eur")
    if ge > 0.4:
        ins.append(("neutral", "Or et EUR/USD portés par le même moteur",
                    f"Corrélation de {ge:+.2f} : les deux bougent avec la faiblesse ou la force du dollar. Un mouvement de l'EUR/USD annonce souvent celui de l'or."))
    ot = c("oil", "us10y")
    if ot > 0.3:
        ins.append(("warn", "Le pétrole pousse les taux",
                    f"Corrélation pétrole / taux 10 ans de {ot:+.2f} : le marché craint que l'énergie relance l'inflation. Une hausse du baril pèse alors sur le Nasdaq."))
    os_ = c("oil", "spx")
    if os_ > 0.35:
        ins.append(("neutral", "Pétrole et actions suivent la croissance",
                    f"Corrélation pétrole / S&P 500 de {os_:+.2f} : les deux reflètent les anticipations de demande mondiale."))
    elif os_ < -0.25:
        ins.append(("warn", "Le pétrole pèse sur les actions",
                    f"Corrélation pétrole / S&P 500 de {os_:+.2f} : la hausse du baril est vue comme une taxe sur la consommation."))
    for k in ("spx", "dxy"):
        d = c("gold", k, 20) - c("gold", k, 120)
        if abs(d) > 0.4:
            ins.append(("warn", f"Changement de régime or / {LABEL[k]}",
                        f"La corrélation sur 20 jours ({c('gold', k, 20):+.2f}) s'écarte fortement de celle sur 120 jours ({c('gold', k, 120):+.2f})."))
    return {"order": order, "labels": [LABEL[k] for k in order], "matrix": mats, "gold_vs": gold_vs,
            "rolling": rolling, "dates": dates, "insights": [{"tone": a, "title": b, "text": t} for a, b, t in ins]}


# ----------------------------------------------------------------------------- calendrier
FR = {
    "Core Retail Sales": "Ventes au détail core", "Retail Sales": "Ventes au détail", "Core CPI": "Inflation CPI core",
    "CPI": "Inflation CPI", "Core PPI": "Prix à la production core", "PPI": "Prix à la production (PPI)",
    "Core PCE Price Index": "Inflation PCE core", "ADP Non-Farm Employment Change": "Emplois privés ADP",
    "Non-Farm Employment Change": "Créations d'emplois (NFP)", "Unemployment Rate": "Taux de chômage",
    "Unemployment Claims": "Inscriptions au chômage", "Average Hourly Earnings": "Salaire horaire moyen",
    "JOLTS Job Openings": "Offres d'emploi JOLTS", "FOMC Meeting Minutes": "Minutes du FOMC",
    "FOMC Statement": "Communiqué du FOMC", "Federal Funds Rate": "Décision de taux de la Fed",
    "FOMC Press Conference": "Conférence de presse de la Fed", "Advance GDP": "PIB (1re estimation)",
    "Prelim GDP": "PIB (2e estimation)", "Final GDP": "PIB (définitif)", "ISM Manufacturing PMI": "ISM manufacturier",
    "ISM Services PMI": "ISM services", "Flash Manufacturing PMI": "PMI manufacturier (flash)",
    "Flash Services PMI": "PMI services (flash)", "Empire State Manufacturing Index": "Indice Empire State",
    "Philly Fed Manufacturing Index": "Indice Philly Fed", "Building Permits": "Permis de construire",
    "Housing Starts": "Mises en chantier", "Existing Home Sales": "Ventes de logements anciens",
    "New Home Sales": "Ventes de logements neufs", "Pending Home Sales": "Promesses de ventes immobilières",
    "Industrial Production": "Production industrielle", "Crude Oil Inventories": "Stocks de pétrole brut",
    "Natural Gas Storage": "Stocks de gaz naturel", "Prelim UoM Consumer Sentiment": "Confiance des consommateurs (UMich)",
    "Revised UoM Consumer Sentiment": "Confiance des consommateurs (UMich, révisé)",
    "CB Consumer Confidence": "Confiance des consommateurs (Conference Board)", "Durable Goods Orders": "Commandes de biens durables",
    "Core Durable Goods Orders": "Commandes de biens durables core", "Trade Balance": "Balance commerciale",
    "Import Prices": "Prix à l'importation", "Business Inventories": "Stocks des entreprises",
    "Capacity Utilization Rate": "Taux d'utilisation des capacités", "Prelim UoM Inflation Expectations": "Anticipations d'inflation (UMich)", "Revised UoM Inflation Expectations": "Anticipations d'inflation (UMich, révisé)",
    "10-y Bond Auction": "Adjudication à 10 ans", "30-y Bond Auction": "Adjudication à 30 ans",
}
_FR_KEYS = sorted(FR, key=len, reverse=True)

# (mots-clés, condition, effets : usd, gold, dow, ndx, spx) ; 0 = réaction mitigée
EFFECTS = [
    (("crude oil inventories",), "si les stocks dépassent les prévisions", dict(usd=0, gold=0, dow=0, ndx=0, spx=0, oil=-1)),
    (("unemployment claims",), "si supérieur aux prévisions", dict(usd=-1, gold=1, dow=0, ndx=0, spx=0, oil=0)),
    (("unemployment rate",), "si supérieur aux prévisions", dict(usd=-1, gold=1, dow=-1, ndx=0, spx=-1, oil=-1)),
    (("cpi", "ppi", "pce", "price index", "inflation expectations", "import prices", "average hourly"),
     "si supérieur aux prévisions", dict(usd=1, gold=-1, dow=-1, ndx=-1, spx=-1, oil=0)),
    (("non-farm", "adp", "employment change", "jolts"), "si supérieur aux prévisions", dict(usd=1, gold=-1, dow=1, ndx=-1, spx=0, oil=1)),
    (("gdp", "retail sales", "ism", "pmi", "durable", "empire state", "philly", "industrial", "housing", "home sales",
      "building permits", "sentiment", "confidence", "capacity"), "si supérieur aux prévisions", dict(usd=1, gold=-1, dow=1, ndx=0, spx=1, oil=1)),
    (("federal funds", "fomc statement", "press conference"), "si ton restrictif (hawkish)", dict(usd=1, gold=-1, dow=-1, ndx=-1, spx=-1, oil=-1)),
    (("fomc", "fed chair", "speaks", "testifies", "minutes"), "si ton restrictif (hawkish)", dict(usd=1, gold=-1, dow=-1, ndx=-1, spx=-1, oil=-1)),
    (("auction",), "si la demande est faible", dict(usd=0, gold=-1, dow=-1, ndx=-1, spx=-1, oil=0)),
]


def fr_title(t):
    m = re.match(r"(FOMC Member|Fed Chair(?:man)?|Fed Vice Chair(?:man)?) (.+?) (Speaks|Testifies)", t)
    if m:
        who = "du président de la Fed" if m.group(1).startswith("Fed Chair") else "d'un membre du FOMC"
        return f"Discours {who} ({m.group(2)})"
    for k in _FR_KEYS:
        if t.startswith(k):
            return FR[k] + t[len(k):]
    return t


def calendar(raw, now):
    days = {}
    for e in raw:
        d = pd.Timestamp(e["date"]).tz_convert(NY)
        tl = e["title"].lower()
        eff = next(((cond, {**ef, "eur": -ef["usd"]}) for keys, cond, ef in EFFECTS if any(k in tl for k in keys)), (None, None))
        days.setdefault(d.strftime("%Y-%m-%d"), []).append({
            "time_ny": d.strftime("%H:%M"), "time_paris": d.tz_convert(PARIS).strftime("%H:%M"),
            "title": fr_title(e["title"]), "original": e["title"], "impact": e.get("impact"),
            "forecast": e.get("forecast") or "", "previous": e.get("previous") or "",
            "past": bool(d < now), "condition": eff[0], "effects": eff[1]})
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    out = []
    for k in sorted(days):
        d = pd.Timestamp(k)
        out.append({"date": k, "label": f"{jours[d.weekday()].capitalize()} {d.day:02d}/{d.month:02d}",
                    "today": k == now.strftime("%Y-%m-%d"), "events": days[k]})
    return out


# ----------------------------------------------------------------------------- session US
def session_profile(intr):
    prof, stats, price = {}, {}, {}

    def sess(h):
        return "Asie" if h >= 18 or h < 3 else "Londres" if h < 8 else "New York" if h < 17 else None

    for k, t in FUTURES.items():
        d = pd.DataFrame({"h": intr["High"][t], "l": intr["Low"][t], "c": intr["Close"][t]}).dropna()
        d["r"] = (d["h"] - d["l"]) / d["c"] * 100
        d["ret"] = d["c"].pct_change() * 100
        d = d.dropna()
        d["hour"] = d.index.hour
        prof[k] = d.groupby("hour")["r"].mean().reindex(range(24)).tolist()
        d["session"] = d["hour"].map(sess)
        d["tday"] = (d.index + pd.Timedelta(hours=6)).date
        g = d.dropna(subset=["session"]).groupby(["tday", "session"])
        ret = g["ret"].sum().unstack()
        rng = ((g["h"].max() - g["l"].min()) / g["c"].mean() * 100).unstack()
        price[k] = d["c"].iloc[-1]
        stats[k] = [{"session": s, "perf": ret[s].mean(), "up": (ret[s] > 0).mean() * 100, "range": rng[s].mean()}
                    for s in ("Asie", "Londres", "New York") if s in ret]
    return {"hourly": prof, "sessions": stats, "price": price}


# ----------------------------------------------------------------------------- API
def build(force=False):
    now = datetime.now(NY)
    px = cached("px", 300, load_prices, force)
    fred = cached("fred", 3600, load_fred, force)
    try:
        cal = cached("cal", 1800, load_calendar)  # pas de forçage : la source bloque les requêtes trop fréquentes
    except Exception:
        cal = []
    try:
        intr = cached("intr", 600, load_intraday, force)
        prof = session_profile(intr)
    except Exception:
        prof = None
    th, totals = themes(px, fred)
    cards = asset_cards(px)
    for c in cards:
        c["macro"] = totals[c["key"]]
    return clean({"updated": now.strftime("%d/%m/%Y %H:%M"), "session": session_info(now), "assets": cards,
                  "drivers": drivers(px, fred), "themes": th, "totals": totals, "corr": correlations(px),
                  "calendar": calendar(cal, now), "profile": prof, "labels": {k: LABEL[k] for k in MAIN}})


app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.get("/api/dashboard")
def api():
    return jsonify(build(force=request.args.get("force") == "1"))


if __name__ == "__main__":
    import os
    # 0.0.0.0 : accessible depuis les appareils du même réseau Wi-Fi (téléphone). HOST=127.0.0.1 pour limiter au Mac.
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8501)), debug=False, threaded=True)
