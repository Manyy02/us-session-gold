"""US Session & Gold — backend.

Lancer :  .venv/bin/python server.py   puis ouvrir http://localhost:8501
Sources : Yahoo Finance (prix), FRED (macro US), ForexFactory (calendrier).
"""
from __future__ import annotations

import html as html_lib
import io
import math
import re
import sys
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
FUTURES = {"dow": "YM=F", "ndx": "NQ=F", "spx": "ES=F", "gold": "GC=F", "oil": "CL=F", "eur": "6E=F"}  # alertes « dernière heure »
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
    """Matrice de corrélation des variations journalières sur les 60 dernières séances."""
    al = px.dropna(subset=["spx"]).ffill()
    order = ["dow", "ndx", "spx", "gold", "oil", "eur", "dxy"]
    rets = pd.DataFrame({k: np.log(al[k]).diff() for k in order}).dropna().tail(60)
    return {"order": order, "matrix": rets.corr().values.tolist(), "n": len(rets)}


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



# ----------------------------------------------------------------------------- bilan économique
def bilan(themes, totals):
    st = {t["key"]: t for t in themes}
    parts = []
    for k, lbl in (("gdp", "croissance"), ("job", "emploi"), ("infl", "inflation"), ("rates", "taux"), ("usd", "dollar"), ("risk", "sentiment")):
        if k in st:
            parts.append(f"{lbl} : {st[k]['status'].lower()}")
    pos = sum(t["tone"] == "pos" for t in themes)
    neg = sum(t["tone"] == "neg" for t in themes) + 0.5 * sum(t["tone"] == "warn" for t in themes)
    if pos >= neg + 1.5:
        verdict, tone = "Économie américaine porteuse", "pos"
    elif neg >= pos + 1.5:
        verdict, tone = "Économie américaine sous tension", "neg"
    else:
        verdict, tone = "Économie américaine mitigée", "warn"
    fav = [LABEL[k] for k in MAIN if totals[k]["score"] >= 1]
    dfv = [LABEL[k] for k in MAIN if totals[k]["score"] <= -1]
    return {"verdict": verdict, "tone": tone, "summary": " · ".join(parts),
            "favours": fav, "hurts": dfv,
            "pillars": [{"name": t["name"], "icon": t["icon"], "status": t["status"], "tone": t["tone"],
                         "figure": t["figures"][0]["label"] + " " + t["figures"][0]["value"] if t["figures"] else ""} for t in themes]}


# ----------------------------------------------------------------------------- actualités
NEWS_FEEDS = [
    ("Google Actualités", "https://news.google.com/rss/search?q=(Wall%20Street%20OR%20Nasdaq%20OR%20%22Dow%20Jones%22%20OR%20%22S%26P%20500%22%20OR%20Fed%20OR%20%22prix%20de%20l%27or%22%20OR%20p%C3%A9trole%20OR%20%22euro%20dollar%22%20OR%20%22march%C3%A9s%20financiers%22)%20when:2d&hl=fr&gl=FR&ceid=FR:fr"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
    ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC,%5ENDX,%5EDJI,GC%3DF,CL%3DF,EURUSD%3DX&region=US&lang=en-US"),
]
# thème -> (mots-clés en minuscules, actifs concernés)
NEWS_TAGS = {
    "Fed / taux": (["fed", "fomc", "powell", "warsh", "taux directeur", "banque centrale américaine", "rate cut", "rate hike", "baisse des taux", "hausse des taux", "treasury yield", "rendement obligataire"], ["dow", "ndx", "spx", "gold", "eur"]),
    "Tech / IA": (["nvidia", "data center", "data centers", "centres de données", "centre de données", "semi-conducteur", "semiconductor", "chip", "puces", "openai", "anthropic", "magnificent", "big tech", "géants de la tech", "microsoft", "alphabet", "apple", "meta", "amazon", "tesla", "broadcom"], ["ndx", "spx"]),
    "Wall Street": (["wall street", "nasdaq", "dow jones", "s&p 500", "s&p", "bourse de new york", "stock market", "us stocks", "actions américaines"], ["dow", "ndx", "spx"]),
    "Or": (["l'or", "l’or", "prix de l'or", "métal jaune", "gold", "bullion", "xau", "métaux précieux", "precious metals"], ["gold"]),
    "Pétrole": (["pétrole", "brent", "wti", "crude", "oil", "opep", "opec", "baril"], ["oil"]),
    "Euro / BCE": (["euro", "bce", "ecb", "lagarde", "zone euro", "eurozone"], ["eur"]),
    "Dollar": (["dollar", "billet vert"], ["eur", "gold"]),
    "Politique US": (["élection", "élections", "election", "elections", "trump", "congrès", "congress", "shutdown", "sénat", "senate", "maison blanche", "white house", "midterm", "mi-mandat"], ["dow", "ndx", "spx", "gold", "eur"]),
    "Commerce": (["droits de douane", "tarifs douaniers", "tariff", "tariffs", "guerre commerciale", "trade war"], ["dow", "ndx", "spx", "gold"]),
    "Géopolitique": (["guerre", "war", "iran", "israël", "israel", "russie", "russia", "ukraine", "taïwan", "taiwan", "missile", "attaque", "attack", "sanctions", "conflit"], ["gold", "oil", "spx"]),
    "Macro US": (["inflation", "cpi", "emploi", "jobs report", "payrolls", "chômage", "unemployment", "pib", "gdp", "récession", "recession", "consommation"], ["dow", "ndx", "spx", "gold", "eur"]),
}
STRONG = ["chute", "plonge", "plongeon", "s'effondre", "effondrement", "dégringol", "krach", "panique", "déroute", "flambe", "flambée",
          "bondit", "s'envole", "envolée", "record", "historique", "choc", "tempête", "débâcle", "lourde", "sell-off", "selloff",
          "plunge", "plummet", "tumble", "crash", "soar", "surge", "rout", "slump", "sink", "spike", "rally", "breaking", "urgent",
          "surprise", "emergency", "turmoil", "worst", "best day", "rebondit", "décroche", "vacille", "secoue"]


# Articles de conseil / listes d'actions sans valeur d'information immédiate
NOISE = re.compile(r"(want to invest|here'?s (what|why|how|the)|should you|stocks? to buy|to buy (now|today)|buy (now|before)|"
                   r"is it time|best (stocks?|etfs?)|top \d|\d+ (stocks?|reasons)|prediction|billionaire|dividend|401\(k\)|"
                   r"retire|millionaire|passive income|history says|motley|could make you|no-brainer|forever)", re.I)


def _kw_hit(text, kw):
    return re.search(r"(?<![\w])" + re.escape(kw) + r"(?![\w])", text) is not None


def load_news():
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    def one(feed):
        src, url = feed
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            out = []
            for it in ET.fromstring(r.content).iter("item"):
                title = (it.findtext("title") or "").strip()
                source = src
                if src == "Google Actualités":
                    source = (it.findtext("source") or "").strip() or src
                    title = re.sub(r"\s+-\s+[^-]+$", "", title)
                try:
                    ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone(ZoneInfo("UTC"))
                except Exception:
                    continue
                out.append({"title": title, "source": source, "link": it.findtext("link") or "", "ts": ts,
                            "lang": "fr" if src == "Google Actualités" else "en"})
            return out
        except Exception:
            return []

    with ThreadPoolExecutor(4) as ex:
        return [x for lst in ex.map(one, NEWS_FEEDS) for x in lst]


# Catalyseurs capables de faire bouger fortement les marchés
CATALYST = re.compile(
    r"(relève ses taux|baisse ses taux|hausse surprise|baisse surprise|rate hike|rate cut|raises rates|cuts rates|hikes rates|"
    r"emergency|d'urgence|surprise|unexpected|inattendu|disappoint|décevant|déçoit|guidance|profit warning|avertissement|"
    r"guerre|war\b|invasion|invade|attaque|attack|missile|frappe|strike on|bombard|"
    r"résultat des élections|election result|wins election|remporte|victoire|landslide|vague bleue|blue wave|"
    r"droits de douane|tariffs? (on|of)|imposes|sanctions|embargo|"
    r"défaut de paiement|default|faillite|bankrupt|bank run|collapse|effondrement|krach|crash|circuit breaker|"
    r"halted|suspendu|black monday|panique|panic|worst day|pire séance|meilleure séance|best day since|"
    r"cpi|inflation (jumps|surges|soars|bondit|s'envole|accélère)|payrolls|nfp|emplois? (non agricoles)?)", re.I)
# Titres d'opinion / d'analyse : rarement la cause d'un mouvement
OPINION = re.compile(r"(chronique|tribune|opinion|analyse|analysis|et si |what to watch|à surveiller|could |pourrait|"
                     r"pourquoi|comment |why |how |\?)", re.I)
BIG_PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s?%")
# Déclaration d'un responsable de la Fed (souvent à l'origine de mouvements sur les taux, le dollar et l'or)
FED_OFFICIAL = re.compile(r"\b(powell|warsh|goolsbee|waller|williams|bowman|jefferson|barr|cook|logan|hammack|musalem|schmid|"
                          r"kashkari|daly|bostic|barkin|harker|collins|paulson|miran|fed chair|président de la fed)\b", re.I)
FED_TONE = re.compile(r"(warns?|signals?|hints?|says|calls for|pushes|backs|rules out|prévient|avertit|signale|plaide|exclut|"
                      r"hike|cut|hausse|baisse|aggressive|restrictive|dovish|hawkish)", re.I)
# Montant spectaculaire (capitalisation effacée, etc.)
BIG_MONEY = re.compile(r"(\$\s?\d+(?:[.,]\d+)?\s?(trillion|tn)|\d+(?:[.,]\d+)?\s?(trillion|billions? de dollars|milliards de dollars)|"
                       r"\d+(?:[.,]\d+)?\s?000 milliards)", re.I)


def _tokens(t):
    return {w for w in re.findall(r"[a-zà-ÿ0-9&]{4,}", t.lower())}


def stars(title, low, strong, tags, cluster):
    """Note de 1 à 5 : chance que l'information fasse bouger fortement les marchés."""
    sc = 1
    if strong:
        sc += 1.5
    if CATALYST.search(title):
        sc += 1.5
    if any(float(x.replace(",", ".")) >= 2 for x in BIG_PCT.findall(title)):
        sc += 1
    if FED_OFFICIAL.search(title) and FED_TONE.search(title):
        sc += 1.5
    if BIG_MONEY.search(title):
        sc += 1
    if cluster >= 3:
        sc += 1          # reprise par plusieurs médias en même temps
    elif cluster == 2:
        sc += 0.5
    if len(tags) >= 2:
        sc += 0.5
    if OPINION.search(title):
        sc -= 1.5
    return max(1, min(5, int(sc + 0.5)))


def news(raw, now):
    hours = 60 if now.weekday() >= 5 or now.weekday() == 0 and now.hour < 12 else 30
    cutoff = now.astimezone(ZoneInfo("UTC")) - timedelta(hours=hours)
    seen, items = set(), []
    for n in sorted(raw, key=lambda x: x["ts"], reverse=True):
        if n["ts"] < cutoff or not n["title"]:
            continue
        key = re.sub(r"[^a-z0-9]", "", n["title"].lower())[:50]
        if key in seen or NOISE.search(n["title"]):
            continue
        seen.add(key)
        low = n["title"].lower()
        tags = [t for t, (kws, _) in NEWS_TAGS.items() if any(_kw_hit(low, k) for k in kws)]
        if re.search(r"\b(IA|AI)\b", n["title"]) and "Tech / IA" not in tags:
            tags.append("Tech / IA")
        if not tags:
            continue
        loc = n["ts"].astimezone(PARIS)
        mins = int((now - n["ts"]).total_seconds() // 60)
        items.append({"title": n["title"], "source": n["source"], "link": n["link"], "lang": n["lang"],
                      "time": loc.strftime("%H:%M"), "day": loc.strftime("%d/%m"), "mins": max(mins, 0), "tags": tags,
                      "assets": sorted({a for t in tags for a in NEWS_TAGS[t][1]}, key=MAIN.index),
                      "strong": any(w in low for w in STRONG), "_tok": _tokens(n["title"])})
    # Nombre de titres proches (même événement couvert par plusieurs médias)
    for a in items:
        a["cluster"] = sum(1 for b in items if len(a["_tok"] & b["_tok"]) >= 3)
    for a in items:
        a["stars"] = stars(a["title"], a["title"].lower(), a["strong"], a["tags"], a["cluster"])
        del a["_tok"]
    top = [a for a in items if a["stars"] >= 5]
    # 4 étoiles : un seul titre par événement (le plus récent), avec le nombre d'articles qui en parlent
    watch, used = [], set()
    for a in items:
        if a["stars"] != 4 or id(a) in used:
            continue
        group = [b for b in items if b["stars"] == 4 and id(b) not in used and len(_tokens(a["title"]) & _tokens(b["title"])) >= 2
                 and set(a["tags"]) & set(b["tags"])]
        for b in group:
            used.add(id(b))
        watch.append({**a, "count": len(group)})
    return {"major": top[:8], "watch": watch[:5], "scanned": len(items)}


# ----------------------------------------------------------------------------- direct vidéo
LIVE_CHANNELS = [  # (nom, identifiant de chaîne YouTube, langue)
    ("Réserve fédérale", "UCAzhpt9DmG6PnHXjmJTvRGQ", "en"),
    ("Bloomberg TV", "UCIALMKvObZNtJ6AmdCLP7Lg", "en"),
    ("Yahoo Finance", "UCEAZeUIeJs0IjQiqTCdVSIg", "en"),
    ("France 24", "UCCCPCZNChQdGa9EkATeye4g", "fr"),
]


def load_live():
    def one(ch):
        name, cid, lang = ch
        try:
            r = requests.get(f"https://www.youtube.com/channel/{cid}/live", timeout=15,
                             headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "fr,en"},
                             cookies={"SOCS": "CAI"})  # CAI = cookies non essentiels refusés (évite la page de consentement en Europe)
            h = r.text
            if "--debug-live" in sys.argv:
                t = re.search(r"<title>([^<]*)</title>", h)
                print(f"[live] {name}: HTTP {r.status_code} {r.url[:70]} titre={t.group(1)[:60] if t else None} "
                      f"canonical={'canonical' in h} isLive={'isLive' in h} taille={len(h)}", file=sys.stderr)
                c = re.search(r'<link rel="canonical"[^>]*>', h)
                print(f"[live]   canonical={c.group(0)[:120] if c else None}", file=sys.stderr)
                print(f"[live]   videoIds={re.findall(r'"videoId":"([\w-]{11})"', h)[:6]}", file=sys.stderr)
                i = h.find('"isLive":true')
                print(f"[live]   autour isLive={h[max(0, i - 300):i + 40]!r}" if i >= 0 else "[live]   isLive absent", file=sys.stderr)
            vid = (re.search(r'rel="canonical"[^>]*watch\?v=([\w-]{11})', h) or re.search(r'og:url"[^>]*watch\?v=([\w-]{11})', h)
                   or re.search(r'"currentVideoEndpoint".{0,200}?"videoId":"([\w-]{11})"', h, re.S))
            live = bool(re.search(r'"isLiveNow":true|"isLive":true', h)) and vid is not None
            title = (re.search(r'<meta name="title" content="([^"]+)"', h) or re.search(r'<meta property="og:title" content="([^"]+)"', h))
            return {"name": name, "channel": cid, "lang": lang, "live": live,
                    "video": vid.group(1) if vid else None, "title": html_lib.unescape(title.group(1)) if (title and live) else ""}
        except Exception:
            return {"name": name, "channel": cid, "lang": lang, "live": False, "video": None, "title": ""}
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, LIVE_CHANNELS))


def next_speech(raw, now):
    """Prochaine prise de parole d'un responsable de la Fed (ou conférence de presse) dans le calendrier."""
    for e in sorted(raw, key=lambda e: e["date"]):
        d = pd.Timestamp(e["date"]).tz_convert(NY)
        t = e["title"].lower()
        if d >= now - timedelta(minutes=90) and any(k in t for k in ("speaks", "testifies", "press conference")):
            return {"title": fr_title(e["title"]), "ny": d.strftime("%a %d/%m %H:%M"), "paris": d.tz_convert(PARIS).strftime("%H:%M"),
                    "iso": d.isoformat(), "impact": e.get("impact")}
    return None


# ----------------------------------------------------------------------------- mouvements anormaux
def agitation(px, intr, now):
    """Compare le dernier mouvement de chaque marché à son mouvement habituel."""
    day, hour = [], []
    for k in MAIN:
        s = px[k].dropna()
        r = (np.log(s).diff() * 100).dropna()
        typ = r.iloc[-61:-1].abs().mean()
        mv = pct(s.iloc[-1], s.iloc[-2])
        ratio = abs(mv) / typ if typ else 0
        day.append({"key": k, "label": LABEL[k], "move": mv, "ratio": ratio, "typ": typ, "date": s.index[-1].strftime("%d/%m"),
                    "level": "anormal" if ratio >= 2.5 else "agité" if ratio >= 1.5 else "calme"})
    if intr is not None:
        for k, t in FUTURES.items():
            try:
                c = intr["Close"][t].dropna()
            except KeyError:
                continue
            if len(c) < 100 or (now - c.index[-1]) > timedelta(hours=3):
                continue
            hr = (c.pct_change() * 100).dropna()
            typ = hr.iloc[-500:-1].abs().mean()
            last = hr.iloc[-1]
            ratio = abs(last) / typ if typ else 0
            if ratio >= 3.5:
                hour.append({"key": k, "label": LABEL[k], "move": last, "ratio": ratio, "at": c.index[-1].strftime("%H:%M"),
                             "at_paris": c.index[-1].astimezone(PARIS).strftime("%H:%M")})
    return {"day": day, "hour": sorted(hour, key=lambda x: -x["ratio"])}


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
        intr = cached("intr", 300, load_intraday, force)
    except Exception:
        intr = None
    try:
        live = cached("live", 600, load_live, force)
    except Exception:
        live = []
    try:
        nw = news(cached("news", 240, load_news, force), now)
    except Exception:
        nw = {"major": [], "watch": [], "scanned": 0}
    th, totals = themes(px, fred)
    cards = asset_cards(px)
    for c in cards:
        c["macro"] = totals[c["key"]]
    return clean({"updated": now.strftime("%d/%m/%Y %H:%M"), "session": session_info(now), "assets": cards,
                  "bilan": bilan(th, totals), "corr": correlations(px), "news": nw, "live": live, "speech": next_speech(cal, now),
                  "agitation": agitation(px, intr, now), "calendar": calendar(cal, now), "labels": {k: LABEL[k] for k in MAIN}})


app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    resp = send_from_directory(BASE / "static", "index.html")
    resp.headers["Cache-Control"] = "no-store"  # toujours servir la dernière version de la page
    return resp


@app.get("/api/dashboard")
def api():
    return jsonify(build(force=request.args.get("force") == "1"))


if __name__ == "__main__":
    import os
    # 0.0.0.0 : accessible depuis les appareils du même réseau Wi-Fi (téléphone). HOST=127.0.0.1 pour limiter au Mac.
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8501)), debug=False, threaded=True)
