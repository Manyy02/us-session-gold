"""Envoie une notification sur le téléphone (via ntfy) quand il se passe quelque chose d'important.

Déclencheurs :
  - mouvement horaire anormal (bougie d'1 h ≥ 3,5× la normale) sur un des 6 marchés ;
  - séance « anormale » (variation du jour ≥ 2,5× la normale) ;
  - actualité ⚡ (mouvement fort annoncé dans le titre) publiée il y a moins de 2 h.

Chaque alerte n'est envoyée qu'une fois : les alertes déjà envoyées sont notées dans notify_state.json.
Le canal ntfy est lu dans la variable d'environnement NTFY_TOPIC (secret GitHub) : sans elle, le script ne fait rien.

Usage : python notify.py   (après export_artifact.py --pages, qui écrit site/data.json)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = Path(__file__).parent
STATE = BASE / "notify_state.json"
PAGE = "https://manyy02.github.io/us-session-gold/"
MAX_PER_RUN = 5
KEEP_SECONDS = 4 * 24 * 3600


def send(topic, title, message, priority=4, tags=None, click=PAGE):
    r = requests.post("https://ntfy.sh/", timeout=15, json={
        "topic": topic, "title": title, "message": message, "priority": priority,
        "tags": tags or [], "click": click})
    r.raise_for_status()


def events(d):
    out = []
    for h in d["agitation"].get("hour", []):
        out.append((f"h:{h['key']}:{h['at']}", 5, ["warning"],
                    f"⚠️ {h['label']} {h['move']:+.2f} % en 1 heure",
                    f"{h['ratio']:.1f}× le mouvement horaire habituel · bougie de {h['at']} NY ({h['at_paris']} Paris)", PAGE))
    for a in d["agitation"]["day"]:
        if a["level"] == "anormal":
            out.append((f"d:{a['key']}:{a['date']}", 4, ["chart_with_upwards_trend" if a["move"] > 0 else "chart_with_downwards_trend"],
                        f"📊 {a['label']} {a['move']:+.2f} % sur la séance",
                        f"{a['ratio']:.1f}× la variation habituelle (séance du {a['date']})", PAGE))
    for n in d["news"]:
        if n["strong"] and n["mins"] <= 120:
            nid = "n:" + hashlib.sha1(n["title"].encode()).hexdigest()[:16]
            out.append((nid, 4, ["zap"], "⚡ " + " · ".join(n["tags"]),
                        f"{n['title']}\n{n['source']} · {n['time']} Paris", n["link"] or PAGE))
    return out


def main():
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print("NTFY_TOPIC absent : aucune notification envoyée.")
        return
    d = json.loads((BASE / "site" / "data.json").read_text())
    now = time.time()
    first_run = not STATE.exists()
    state = {} if first_run else json.loads(STATE.read_text())
    state = {k: v for k, v in state.items() if now - v < KEEP_SECONDS}

    evs = events(d)
    if first_run:
        # Première exécution : on note l'existant sans l'envoyer, pour éviter une rafale.
        for e in evs:
            state[e[0]] = now
        send(topic, "✅ Notifications US Session & Gold activées",
             "Vous serez prévenu des mouvements anormaux et des actualités ⚡ sur le Dow, le Nasdaq, le S&P 500, l'or, le pétrole et l'EUR/USD.",
             priority=3, tags=["white_check_mark"])
        print(f"Première exécution : {len(evs)} alertes existantes notées, message de confirmation envoyé.")
    else:
        sent = 0
        for eid, prio, tags, title, msg, click in evs:
            if eid in state or sent >= MAX_PER_RUN:
                continue
            send(topic, title, msg, prio, tags, click)
            state[eid] = now
            sent += 1
        print(f"{sent} notification(s) envoyée(s), {len(evs)} alerte(s) actives.")
    STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # une erreur de notification ne doit jamais bloquer la mise à jour de la page
        print("Notification impossible :", e, file=sys.stderr)
