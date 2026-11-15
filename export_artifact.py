"""Construit une version autonome du tableau de bord, avec les données intégrées.

Usage :
  .venv/bin/python export_artifact.py           → dist/us-session-gold.html (lien claude.ai, page sans squelette)
  .venv/bin/python export_artifact.py --pages   → site/index.html (GitHub Pages, page complète, se recharge seule)
"""
import json
import re
import sys
from pathlib import Path

import server

BASE = Path(__file__).parent
PAGES = "--pages" in sys.argv

html = (BASE / "static" / "index.html").read_text()
data = server.build(force=False)
data["mode"] = "pages" if PAGES else "artifact"

if not PAGES:
    # La page claude.ai reçoit son propre squelette : on retire doctype, html, head, body et meta viewport.
    for pat in (r"<!doctype html>\s*", r"<html[^>]*>\s*", r"</html>\s*", r"<head>\s*", r"</head>\s*", r"<body>\s*", r"</body>\s*",
                r'<meta charset="utf-8">\s*', r'<meta name="viewport"[^>]*>\s*'):
        html = re.sub(pat, "", html, flags=re.I)
    html = html.replace("<title>US Session & Gold</title>",
                        '<title>US Session & Gold</title>\n<meta name="claude:viewport-fit-sides" content="cover">', 1)
else:
    # Page web autonome : ajout des réglages « écran d'accueil » pour téléphone.
    html = html.replace('<meta name="viewport" content="width=device-width, initial-scale=1">',
                        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                        '<meta name="theme-color" content="#0a0e15">\n'
                        '<meta name="apple-mobile-web-app-capable" content="yes">\n'
                        '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">\n'
                        '<meta name="apple-mobile-web-app-title" content="US Gold">', 1)

snap = "<script>window.SNAPSHOT = " + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + ";</script>\n"
i = html.rfind("<script>")
html = html[:i] + snap + html[i:]

out = BASE / ("site/index.html" if PAGES else "dist/us-session-gold.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"{out.relative_to(BASE)} · {len(html) / 1024:.0f} Ko · données du {data['updated']}")
