"""Render a debate transcript as a shareable verdict card.

The card shows the panel's reasoning and the evidence trail, not just the
answer. A verdict you cannot audit is worth no more than the argument it was
meant to settle -- so the grounding result is printed on the card itself, even
when it is unflattering.
"""

from __future__ import annotations

import html
import json
from typing import Any

# Same "Wisden Almanack" palette as the static site (src/build_site.py):
# warm paper, ink type, one almanack-yellow accent, serif headlines. Kept in
# sync by hand -- the two files don't share an import path, but a shared
# design system means one page and one card should never look unrelated.
CSS = """
:root{
  --bg:#f4f0e6;--card:#fbf9f3;--ink:#14120c;--muted:#6e6455;--line:#ded6c2;
  --accent:#f2c300;--accent-ink:#8a6b00;--soft:#faf0cc;
  --warn:#8c2318;--warnbg:#f6e4de;--ok:#2f6b45;--okbg:#e3ede2;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
:root:not([data-theme="light"]){@media(prefers-color-scheme:dark){
  --bg:#14130f;--card:#1c1a15;--ink:#f0ebdd;--muted:#9a9284;--line:#2e2b23;
  --accent:#e8be2a;--accent-ink:#e8be2a;--soft:#332b12;
  --warn:#d4685c;--warnbg:#3a201d;--ok:#6aa77f;--okbg:#1c2e21}}
:root[data-theme="dark"]{
  --bg:#14130f;--card:#1c1a15;--ink:#f0ebdd;--muted:#9a9284;--line:#2e2b23;
  --accent:#e8be2a;--accent-ink:#e8be2a;--soft:#332b12;
  --warn:#d4685c;--warnbg:#3a201d;--ok:#6aa77f;--okbg:#1c2e21}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;padding:36px 18px 70px;
  font:16px/1.6 var(--serif);-webkit-font-smoothing:antialiased}
.card{max-width:720px;margin:0 auto;background:var(--card);border:1px solid var(--line);
  border-radius:14px;overflow:hidden}
.q{padding:26px 28px 20px;border-bottom:1px solid var(--line)}
.tag{font:600 11px var(--sans);letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--soft);padding:3px 9px;border-radius:5px}
.q h1{font:400 26px/1.3 var(--serif);margin:14px 0 0;letter-spacing:-.02em}
.vd{padding:24px 28px;background:var(--soft)}
.vd .big{font:600 21px/1.3 var(--sans);letter-spacing:-.015em;margin:0 0 10px}
.meta{display:flex;gap:26px;flex-wrap:wrap;margin-top:14px}
.meta div{font:14px var(--serif)}
.meta b{display:block;color:var(--muted);font:600 10.5px var(--sans);text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:3px}
.conf{font:700 15px var(--sans);font-variant-numeric:tabular-nums;color:var(--accent-ink)}
.panel{padding:8px 28px 22px}
.turn{padding:16px 0;border-bottom:1px solid var(--line)}
.turn:last-child{border-bottom:none}
.role{font:600 11px var(--sans);letter-spacing:.1em;color:var(--muted);
  text-transform:uppercase}
.tool{font:11px var(--mono);color:var(--accent-ink);margin-left:8px}
.turn p{margin:7px 0 0;font:15px/1.6 var(--serif)}
.warn{margin:0 28px 22px;padding:12px 15px;background:var(--warnbg);color:var(--warn);
  border-radius:9px;font:14px var(--sans)}
.ok{margin:0 28px 22px;padding:12px 15px;background:var(--okbg);color:var(--ok);
  border-radius:9px;font:14px var(--sans)}
details{margin:0 28px 24px;font:13px var(--sans)}
summary{cursor:pointer;color:var(--muted);user-select:none}
pre{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:13px;
  overflow-x:auto;font:11.5px/1.5 var(--mono);margin:10px 0 0}
footer{max-width:720px;margin:18px auto 0;color:var(--muted);font:12.5px var(--sans);text-align:center}
"""


def render_card(tr: Any) -> str:
    v = tr.verdict or {}
    e = html.escape

    turns = "".join(
        f'<div class="turn"><span class="role">{e(t.role)}</span>'
        + (f'<span class="tool">{e(", ".join(c["name"] for c in t.tool_calls))}</span>'
           if t.tool_calls else "")
        + f"<p>{e(t.text)}</p></div>"
        for t in tr.turns[:-1] if t.text
    )

    ung = tr.ungrounded_numbers()
    grounding = (
        f'<div class="warn"><b>Grounding check failed.</b> '
        f'{len(ung)} number(s) in this verdict were never returned by a tool: '
        f'{e(", ".join(ung))}. Treat them as unsupported.</div>'
        if ung else
        '<div class="ok"><b>Grounding check passed.</b> '
        'Every number in this verdict traces to a tool result.</div>'
    )

    conf = v.get("confidence")
    conf_html = f'<span class="conf">{e(str(conf))}%</span>' if conf is not None else "·"
    dissent = (
        f'<div><b>Strongest objection</b>{e(str(v["dissent"]))}</div>'
        if v.get("dissent") else ""
    )

    evidence = e(json.dumps(tr.facts, indent=2, default=str)[:9000])

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verdict: {e(tr.question[:60])}</title>
<style>{CSS}</style>
<div class="card">
  <div class="q"><span class="tag">Howzat verdict</span><h1>{e(tr.question)}</h1></div>
  <div class="vd">
    <p class="big">{e(str(v.get("verdict", "no verdict")))}</p>
    <div class="meta">
      <div><b>Confidence</b>{conf_html}</div>
      <div><b>Decisive stat</b>{e(str(v.get("decisive_stat") or "·"))}</div>
      {dissent}
    </div>
  </div>
  {grounding}
  <div class="panel">{turns}</div>
  <details><summary>Evidence trail: every tool call behind this verdict</summary>
    <pre>{evidence}</pre></details>
</div>
<footer>Grounded in era-adjusted ratings over every Test, ODI and T20I innings,
men's and women's, 1877–2026. 100 = an average player of the same era and format.</footer>"""
