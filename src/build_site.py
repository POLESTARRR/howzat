"""Generate a self-contained CRI+ explorer.

Everything is embedded in one HTML file: no backend, no CDN, no API key. Open
out/index.html and it works. This is the 30-second demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cri_plus import load_innings

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "out"


def _bat_view(path: Path, label: str) -> dict | None:
    """One batting dataset (a format) shaped for the client."""
    if not path.exists():
        return None
    r = pd.read_parquet(path)
    main = "bat_plus" if "bat_plus" in r.columns else "cri_plus"
    r = r.sort_values(main, ascending=False).reset_index(drop=True)
    r["rank"] = r.index + 1
    r["avg_rank"] = r.average.rank(ascending=False).astype(int)
    r["move"] = r.avg_rank - r["rank"]
    rows = []
    for _, row in r.iterrows():
        sr = row.get("sr_plus")
        rows.append({
            "r": int(row["rank"]), "n": row.player, "c": row.country,
            "i": int(row.innings), "a": round(float(row.average), 2),
            "p": round(float(row[main]), 1),
            "lo": round(float(row.get("cri_lo", row[main])), 1),
            "hi": round(float(row.get("cri_hi", row[main])), 1),
            "sr": None if sr is None or pd.isna(sr) else round(float(sr), 1),
            "y": f"{int(row.first_year)}\u2013{int(row.last_year)}",
            "m": int(row["move"]),
        })
    return {"label": label, "kind": "bat",
            "metric": "BAT+" if main == "bat_plus" else "CRI+", "rows": rows}


def _bowl_view(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    b = pd.read_parquet(path).sort_values("bowl_plus", ascending=False).reset_index(drop=True)
    b["rank"] = b.index + 1
    b["avg_rank"] = b.bowl_average.rank(ascending=True).astype(int)
    b["move"] = b.avg_rank - b["rank"]
    rows = [{
        "r": int(x["rank"]), "n": x.player, "c": x.country,
        "i": int(x.wickets), "a": round(float(x.bowl_average), 2),
        "p": round(float(x.bowl_plus), 1),
        "lo": round(float(x.get("bowl_lo", x.bowl_plus)), 1),
        "hi": round(float(x.get("bowl_hi", x.bowl_plus)), 1),
        "sr": round(float(x.economy), 2),
        "y": f"{int(x.first_year)}\u2013{int(x.last_year)}",
        "m": int(x["move"]),
    } for _, x in b.iterrows()]
    return {"label": label, "kind": "bowl", "metric": "BOWL+", "rows": rows}


def build_payload() -> dict:
    r = pd.read_parquet(PROC / "cri_plus.parquet")
    inn = load_innings()  # adds the derived `decade` column

    r = r.sort_values("cri_plus", ascending=False).reset_index(drop=True)
    r["rank"] = r.index + 1
    r["avg_rank"] = r.average.rank(ascending=False).astype(int)
    r["move"] = r.avg_rank - r["rank"]

    players = [
        {
            "r": int(row["rank"]),
            "n": row.player,
            "c": row.country,
            "i": int(row.innings),
            "a": round(float(row.average), 2),
            "p": round(float(row.cri_plus), 1),
            "lo": round(float(row.cri_lo), 1),
            "hi": round(float(row.cri_hi), 1),
            "y": f"{int(row.first_year)}–{int(row.last_year)}",
            "m": int(row["move"]),
        }
        for _, row in r.iterrows()
    ]

    g = inn.groupby("decade")
    era = (g.runs.sum() / (g.size() - g.not_out.sum()).clip(lower=1)).round(2)
    eras = [{"d": int(d), "avg": float(v), "n": int(g.size()[d])} for d, v in era.items()]

    views = {"test_bat": {"label": "Test batting", "kind": "bat",
                          "metric": "CRI+", "rows": players}}
    for key, path, label in [
        ("odi_bat", PROC / "cri_plus_odi.parquet", "ODI batting"),
        ("t20_bat", PROC / "cri_plus_t20i.parquet", "T20I batting"),
    ]:
        v = _bat_view(path, label)
        if v:
            views[key] = v
    for key, path, label in [
        ("test_bowl", PROC / "bowl_plus_test.parquet", "Test bowling"),
        ("odi_bowl", PROC / "bowl_plus_odi.parquet", "ODI bowling"),
        ("t20_bowl", PROC / "bowl_plus_t20i.parquet", "T20I bowling"),
    ]:
        v = _bowl_view(path, label)
        if v:
            views[key] = v

    return {
        "players": players,
        "views": views,
        "eras": eras,
        "meta": {
            "innings": int(len(inn)),
            "span": f"{int(inn.year.min())}–{int(inn.year.max())}",
            "rated": int(len(r)),
            "total_players": int(inn.player.nunique()),
        },
    }


CSS = """
:root{
  --bg:#faf9f7; --panel:#fff; --ink:#16150f; --muted:#6b6862; --line:#e5e1d8;
  --accent:#0f6b4f; --accent-soft:#d9ece4; --bar:#c8bfae; --up:#0f6b4f; --down:#a4423a;
}
:root:not([data-theme="light"]){ @media (prefers-color-scheme:dark){
  --bg:#131311; --panel:#1c1b19; --ink:#f2efe8; --muted:#9d9890; --line:#2e2c28;
  --accent:#5ec39a; --accent-soft:#1e3b31; --bar:#453f36; --up:#5ec39a; --down:#e0796e;
}}
:root[data-theme="dark"]{
  --bg:#131311; --panel:#1c1b19; --ink:#f2efe8; --muted:#9d9890; --line:#2e2c28;
  --accent:#5ec39a; --accent-soft:#1e3b31; --bar:#453f36; --up:#5ec39a; --down:#e0796e;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
  margin:0;padding:0 20px 80px}
.wrap{max-width:1000px;margin:0 auto}
header{padding:56px 0 28px;border-bottom:1px solid var(--line)}
h1{font-size:clamp(30px,5vw,46px);margin:0 0 10px;letter-spacing:-.025em;font-weight:640}
.sub{color:var(--muted);font-size:17px;max-width:62ch;margin:0}
.kbd{font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--accent-soft);
  color:var(--accent);padding:2px 7px;border-radius:5px}
.stats{display:flex;flex-wrap:wrap;gap:26px;margin:26px 0 0}
.stat b{display:block;font-size:24px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.stat span{color:var(--muted);font-size:12.5px;text-transform:uppercase;letter-spacing:.07em}
h2{font-size:20px;margin:44px 0 6px;letter-spacing:-.015em}
.note{color:var(--muted);font-size:14px;margin:0 0 18px;max-width:70ch}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.controls{display:flex;gap:10px;flex-wrap:wrap;padding:14px;border-bottom:1px solid var(--line)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;padding:12px 14px 0}
.tabs button{background:transparent;border:1px solid var(--line);color:var(--muted);
  padding:6px 13px;border-radius:20px;font:inherit;font-size:13px;cursor:pointer}
.tabs button.on{background:var(--accent);border-color:var(--accent);color:var(--bg);font-weight:600}
input,select{background:var(--bg);border:1px solid var(--line);color:var(--ink);
  padding:8px 11px;border-radius:8px;font-size:14px;font-family:inherit}
input{flex:1;min-width:190px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-weight:600;
  font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;text-align:right;cursor:pointer;
  user-select:none}
th:first-child,td:first-child{text-align:center;color:var(--muted);font-variant-numeric:tabular-nums}
th:nth-child(2),td:nth-child(2){text-align:left}
td{font-variant-numeric:tabular-nums}
tbody tr:hover{background:var(--accent-soft)}
.nm{font-weight:600}
.ctry{color:var(--muted);font-size:12px;margin-left:6px}
.yr{color:var(--muted);font-size:12px}
.pv{font-weight:700;color:var(--accent)}
.citrack{position:relative;display:inline-block;width:170px;height:9px;
  background:var(--bar);opacity:.55;border-radius:5px;vertical-align:middle}
.ci{position:absolute;top:0;height:100%;background:var(--accent);opacity:.42;border-radius:5px}
.ci i{position:absolute;top:-3px;width:2.5px;height:15px;background:var(--accent);
  border-radius:2px;opacity:1}
.up{color:var(--up)} .down{color:var(--down)}
.bars{display:flex;align-items:flex-end;gap:5px;height:150px;padding:16px 14px 0}
.bars div{flex:1;background:var(--accent);border-radius:3px 3px 0 0;opacity:.8;position:relative}
.bars div:hover{opacity:1}
.blabels{display:flex;gap:5px;padding:6px 14px 14px;color:var(--muted);font-size:10.5px}
.blabels span{flex:1;text-align:center}
footer{margin-top:52px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--accent-soft);
  padding:1.5px 5px;border-radius:4px}
"""


def render(payload: dict) -> str:
    m = payload["meta"]
    return f"""<title>CRI+ — Era-Adjusted Cricket Ratings</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Cricket has no <span class="kbd">wRC+</span></h1>
  <p class="sub">Baseball fixed cross-era comparison decades ago. Cricket still uses
  batting average — invented in the 1800s, adjusted for nothing. <b>CRI+</b> is an
  era-adjusted Test batting rating where <b>100 = an average batter of the same era</b>,
  so 1930 and 2026 are finally on one scale.</p>
  <div class="stats">
    <div class="stat"><b>{m['innings']:,}</b><span>Test innings</span></div>
    <div class="stat"><b>{m['span']}</b><span>every Test ever</span></div>
    <div class="stat"><b>{m['total_players']:,}</b><span>players</span></div>
    <div class="stat"><b>{m['rated']:,}</b><span>rated (20+ inns)</span></div>
  </div>
</header>

<h2>How hard was it to score?</h2>
<p class="note">Runs per dismissal by decade, across every Test batter. This spans
<b>1.96×</b> — which is exactly what raw batting average ignores, and what CRI+ divides out.</p>
<div class="panel"><div class="bars" id="bars"></div><div class="blabels" id="blabels"></div></div>

<h2>The ratings</h2>
<p class="note">Bars show the <b>95% interval</b>. A short career gives a wide bar — that is the
point. <code>Δ</code> is how many places a batter moves versus raw batting average;
positive means the era adjustment promoted them. Click any column to sort.</p>
<div class="panel">
  <div class="tabs" id="tabs"></div>
  <div class="controls">
    <input id="q" placeholder="Search a player…" autocomplete="off">
    <select id="minInns">
      <option value="20">20+ innings</option>
      <option value="40" selected>40+ innings</option>
      <option value="80">80+ innings</option>
      <option value="150">150+ innings</option>
    </select>
    <select id="ctry"><option value="">All countries</option></select>
  </div>
  <div class="scroll"><table>
    <thead><tr>
      <th data-k="r">#</th><th data-k="n">Player</th><th data-k="i" id="hN">Inns</th>
      <th data-k="a" id="hAvg">Avg</th><th data-k="sr" id="hSr">SR+</th>
      <th data-k="p" id="hMain">CRI+</th>
      <th id="hCi">95% interval</th><th data-k="m">Δ</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
</div>

<footer>
  <b>Method.</b> Innings runs modelled as Geometric with a log link; not-outs enter as
  right-censored observations rather than being averaged in. Player skill carries a Gaussian
  prior whose strength is set by empirical Bayes (λ = 1.90), so short careers shrink toward the
  mean. The era term is <i>fixed from observed scoring</i>, not fitted — estimated freely it came
  out anti-correlated with reality (r = −0.21), because only 12.7% of players span three decades
  and era is confounded with skill. Fitted by penalised MLE with analytic gradients.
  <br><br>
  CRI+ measures <b>dominance over contemporaries</b>, not absolute skill across eras — the same
  claim baseball's <code>+</code> metrics make, and the strongest this data supports.
  Data: ESPNcricinfo Statsguru.
</footer>
</div>

<script>
const VIEWS = {json.dumps(payload["views"], separators=(",", ":"))};
let VIEW = Object.keys(VIEWS)[0];
let DATA = VIEWS[VIEW].rows;
const ERAS = {json.dumps(payload["eras"], separators=(",", ":"))};

const bars = document.getElementById('bars'), blab = document.getElementById('blabels');
const mx = Math.max(...ERAS.map(e => e.avg));
ERAS.forEach(e => {{
  const d = document.createElement('div');
  d.style.height = (e.avg / mx * 100) + '%';
  d.title = e.d + 's — ' + e.avg + ' runs per dismissal (' + e.n.toLocaleString() + ' innings)';
  bars.appendChild(d);
  const s = document.createElement('span');
  s.textContent = String(e.d).slice(2) + "s";
  blab.appendChild(s);
}});

let LO = 0, HI = 1;
const scale = v => (v - LO) / (HI - LO) * 100;
const sel = document.getElementById('ctry');

function setView(k) {{
  VIEW = k; DATA = VIEWS[k].rows;
  LO = Math.min(...DATA.map(p => p.lo)); HI = Math.max(...DATA.map(p => p.hi));
  const bowl = VIEWS[k].kind === 'bowl';
  document.getElementById('hN').textContent   = bowl ? 'Wkts' : 'Inns';
  document.getElementById('hAvg').textContent = bowl ? 'Bowl avg' : 'Avg';
  document.getElementById('hSr').textContent  = bowl ? 'Econ' : 'SR+';
  document.getElementById('hMain').textContent = VIEWS[k].metric || 'CRI+';
  document.getElementById('hCi').textContent  = '95% interval';
  document.getElementById('minInns').style.display = bowl ? 'none' : '';
  sel.innerHTML = '<option value="">All countries</option>';
  [...new Set(DATA.map(p => p.c))].sort().forEach(c => {{
    const o = document.createElement('option'); o.value = o.textContent = c; sel.appendChild(o);
  }});
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('on', b.dataset.v === k));
  draw();
}}

const tabs = document.getElementById('tabs');
Object.entries(VIEWS).forEach(([k, v]) => {{
  const b = document.createElement('button');
  b.textContent = v.label; b.dataset.v = k; b.onclick = () => setView(k);
  tabs.appendChild(b);
}});

let sortK = 'p', sortAsc = false;

function draw() {{
  const q = document.getElementById('q').value.trim().toLowerCase();
  const mi = +document.getElementById('minInns').value;
  const cc = sel.value;

  const bowl = VIEWS[VIEW].kind === 'bowl';
  let rows = DATA.filter(p => (bowl || p.i >= mi) && (!cc || p.c === cc) &&
                              (!q || p.n.toLowerCase().includes(q)));
  rows.sort((a, b) => {{
    const x = a[sortK], y = b[sortK];
    const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
    return sortAsc ? c : -c;
  }});

  document.getElementById('rows').innerHTML = rows.slice(0, 300).map(p => {{
    const l = scale(p.lo), w = Math.max(scale(p.hi) - l, 1.2), c = scale(p.p);
    const pt = Math.min(Math.max((c - l) / w * 100, 0), 100);
    const mv = p.m > 0 ? `<span class="up">+${{p.m}}</span>`
             : p.m < 0 ? `<span class="down">${{p.m}}</span>` : '·';
    return `<tr>
      <td>${{p.r}}</td>
      <td><span class="nm">${{p.n}}</span><span class="ctry">${{p.c}}</span>
          <div class="yr">${{p.y}}</div></td>
      <td>${{p.i}}</td><td>${{p.a.toFixed(bowl ? 2 : 1)}}</td>
      <td>${{p.sr == null ? '·' : p.sr.toFixed(bowl ? 2 : 0)}}</td>
      <td class="pv">${{p.p.toFixed(0)}}</td>
      <td>${{`<span class="citrack" title="95% CI ${{p.lo.toFixed(0)}}–${{p.hi.toFixed(0)}} (width ${{(p.hi-p.lo).toFixed(0)}} pts)"><span class="ci" style="left:${{l.toFixed(2)}}%;width:${{w.toFixed(2)}}%"><i style="left:${{pt.toFixed(1)}}%"></i></span></span>`}}</td>
      <td>${{mv}}</td></tr>`;
  }}).join('');
}}

document.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => {{
  const k = th.dataset.k;
  if (k === sortK) sortAsc = !sortAsc; else {{ sortK = k; sortAsc = (k === 'r' || k === 'n'); }}
  draw();
}});
['q', 'minInns', 'ctry'].forEach(id =>
  document.getElementById(id).addEventListener('input', draw));
setView(VIEW);
</script>"""


def main() -> None:
    payload = build_payload()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "index.html"
    dest.write_text(render(payload), encoding="utf-8")
    kb = dest.stat().st_size / 1024
    print(f"wrote {dest}  ({kb:.0f} KB, {len(payload['players'])} players, no backend)")


if __name__ == "__main__":
    main()
