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
# GitHub Pages publishes main:/docs, so a build that only wrote out/ left the
# live site on whatever was last copied across by hand. Write both.
DOCS = ROOT / "docs"


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
            # Show the interval that belongs to the metric on screen.
            "lo": round(float(row.get("bat_lo" if main == "bat_plus" else "cri_lo",
                                      row[main])), 1),
            "hi": round(float(row.get("bat_hi" if main == "bat_plus" else "cri_hi",
                                      row[main])), 1),
            "sr": None if sr is None or pd.isna(sr) else round(float(sr), 1),
            "y": f"{int(row.first_year)}\u2013{int(row.last_year)}",
            "m": int(row["move"]),
        })
    # A fixed 40-innings default blanked the women's Test tab entirely, whose
    # careers run 8-24 innings. Each view picks a floor its own data supports.
    # Default to a floor that keeps most of the field visible: the 25th
    # percentile of career lengths, capped at 40. A fixed 40 blanked the
    # women's Test tab, whose careers run 8-24 innings.
    inns = sorted(r["i"] for r in rows) or [0]
    floor = inns[max(len(inns) // 4 - 1, 0)]
    return {"label": label, "kind": "bat", "min_default": int(min(floor, 40)),
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
    return {"label": label, "kind": "bowl", "min_default": 0,
            "metric": "BOWL+", "rows": rows}


def _aliases() -> dict:
    """Fans type nicknames, not Statsguru's 'initials surname' spelling."""
    try:
        from tools import ALIASES

        return dict(ALIASES)
    except Exception:
        return {}


def _all_innings_count() -> int:
    """Every innings across all formats, batting and bowling."""
    import glob

    total = 0
    for f in glob.glob(str(PROC / "*_innings.parquet")) + \
             glob.glob(str(PROC / "*_bowling.parquet")):
        try:
            total += len(pd.read_parquet(f, columns=["year"]))
        except Exception:
            pass
    return total


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
                          "min_default": 40, "metric": "CRI+", "rows": players}}
    for key, path, label in [
        ("odi_bat", PROC / "cri_plus_odi.parquet", "ODI batting"),
        ("t20_bat", PROC / "cri_plus_t20i.parquet", "T20I batting"),
        ("wodi_bat", PROC / "cri_plus_wodi.parquet", "Women's ODI"),
        ("wt20_bat", PROC / "cri_plus_wt20i.parquet", "Women's T20I"),
        ("wtest_bat", PROC / "cri_plus_wtest.parquet", "Women's Test"),
    ]:
        v = _bat_view(path, label)
        if v:
            views[key] = v
    for key, path, label in [
        ("test_bowl", PROC / "bowl_plus_test.parquet", "Test bowling"),
        ("odi_bowl", PROC / "bowl_plus_odi.parquet", "ODI bowling"),
        ("t20_bowl", PROC / "bowl_plus_t20i.parquet", "T20I bowling"),
        ("wodi_bowl", PROC / "bowl_plus_wodi.parquet", "Women's ODI bowling"),
        ("wt20_bowl", PROC / "bowl_plus_wt20i.parquet", "Women's T20I bowling"),
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
            "all_innings": _all_innings_count(),
            "views": len(views),
        },
        "aliases": _aliases(),
    }


# Wisden Almanack: warm paper stock, ink type, one almanack-yellow accent.
# Serif for prose and headlines, system sans with tabular figures for every
# number. All stacks are system-resident: this file must stay self-contained,
# so no webfont or CDN request is allowed.
CSS = """
:root{
  --paper:#f4f0e6; --panel:#fbf9f3; --ink:#14120c; --muted:#6e6455;
  --line:#ded6c2; --hair:#e8e2d3;
  --accent:#f2c300; --accent-ink:#8a6b00; --accent-wash:#faf0cc;
  --ball:#8c2318; --turf:#2f6b45; --bar:#b9ad93;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
:root:not([data-theme="light"]){ @media (prefers-color-scheme:dark){
  --paper:#14130f; --panel:#1c1a15; --ink:#f0ebdd; --muted:#9a9284;
  --line:#2e2b23; --hair:#26241d;
  --accent:#e8be2a; --accent-ink:#e8be2a; --accent-wash:#332b12;
  --ball:#d4685c; --turf:#6aa77f; --bar:#3b372d;
}}
:root[data-theme="dark"]{
  --paper:#14130f; --panel:#1c1a15; --ink:#f0ebdd; --muted:#9a9284;
  --line:#2e2b23; --hair:#26241d;
  --accent:#e8be2a; --accent-ink:#e8be2a; --accent-wash:#332b12;
  --ball:#d4685c; --turf:#6aa77f; --bar:#3b372d;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:16px/1.6 var(--serif);
  margin:0;padding:0 24px 90px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto}

/* ---- masthead ---------------------------------------------------------- */
header{padding:52px 0 0}
.eyebrow{font:600 11px/1 var(--sans);text-transform:uppercase;letter-spacing:.18em;
  color:var(--muted);display:flex;align-items:center;gap:10px;margin:0 0 26px}
.eyebrow::after{content:"";flex:1;height:1px;background:var(--line)}
.eyebrow b{color:var(--ink);font-weight:700}
h1{font:400 clamp(44px,7.5vw,78px)/0.95 var(--serif);margin:0 0 18px;
  letter-spacing:-.02em}
h1 em{font-style:italic;color:var(--accent-ink)}
.sub{font-size:19px;line-height:1.55;max-width:34em;margin:0;color:var(--ink)}
.sub b{font-weight:600;box-shadow:inset 0 -.5em 0 var(--accent-wash)}
.rule{height:1px;background:var(--line);margin:30px 0 0}
.rule.dbl{height:3px;background:none;border-top:1px solid var(--line);
  border-bottom:1px solid var(--line)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  border-top:1px solid var(--line);border-bottom:3px double var(--line);
  margin:28px 0 0}
.stat{padding:16px 20px 15px;border-left:1px solid var(--hair)}
.stat:first-child{border-left:0;padding-left:0}
.stat b{display:block;font:600 28px/1 var(--sans);font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;margin-bottom:6px}
.stat span{color:var(--muted);font:600 10.5px/1 var(--sans);text-transform:uppercase;
  letter-spacing:.13em}

/* ---- section headings -------------------------------------------------- */
h2{font:400 30px/1.15 var(--serif);margin:64px 0 8px;letter-spacing:-.015em}
.note{color:var(--muted);font-size:15.5px;margin:0 0 20px;max-width:38em}
.note b{color:var(--ink);font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:3px}

/* ---- the settler ------------------------------------------------------- */
.settle{margin:34px 0 0;padding:0;overflow:hidden}
.pickers{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;padding:20px 22px;
  border-bottom:1px solid var(--line);background:var(--paper)}
.pick{flex:1;min-width:170px;position:relative}
/* A real listbox. The native <datalist> this replaces rendered as browser
   chrome rather than DOM: nothing opened on click, and on mobile it never
   appeared at all, so the field could only be typed into. */
.opts{position:absolute;top:calc(100% + 4px);left:0;right:0;z-index:40;
  background:var(--panel);border:1px solid var(--line);border-radius:3px;
  max-height:302px;overflow-y:auto;box-shadow:0 8px 26px rgba(20,18,12,.16)}
.opt{display:flex;align-items:baseline;gap:8px;padding:9px 13px;cursor:pointer;
  border-bottom:1px solid var(--hair);font:14px var(--sans)}
.opt:last-child{border-bottom:0}
.opt:hover,.opt.on{background:var(--accent-wash)}
.opt .on-nm{font-weight:600;color:var(--ink)}
.opt .on-c{color:var(--muted);font:600 10px var(--sans);letter-spacing:.09em}
.opt .on-p{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums;
  font-size:13px}
.opt mark{background:none;color:var(--accent-ink);font-weight:700}
.opts .none{padding:14px;color:var(--muted);font:14px var(--serif)}
.pick label,.ctl label{display:block;font:600 10.5px/1 var(--sans);text-transform:uppercase;
  letter-spacing:.13em;color:var(--muted);margin-bottom:7px}
.pick input{width:100%}
.vs{color:var(--muted);font:italic 17px var(--serif);padding-bottom:9px}
.verdict{padding:22px}
.vhead{font:400 27px/1.2 var(--serif);letter-spacing:-.015em;margin:0 0 6px}
.vsub{color:var(--muted);font-size:15px;margin:0 0 20px;max-width:44em}
.vrow{display:flex;gap:16px;flex-wrap:wrap}
.vcard{flex:1;min-width:210px;padding:14px 16px 15px;border-left:3px solid var(--line)}
.vcard b{display:block;font:600 15px var(--sans);margin-bottom:6px;letter-spacing:.01em}
.vcard .big{font:600 44px/1 var(--sans);font-variant-numeric:tabular-nums;
  letter-spacing:-.03em}
.vcard small{color:var(--muted);font:12.5px/1.5 var(--sans);display:block;margin-top:8px}
.win{border-left-color:var(--accent)}
.win .big{color:var(--accent-ink)}
.tie{padding:13px 16px;border-left:3px solid var(--accent);background:var(--accent-wash);
  font-size:14.5px;margin-top:18px;border-radius:0 3px 3px 0}
/* Both intervals on one scale: the actual argument, drawn. */
.cmp{margin-top:20px;padding-top:18px;border-top:1px solid var(--hair)}
.cmprow{display:grid;grid-template-columns:minmax(96px,auto) 1fr;gap:14px;
  align-items:center;margin-bottom:11px}
.cmprow span{font:600 12px var(--sans);color:var(--muted);text-align:right;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cmptrack{position:relative;height:22px}
.cmptrack::before{content:"";position:absolute;left:0;right:0;top:50%;height:1px;
  background:var(--hair)}
.cmpband{position:absolute;top:6px;height:10px;background:var(--bar);opacity:.5;
  border-radius:2px}
.cmprow.w .cmpband{background:var(--accent);opacity:.55}
.cmpdot{position:absolute;top:2px;width:3px;height:18px;background:var(--ink);
  border-radius:1px;transform:translateX(-1.5px)}
.cmprow.w .cmpdot{background:var(--accent-ink)}

/* ---- controls ---------------------------------------------------------- */
input,select{background:var(--paper);border:1px solid var(--line);color:var(--ink);
  padding:9px 12px;border-radius:3px;font:15px var(--sans)}
input::placeholder{color:var(--muted);opacity:.75}
input:focus,select:focus{outline:2px solid var(--accent);outline-offset:-1px;
  border-color:var(--accent)}
select{appearance:none;-webkit-appearance:none;padding-right:32px;cursor:pointer;
  background-image:linear-gradient(45deg,transparent 50%,currentColor 50%),
    linear-gradient(135deg,currentColor 50%,transparent 50%);
  background-position:calc(100% - 16px) calc(50% + 1px),calc(100% - 11px) calc(50% + 1px);
  background-size:5px 5px,5px 5px;background-repeat:no-repeat}
.controls{display:flex;gap:10px;flex-wrap:wrap;padding:14px 16px;
  border-bottom:1px solid var(--line);align-items:center}
.controls input{flex:1;min-width:190px}

/* ---- era chart --------------------------------------------------------- */
.chart{padding:34px 22px 0 56px;position:relative}
.plot{position:relative;height:210px;display:flex;align-items:flex-end;gap:6px;
  border-bottom:1.5px solid var(--ink)}
/* The grid must outrank `.plot div` below, which sets position:relative. */
.plot .grid{position:absolute;left:0;right:0;top:0;bottom:0;pointer-events:none}
.grid i{position:absolute;left:0;right:0;height:1px;background:var(--hair)}
.grid u{position:absolute;left:0;transform:translate(-100%,-50%);padding-right:12px;
  font:11px var(--sans);font-variant-numeric:tabular-nums;color:var(--muted);
  text-decoration:none;line-height:1}
.plot div:not(.grid){flex:1;background:var(--bar);position:relative;
  border-radius:2px 2px 0 0;transition:background .12s}
.plot div:not(.grid):hover{background:var(--accent)}
.plot div:not(.grid)::after{content:attr(data-v);position:absolute;top:-19px;left:0;right:0;
  text-align:center;font:600 11px var(--sans);font-variant-numeric:tabular-nums;
  color:var(--muted);opacity:0;transition:opacity .12s}
.plot div:not(.grid):hover::after{opacity:1}
.ylab{position:absolute;left:56px;top:10px;font:600 10.5px var(--sans);
  text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
.blabels{display:flex;gap:6px;padding:9px 22px 20px 56px;color:var(--muted);
  font:11px var(--sans);font-variant-numeric:tabular-nums}
.blabels span{flex:1;text-align:center}

/* ---- tabs -------------------------------------------------------------- */
.tabs{display:flex;gap:0;flex-wrap:wrap;padding:0 16px;border-bottom:1px solid var(--line);
  background:var(--paper)}
.tabs button{background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);
  padding:13px 14px;font:600 13px var(--sans);cursor:pointer;margin-bottom:-1px}
.tabs button:hover{color:var(--ink)}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}

/* ---- table ------------------------------------------------------------- */
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font:14px var(--sans)}
th,td{padding:10px 12px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--hair)}
th{position:sticky;top:0;background:var(--panel);color:var(--muted);
  font:600 10.5px var(--sans);text-transform:uppercase;letter-spacing:.12em;
  cursor:pointer;user-select:none;border-bottom:1px solid var(--ink);z-index:2}
th:hover{color:var(--ink)}
th.sorted{color:var(--ink)}
th.sorted::after{content:" \\2193";font-size:11px}
th.sorted.asc::after{content:" \\2191"}
th:first-child,td:first-child{text-align:right;color:var(--muted);
  font-variant-numeric:tabular-nums;padding-left:22px;width:1%}
th:nth-child(2),td:nth-child(2){text-align:left}
td:last-child{padding-right:22px}
td{font-variant-numeric:tabular-nums;color:var(--muted)}
tbody tr:hover{background:var(--accent-wash)}
.nm{font-weight:600;color:var(--ink);font-size:14.5px}
.ctry{color:var(--muted);font:600 10.5px var(--sans);letter-spacing:.09em;margin-left:7px}
.yr{color:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums;margin-top:1px}
.pv{font-weight:700;color:var(--ink);font-size:15px}
.citrack{position:relative;display:inline-block;width:150px;height:16px;
  vertical-align:middle}
.citrack::before{content:"";position:absolute;left:0;right:0;top:50%;height:1px;
  background:var(--hair)}
.ci{position:absolute;top:6px;height:4px;background:var(--bar);border-radius:2px}
.ci i{position:absolute;top:-4px;width:2px;height:12px;background:var(--ink);
  border-radius:1px}
tbody tr:hover .ci{background:var(--accent)}
.up{color:var(--turf);font-weight:600} .down{color:var(--ball);font-weight:600}
.empty{text-align:center;padding:44px 20px;color:var(--muted);font:16px var(--serif)}

/* ---- footer ------------------------------------------------------------ */
footer{margin-top:66px;padding-top:24px;border-top:3px double var(--line);
  color:var(--muted);font-size:14.5px;line-height:1.65;max-width:44em}
footer b{color:var(--ink)}
code{font:12.5px var(--mono);background:var(--accent-wash);color:var(--accent-ink);
  padding:2px 6px;border-radius:2px}
@media (max-width:640px){
  body{padding:0 16px 60px}
  .stat{border-left:0;padding:12px 0}
  .stats{border-bottom-width:1px}
  /* Stack the pickers. Side by side they squeeze to nothing on a phone. */
  .pickers{padding:16px}
  .pick{flex:1 1 100%;min-width:0}
  .vs{flex:1 1 100%;text-align:center;padding:0}
  #pfmt{width:100%}
  .verdict{padding:16px}
  .vcard{min-width:0}
}
"""


def render(payload: dict) -> str:
    m = payload["meta"]
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Howzat: Era-Adjusted Cricket Ratings</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow"><b>Howzat</b> Era-adjusted cricket ratings</p>
  <h1>Settle <em>it.</em></h1>
  <p class="sub">Pick two players. You get a verdict and the number behind it. When the
  data cannot separate them it says so. Every comparison is <b>era-adjusted</b>, so
  1930 and 2026 sit on one scale.</p>
  <div class="stats">
    <div class="stat"><b>{m['all_innings']:,}</b><span>innings analysed</span></div>
    <div class="stat"><b>{m['span']}</b><span>every Test ever</span></div>
    <div class="stat"><b>{m['views']}</b><span>rating tables</span></div>
    <div class="stat"><b>{m['total_players']:,}</b><span>Test players</span></div>
  </div>
</header>

<div class="panel settle">
  <div class="pickers">
    <div class="pick">
      <label for="pa">Player A</label>
      <input id="pa" placeholder="Bradman" autocomplete="off" role="combobox"
             aria-expanded="false" aria-autocomplete="list" aria-controls="pa-opts">
      <div class="opts" id="pa-opts" role="listbox" hidden></div>
    </div>
    <div class="vs">vs</div>
    <div class="pick">
      <label for="pb">Player B</label>
      <input id="pb" placeholder="Tendulkar" autocomplete="off" role="combobox"
             aria-expanded="false" aria-autocomplete="list" aria-controls="pb-opts">
      <div class="opts" id="pb-opts" role="listbox" hidden></div>
    </div>
    <select id="pfmt"></select>
  </div>
  <div id="verdict" class="verdict"></div>
</div>

<h2>How hard was it to score?</h2>
<p class="note">Runs per dismissal by decade, across every Test batter. The easiest decade
to bat in was <b>1.96×</b> the hardest. Raw batting average ignores all of that. CRI+
divides it out.</p>
<div class="panel">
  <div class="chart">
    <div class="ylab">Runs per dismissal</div>
    <div class="plot" id="bars"><div class="grid" id="grid"></div></div>
  </div>
  <div class="blabels" id="blabels"></div>
</div>

<h2>The ratings</h2>
<p class="note">Bars show the <b>95% interval</b>. Short careers get wide bars. That is
deliberate. <code>Δ</code> counts the places a batter moves against raw batting average.
Positive means the era adjustment promoted them. Click any column to sort.</p>
<div class="panel">
  <div class="tabs" id="tabs"></div>
  <div class="controls">
    <input id="q" placeholder="Search a player…" autocomplete="off">
    <select id="minInns">
      <option value="5">5+ innings</option>
      <option value="10">10+ innings</option>
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
  <b>Method.</b> Innings runs are modelled as Geometric with a log link. Not-outs enter as
  right-censored observations instead of being averaged in. Player skill carries a Gaussian
  prior whose strength is set by empirical Bayes (λ = 1.90), so short careers shrink toward
  the mean. The era term is <i>fixed from observed scoring</i> rather than fitted. Estimated
  freely it came out anti-correlated with reality (r = −0.21). Only 12.7% of players span
  three decades, so the likelihood cannot tell era and skill apart. Fitted by penalised MLE
  with analytic gradients.
  <br><br>
  CRI+ measures <b>dominance over contemporaries</b>. It does not measure absolute skill
  across eras. That is the same claim baseball's <code>+</code> metrics make and the
  strongest thing this data supports. Data: ESPNcricinfo Statsguru.
</footer>
</div>

<script>
const VIEWS = {json.dumps(payload["views"], separators=(",", ":"))};
let VIEW = Object.keys(VIEWS)[0];
let DATA = VIEWS[VIEW].rows;
const ERAS = {json.dumps(payload["eras"], separators=(",", ":"))};

const bars = document.getElementById('bars'), blab = document.getElementById('blabels');
// Bars scaled to the largest value make every chart look the same. Scale to a
// round number above the max instead, and label the axis, so the 1.96x spread
// is something you can read off rather than take on trust.
const mx = Math.max(...ERAS.map(e => e.avg));
const step = mx > 60 ? 20 : mx > 30 ? 10 : 5;
// Not `top`: window.top already owns that name and redeclaring it kills the script.
const axisTop = Math.ceil(mx / step) * step;
const grid = document.getElementById('grid');
for (let v = 0; v <= axisTop; v += step) {{
  const y = 100 - v / axisTop * 100;
  const line = document.createElement('i'); line.style.top = y + '%';
  const lab = document.createElement('u'); lab.style.top = y + '%'; lab.textContent = v;
  grid.append(line, lab);
}}
ERAS.forEach(e => {{
  const d = document.createElement('div');
  d.style.height = (e.avg / axisTop * 100) + '%';
  d.dataset.v = e.avg;
  d.title = e.d + 's: ' + e.avg + ' runs per dismissal (' + e.n.toLocaleString() + ' innings)';
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
  const sel2 = document.getElementById('minInns');
  const want = VIEWS[k].min_default;
  if (want !== undefined && !bowl) {{
    const opts = [...sel2.options].map(o => +o.value);
    let best = opts[0];
    for (const o of opts) if (o <= want) best = o;
    sel2.value = String(best);
  }}
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
    let c = typeof x === 'string' ? x.localeCompare(y) : x - y;
    // Lower is better for a bowling average and economy, so descending on
    // those columns must mean "best first", not "biggest number first".
    if (bowl && (sortK === 'a' || sortK === 'sr')) c = -c;
    return sortAsc ? c : -c;
  }});

  document.querySelectorAll('th[data-k]').forEach(th => {{
    th.classList.toggle('sorted', th.dataset.k === sortK);
    th.classList.toggle('asc', th.dataset.k === sortK && sortAsc);
  }});

  const body = document.getElementById('rows');
  if (!rows.length) {{
    body.innerHTML = `<tr><td colspan="8" class="empty">No players match those filters in
      ${{VIEWS[VIEW].label}}. Clear the search or lower the innings filter.</td></tr>`;
    return;
  }}
  body.innerHTML = rows.slice(0, 300).map(p => {{
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

// ---- the argument settler -------------------------------------------------
const ALIASES = {json.dumps(payload.get("aliases", {}), separators=(",", ":"))};
const BATVIEWS = Object.entries(VIEWS).filter(([k, v]) => v.kind === 'bat');
const pf = document.getElementById('pfmt');
BATVIEWS.forEach(([k, v]) => {{
  const o = document.createElement('option'); o.value = k; o.textContent = v.label;
  pf.appendChild(o);
}});

function findPlayer(q, key) {{
  const rows = VIEWS[key].rows;
  let s = q.trim().toLowerCase();
  if (!s) return null;
  if (ALIASES[s]) s = ALIASES[s].toLowerCase();
  return rows.find(p => p.n.toLowerCase() === s)
      || rows.find(p => p.n.toLowerCase().split(/[^a-z]+/).includes(s))
      || rows.find(p => p.n.toLowerCase().includes(s)) || null;
}}

function card(p, won) {{
  return `<div class="vcard ${{won ? 'win' : ''}}">
    <b>${{p.n}} <span class="ctry">${{p.c}}</span></b>
    <div class="big">${{p.p.toFixed(0)}}</div>
    <small>95% interval ${{p.lo.toFixed(0)}}–${{p.hi.toFixed(0)}} ·
    ${{p.i}} inns · avg ${{p.a.toFixed(1)}} · rank #${{p.r}}</small>
  </div>`;
}}

// The overlap is the whole claim, so draw it on one shared scale rather than
// leaving the reader to compare two pairs of numbers in a sentence.
function intervals(hi, lo, separated) {{
  const a = Math.min(hi.lo, lo.lo), b = Math.max(hi.hi, lo.hi);
  const pad = (b - a) * 0.08 || 1;
  const L = a - pad, R = b + pad;
  const at = v => (v - L) / (R - L) * 100;
  const row = (p, won) => `<div class="cmprow ${{won ? 'w' : ''}}">
      <span>${{p.n}}</span>
      <div class="cmptrack" title="95% interval ${{p.lo.toFixed(0)}}\\u2013${{p.hi.toFixed(0)}}">
        <div class="cmpband" style="left:${{at(p.lo).toFixed(2)}}%;
             width:${{(at(p.hi) - at(p.lo)).toFixed(2)}}%"></div>
        <div class="cmpdot" style="left:${{at(p.p).toFixed(2)}}%"></div>
      </div></div>`;
  return `<div class="cmp">${{row(hi, separated)}}${{row(lo, false)}}</div>`;
}}

function settle() {{
  const key = pf.value || BATVIEWS[0][0];
  const a = findPlayer(document.getElementById('pa').value, key);
  const b = findPlayer(document.getElementById('pb').value, key);
  const out = document.getElementById('verdict');
  const qa = document.getElementById('pa').value.trim();
  const qb = document.getElementById('pb').value.trim();
  if (!qa || !qb) {{ out.innerHTML = ''; return; }}
  // Saying nothing when a name does not resolve reads as a broken page.
  const missing = [!a && qa, !b && qb].filter(Boolean);
  if (missing.length) {{
    out.innerHTML = `<p class="vsub">No ${{VIEWS[key].label}} player matching
      <b>${{missing.map(m => m.replace(/[<>]/g, '')).join('</b> or <b>')}}</b>.
      Try a surname, or pick from the suggestions.</p>`;
    return;
  }}
  if (a.n === b.n) {{ out.innerHTML = '<p class="vsub">Pick two different players.</p>'; return; }}

  const [hi, lo] = a.p >= b.p ? [a, b] : [b, a];
  // The whole point: only call a winner when the intervals do not overlap.
  const separated = hi.lo > lo.hi;
  const head = separated
    ? `${{hi.n}}. Not close.`
    : `Too close to call.`;
  const sub = separated
    ? `${{hi.n}}'s 95% interval (${{hi.lo.toFixed(0)}}–${{hi.hi.toFixed(0)}}) clears
       ${{lo.n}}'s entirely (${{lo.lo.toFixed(0)}}–${{lo.hi.toFixed(0)}}). The gap is real.`
    : `Their 95% intervals overlap. This data cannot separate them.
       ${{hi.n}} rates higher but not by enough to be sure.`;
  out.innerHTML = `<p class="vhead">${{head}}</p><p class="vsub">${{sub}}</p>
    <div class="vrow">${{card(hi, separated)}}${{card(lo, false)}}</div>` +
    intervals(hi, lo, separated) +
    (separated ? '' : `<div class="tie">An overlap is a result. The evidence cannot
       separate these two, and saying so is more honest than inventing a ranking.</div>`);
}}

// ---- player picker --------------------------------------------------------
const esc = s => s.replace(/[&<>"]/g, c => (
  {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}}[c]));

function combobox(input, box) {{
  let items = [], active = -1;

  const rowsFor = q => {{
    const all = VIEWS[pf.value || BATVIEWS[0][0]].rows;
    if (!q) return all.slice(0, 80);
    const s = q.toLowerCase();
    // Surname first, then anything containing the query, so typing "ten"
    // reaches Tendulkar before it reaches players merely containing "ten".
    const starts = [], has = [];
    for (const p of all) {{
      const n = p.n.toLowerCase();
      if (n.split(/[^a-z]+/).some(w => w.startsWith(s))) starts.push(p);
      else if (n.includes(s)) has.push(p);
    }}
    return starts.concat(has).slice(0, 80);
  }};

  const mark = (name, q) => {{
    if (!q) return esc(name);
    const i = name.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return esc(name);
    return esc(name.slice(0, i)) + '<mark>' + esc(name.slice(i, i + q.length)) +
           '</mark>' + esc(name.slice(i + q.length));
  }};

  // typing filters by what is typed, but clicking a field that already holds a
  // chosen player shows the whole list, otherwise you could only ever see the
  // name already in the box and never browse to a different one.
  function open(typing) {{
    const raw = input.value.trim();
    const all = VIEWS[pf.value || BATVIEWS[0][0]].rows;
    const committed = !typing && all.some(p => p.n.toLowerCase() === raw.toLowerCase());
    const q = committed ? '' : raw;
    items = rowsFor(q);
    if (!items.length) {{
      box.innerHTML = '<div class="none">No player of that name in this table.</div>';
    }} else {{
      box.innerHTML = items.map((p, i) => `<div class="opt" role="option" data-i="${{i}}">
        <span class="on-nm">${{mark(p.n, q)}}</span><span class="on-c">${{esc(p.c)}}</span>
        <span class="on-p">${{p.p.toFixed(0)}}</span></div>`).join('');
    }}
    box.hidden = false; input.setAttribute('aria-expanded', 'true');
    active = -1; box.scrollTop = 0;
  }}

  function close() {{
    box.hidden = true; input.setAttribute('aria-expanded', 'false'); active = -1;
  }}

  function choose(i) {{
    if (!items[i]) return;
    input.value = items[i].n;
    close(); settle();
  }}

  function highlight(n) {{
    const els = [...box.querySelectorAll('.opt')];
    if (!els.length) return;
    active = (n + els.length) % els.length;
    els.forEach((e, i) => e.classList.toggle('on', i === active));
    els[active].scrollIntoView({{block: 'nearest'}});
  }}

  input.addEventListener('focus', () => open(false));
  input.addEventListener('click', () => open(false));
  input.addEventListener('input', () => {{ open(true); settle(); }});
  // mousedown, not click: it fires before the input's blur closes the list.
  box.addEventListener('mousedown', e => {{
    const row = e.target.closest('.opt');
    if (row) {{ e.preventDefault(); choose(+row.dataset.i); }}
  }});
  input.addEventListener('blur', () => setTimeout(close, 120));
  input.addEventListener('keydown', e => {{
    if (e.key === 'ArrowDown') {{ e.preventDefault(); if (box.hidden) open(false); highlight(active + 1); }}
    else if (e.key === 'ArrowUp') {{ e.preventDefault(); highlight(active - 1); }}
    else if (e.key === 'Enter') {{ if (active >= 0) {{ e.preventDefault(); choose(active); }} else close(); }}
    else if (e.key === 'Escape') close();
  }});
}}

combobox(document.getElementById('pa'), document.getElementById('pa-opts'));
combobox(document.getElementById('pb'), document.getElementById('pb-opts'));
pf.addEventListener('change', settle);
// Seed with the resolved full names, not the surnames. The field should show
// the player actually selected, which is also what lets the picker tell a
// committed choice from something half-typed.
(() => {{
  const k = pf.value || BATVIEWS[0][0];
  const a = findPlayer('Bradman', k), b = findPlayer('Tendulkar', k);
  document.getElementById('pa').value = a ? a.n : 'Bradman';
  document.getElementById('pb').value = b ? b.n : 'Tendulkar';
}})();
settle();
</script>"""


def main() -> None:
    payload = build_payload()
    html = render(payload)
    kb = len(html.encode("utf-8")) / 1024
    for d in (OUT, DOCS):
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(html, encoding="utf-8")
        print(f"wrote {d / 'index.html'}")
    print(f"({kb:.0f} KB, {len(payload['players'])} players, no backend)")


if __name__ == "__main__":
    main()
