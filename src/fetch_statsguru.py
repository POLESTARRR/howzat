"""Fetch innings-by-innings Test batting records from ESPNcricinfo Statsguru.

Cricsheet ball-by-ball only reaches back to 2001, so it cannot support era
adjustment. Statsguru's `view=innings` list covers every Test innings since
1877, which is the substrate CRI+ needs.

Raw HTML is cached to data/raw/statsguru/ so a re-run never refetches.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "statsguru"
OUT = ROOT / "data" / "processed"

BASE = "https://stats.espncricinfo.com/ci/engine/stats/index.html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
DELAY = 1.5  # be polite

# Statsguru format codes and the year each format began.
FORMATS = {"test": (1, 1877), "odi": (2, 1971), "t20i": (3, 2005)}
FIRST_TEST_YEAR = 1877


def url_for(year: int, page: int, cls: int = 1, disc: str = "batting") -> str:
    return (
        f"{BASE}?class={cls};template=results;type={disc};view=innings"
        f";size=200;page={page}"
        f";spanmin1=01+Jan+{year};spanmax1=31+Dec+{year};spanval1=span"
    )


def fetch(year: int, page: int, cls: int = 1, disc: str = "batting", attempts: int = 5) -> str:
    """Return page HTML, from cache when present.

    Statsguru intermittently stalls, so retry with backoff rather than losing
    a whole run to one dropped connection.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    prefix = "" if cls == 1 else f"c{cls}_"
    if disc != "batting":
        prefix = f"{disc[:4]}_{prefix}"
    path = CACHE / f"{prefix}{year}_p{page}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")

    last = None
    for a in range(attempts):
        try:
            r = requests.get(url_for(year, page, cls, disc), headers=HEADERS, timeout=(15, 45))
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
            time.sleep(DELAY)
            return r.text
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            wait = 5 * (a + 1)
            print(f"  retry {year} p{page} in {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
    raise last  # type: ignore[misc]


# Statsguru marks a non-innings with these tokens in the Runs column.
NON_INNINGS = {"DNB", "TDNB", "absent", "sub", "-"}


def parse_page(html: str) -> list[dict]:
    """Extract innings rows from the one engineTable that has the innings header."""
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table", class_="engineTable"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if not (header[:2] == ["Player", "Runs"] and "Start Date" in header):
            continue

        out = []
        for tr in rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 12:
                continue
            player_raw, runs_raw = cells[0], cells[1]

            m = re.match(r"^(.*?)\s*\(([^)]*)\)$", player_raw)
            if not m:
                continue
            player, country = m.group(1).strip(), m.group(2).strip()

            if runs_raw in NON_INNINGS or not runs_raw:
                continue
            not_out = runs_raw.endswith("*")
            runs_txt = runs_raw.rstrip("*")
            if not runs_txt.isdigit():
                continue

            def num(v):
                return int(v) if v.isdigit() else None

            out.append(
                {
                    "player": player,
                    "country": country,
                    "runs": int(runs_txt),
                    "not_out": not_out,
                    "mins": num(cells[2]),
                    "balls_faced": num(cells[3]),
                    "fours": num(cells[4]),
                    "sixes": num(cells[5]),
                    "innings_no": num(cells[7]),
                    "opposition": cells[9].lstrip("v").strip(),
                    "ground": cells[10],
                    "start_date": cells[11],
                }
            )
        return out
    return []


def parse_bowling_page(html: str) -> list[dict]:
    """Extract bowling innings rows.

    Columns are Player, Overs, Mdns, Runs, Wkts, Econ, Inns, _, Opposition,
    Ground, Start Date. Overs are recorded in cricket notation (12.3 = twelve
    overs and three balls), which must be converted before it means anything.
    """
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table", class_="engineTable"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if not (header[:2] == ["Player", "Overs"] and "Start Date" in header):
            continue

        out = []
        for tr in rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 11:
                continue
            m = re.match(r"^(.*?)\s*\(([^)]*)\)$", cells[0])
            if not m:
                continue
            overs_raw = cells[1]
            if overs_raw in NON_INNINGS or not overs_raw:
                continue
            try:
                whole, _, part = overs_raw.partition(".")
                balls = int(whole) * 6 + (int(part) if part else 0)
            except ValueError:
                continue
            if balls <= 0:
                continue

            def num(v, cast=int):
                try:
                    return cast(v)
                except (ValueError, TypeError):
                    return None

            out.append({
                "player": m.group(1).strip(),
                "country": m.group(2).strip(),
                "balls": balls,
                "maidens": num(cells[2]),
                "runs_conceded": num(cells[3]),
                "wickets": num(cells[4]),
                "innings_no": num(cells[6]),
                "opposition": cells[8].lstrip("v").strip(),
                "ground": cells[9],
                "start_date": cells[10],
            })
        return out
    return []


def fetch_year(year: int, cls: int = 1, disc: str = "batting") -> list[dict]:
    """Page through a year until a page returns nothing."""
    rows: list[dict] = []
    for page in range(1, 60):
        try:
            raw = fetch(year, page, cls, disc)
            got = parse_page(raw) if disc == "batting" else parse_bowling_page(raw)
        except requests.HTTPError as e:
            print(f"  {year} p{page}: HTTP {e.response.status_code}", flush=True)
            break
        if not got:
            break
        rows.extend(got)
        if len(got) < 200:
            break
    return rows


def main(start: int, end: int, fmt: str = "test", disc: str = "batting") -> None:
    cls, _ = FORMATS[fmt]
    all_rows: list[dict] = []
    for year in range(start, end + 1):
        rows = fetch_year(year, cls, disc)
        all_rows.extend(rows)
        if rows:
            print(f"{year}: {len(rows):5d} innings (total {len(all_rows)})", flush=True)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no rows fetched")
        return

    df["start_date"] = pd.to_datetime(df["start_date"], format="%d %b %Y", errors="coerce")
    df["year"] = df["start_date"].dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    OUT.mkdir(parents=True, exist_ok=True)
    df["format"] = fmt
    suffix = "innings" if disc == "batting" else "bowling"
    dest = OUT / f"{fmt}_{suffix}.parquet"
    df.to_parquet(dest, index=False)
    print(f"\nwrote {len(df):,} innings -> {dest}")
    print(f"span {df.year.min()}-{df.year.max()}, {df.player.nunique():,} players")


if __name__ == "__main__":
    fmt = sys.argv[1] if len(sys.argv) > 1 else "test"
    _, first = FORMATS[fmt]
    a = int(sys.argv[2]) if len(sys.argv) > 2 else first
    b = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
    disc = sys.argv[4] if len(sys.argv) > 4 else "batting"
    main(a, b, fmt, disc)
