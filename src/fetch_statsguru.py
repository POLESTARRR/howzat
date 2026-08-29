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
        col = _header_map(header)
        if not {"Player", "Runs", "Start Date"} <= set(col):
            continue

        def cell(cells, name):
            i = col.get(name)
            return cells[i] if i is not None and i < len(cells) else ""

        out = []
        for tr in rows[1:]:
            tds = tr.find_all("td")
            cells = [td.get_text(strip=True) for td in tds]
            if len(cells) < len(header) - 1:
                continue
            player_raw, runs_raw = cell(cells, "Player"), cell(cells, "Runs")

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
                    "player_id": _player_id(tds[col.get("Player", 0)])
                    if col.get("Player", 0) < len(tds) else None,
                    "country": country,
                    "runs": int(runs_txt),
                    "not_out": not_out,
                    "mins": num(cell(cells, "Mins")),
                    "balls_faced": num(cell(cells, "BF")),
                    "fours": num(cell(cells, "4s")),
                    "sixes": num(cell(cells, "6s")),
                    "innings_no": num(cell(cells, "Inns")),
                    "opposition": cell(cells, "Opposition").lstrip("v").strip(),
                    "ground": cell(cells, "Ground"),
                    "start_date": cell(cells, "Start Date"),
                }
            )
        return out
    return []


def _player_id(td) -> str | None:
    """Statsguru's own player id, taken from the profile link.

    Names are not unique. Pakistan has fielded two "Imran Khan"s, and keying on
    name+country silently merged them into one 1971-2019 career with 391
    wickets instead of the great one's 362. The href carries a stable id, so
    use that as the identity and treat the name as a label.
    """
    a = td.find("a", href=True)
    if not a:
        return None
    m = re.search(r"/player/(\d+)", a["href"])
    return m.group(1) if m else None


def _header_map(header: list[str]) -> dict[str, int]:
    """Column name -> index. Statsguru's column set is not fixed."""
    return {h: i for i, h in enumerate(header) if h}


def parse_bowling_page(html: str) -> list[dict]:
    """Extract bowling innings rows, mapping columns BY NAME.

    Fixed positions silently corrupted three decades of data. In 8-ball-over
    eras Statsguru inserts a BPO (balls per over) column, shifting everything
    right by one, so `start_date` read the ground name, `to_datetime` returned
    NaT, and `dropna` deleted the row. 1946-1979 disappeared without an error
    -- taking Garry Sobers's 235 wickets with it.

    BPO is also load-bearing in its own right: Australian domestic and Test
    cricket used 8-ball overs until 1979, so "29.0 overs" is 232 balls there
    and 174 balls elsewhere.
    """
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table", class_="engineTable"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        col = _header_map(header)
        if not {"Player", "Overs", "Wkts", "Start Date"} <= set(col):
            continue

        def cell(cells, name):
            i = col.get(name)
            return cells[i] if i is not None and i < len(cells) else ""

        out = []
        for tr in rows[1:]:
            tds = tr.find_all("td")
            cells = [td.get_text(strip=True) for td in tds]
            if len(cells) < len(header) - 1:
                continue
            m = re.match(r"^(.*?)\s*\(([^)]*)\)$", cell(cells, "Player"))
            if not m:
                continue

            overs_raw = cell(cells, "Overs")
            if overs_raw in NON_INNINGS or not overs_raw:
                continue
            # Balls per over defaults to 6, but is stated explicitly when not.
            try:
                bpo = int(cell(cells, "BPO") or 6)
            except ValueError:
                bpo = 6
            try:
                whole, _, part = overs_raw.partition(".")
                balls = int(whole) * bpo + (int(part) if part else 0)
            except ValueError:
                continue
            if balls <= 0:
                continue

            def num(name, cast=int):
                try:
                    return cast(cell(cells, name))
                except (ValueError, TypeError):
                    return None

            pid_col = col.get("Player", 0)
            out.append({
                "player": m.group(1).strip(),
                "player_id": _player_id(tds[pid_col]) if pid_col < len(tds) else None,
                "country": m.group(2).strip(),
                "balls": balls,
                "balls_per_over": bpo,
                "maidens": num("Mdns"),
                "runs_conceded": num("Runs"),
                "wickets": num("Wkts"),
                "innings_no": num("Inns"),
                "opposition": cell(cells, "Opposition").lstrip("v").strip(),
                "ground": cell(cells, "Ground"),
                "start_date": cell(cells, "Start Date"),
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
