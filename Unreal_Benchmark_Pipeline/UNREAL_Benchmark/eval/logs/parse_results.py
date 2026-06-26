#!/usr/bin/env python3
"""Parse eval logs in this directory and write results.csv."""
import os
import re
import csv
import glob

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(LOG_DIR, "results.csv")

# Match "X.X% (n/N)" capturing percentage and counts.
PCT_RE = re.compile(r"(\d+\.\d+)%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)")

# Header lines that introduce sections.
SECTION_RES = {
    "domain": re.compile(r"^\s*By\s+[Dd]omain\s*:\s*$"),
    "subdomain": re.compile(r"^\s*By\s+[Ss]ubdomain\s*:\s*$"),
    "qtype": re.compile(r"^\s*By\s+qtype\s*:\s*$"),
}

# Item line in a section: "    Name   :  XX.X% (n/N) ..." or "    Name   XX.X% (n/N) ..."
ITEM_RE = re.compile(r"^\s{2,}(\S[^:%]*?)\s*:?\s+(\d+\.\d+)%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)")


def model_name_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    # strip trailing _YYYYMMDD_HHMMSS.log
    return re.sub(r"_\d{8}_\d{6}\.log$", "", base)


def parse_log(path: str):
    """Return dict {Overall, By qtype/domain/subdomain dicts}."""
    with open(path, "r", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return None

    result = {"overall_pct": None, "overall_n": None, "overall_N": None,
              "qtype": {}, "domain": {}, "subdomain": {}}

    lines = text.splitlines()
    # Find overall line: closed-source "Overall: X% (n/N)" or open "ModelName: X% (n/N) errors=..."
    for i, line in enumerate(lines):
        s = line.strip()
        # closed-source style
        m = re.match(r"^Overall:\s*(\d+\.\d+)%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)", s)
        if m:
            result["overall_pct"] = float(m.group(1))
            result["overall_n"] = int(m.group(2))
            result["overall_N"] = int(m.group(3))
            break
        # open-source: "<ModelName>: XX.X% (n/N)  errors=N"
        m = re.match(r"^\S[^:]*:\s*(\d+\.\d+)%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*errors=", s)
        if m:
            result["overall_pct"] = float(m.group(1))
            result["overall_n"] = int(m.group(2))
            result["overall_N"] = int(m.group(3))
            break

    if result["overall_pct"] is None:
        return None

    # Now parse sections.
    cur = None
    for line in lines:
        for k, rx in SECTION_RES.items():
            if rx.match(line):
                cur = k
                break
        else:
            if cur:
                m = ITEM_RE.match(line)
                if m:
                    name = m.group(1).strip()
                    pct = float(m.group(2))
                    n = int(m.group(3))
                    N = int(m.group(4))
                    result[cur][name] = (pct, n, N)
                elif line.strip() == "" or line.strip().startswith("="):
                    cur = None
                elif not line.startswith(" "):
                    cur = None
        # if line matched a section header we already set cur; reset detection
        for k, rx in SECTION_RES.items():
            if rx.match(line):
                cur = k
                break

    return result


def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    parsed = {}
    for p in logs:
        r = parse_log(p)
        if r is None:
            print(f"[skip] {os.path.basename(p)} (no results)")
            continue
        parsed[model_name_from_filename(p)] = r

    # Collect column order
    domains, subdomains, qtypes = [], [], []
    seen_d, seen_s, seen_q = set(), set(), set()
    for r in parsed.values():
        for name in r["domain"]:
            if name not in seen_d:
                seen_d.add(name); domains.append(name)
        for name in r["subdomain"]:
            if name not in seen_s:
                seen_s.add(name); subdomains.append(name)
        for name in r["qtype"]:
            if name not in seen_q:
                seen_q.add(name); qtypes.append(name)

    header = ["model", "overall_pct", "overall_correct", "overall_total"]
    for q in qtypes:
        header.append(f"qtype:{q}")
    for d in domains:
        header.append(f"domain:{d}")
    for s in subdomains:
        header.append(f"subdomain:{s}")

    rows = []
    for model, r in sorted(parsed.items()):
        row = [model, r["overall_pct"], r["overall_n"], r["overall_N"]]
        for q in qtypes:
            v = r["qtype"].get(q)
            row.append(v[0] if v else "")
        for d in domains:
            v = r["domain"].get(d)
            row.append(v[0] if v else "")
        for s in subdomains:
            v = r["subdomain"].get(s)
            row.append(v[0] if v else "")
        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} with {len(rows)} models, {len(header)} columns")


if __name__ == "__main__":
    main()
