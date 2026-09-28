"""Table 1 of the paper (Appendix B), the median time to generate one network, from data/derived/generator_benchmark/.

The pruning columns (C and Python) come from t8_final_summary.csv and the exact-matching columns (C and Python) from
t8_exact_v2w10_summary.csv, which holds the exact search with its fallback. The exact-matching rows of
t8_final_summary.csv are from the search before the fallback and are not used. A cell gives the median wall time per
network over the successful attempts, with the number of networks found out of 100 in parentheses when it is below
100, and a dash when none of the 100 attempts succeeded.

Writes the tabular of Table 1 to tables_checks/table_timing.tex and prints it. The paths are relative to this file, so
it runs from any folder.

    python scripts/make_timing_table.py
"""
from pathlib import Path

import pandas as pd

COMPANION = Path(__file__).resolve().parents[1]          # demos/npjcompmat
BENCH = COMPANION / "data" / "derived" / "generator_benchmark"
OUT = COMPANION / "tables_checks" / "table_timing.tex"


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


PRUNING = pd.read_csv(require(BENCH / "t8_final_summary.csv"))
EXACT = pd.read_csv(require(BENCH / "t8_exact_v2w10_summary.csv"))
S = pd.concat([PRUNING[PRUNING.method.isin(["C_pruning", "python_pruning"])],
               EXACT[EXACT.method.isin(["C_exact", "python_exact"])]], ignore_index=True)
COLS = [("Pruning", [("C", "C_pruning"), ("Python", "python_pruning")]),
        ("Exact matching", [("C", "C_exact"), ("Python", "python_exact")])]
METHODS = [m for _, sub in COLS for _, m in sub]


def fmt_time(t):
    if t < 0.1:
        return f"{t * 1000:.0f} ms"
    if t < 1:
        return f"{t:.2f} s"
    if t < 10:
        return f"{t:.1f} s"
    return f"{t:.0f} s"


def cell(method, sites, partners, target):
    r = S[(S.method == method) & (S.sites == sites) & (S.partners == partners) & (S.target == target)]
    if r.empty:
        return "n/a"
    r = r.iloc[0]
    if r.successes == 0:
        return "--"
    txt = fmt_time(r.median_s)
    if r.successes < 100:
        txt += f" ({int(r.successes)})"
    return txt


def body(sizes):
    ncol = 2 + len(METHODS)
    head1 = " & & " + " & ".join(rf"\multicolumn{{{len(sub)}}}{{c}}{{{lab}}}" for lab, sub in COLS) + r" \\"
    rules, c0 = [], 3
    for _, sub in COLS:
        rules.append(rf"\cline{{{c0}-{c0 + len(sub) - 1}}}")
        c0 += len(sub)
    head2 = "Target & Partners & " + " & ".join(lang for _, sub in COLS for lang, _ in sub) + r" \\"
    lines = [rf"\begin{{tabular}}{{cc{'c' * len(METHODS)}}}", r"\hline", head1, "".join(rules), head2]
    for n in sizes:
        lines += [r"\hline", rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{n:,} sites}}}} \\"]
        for t in ["A", "T", "B"]:
            for k in [6, 18, 26]:
                row = [t if k == 6 else "", str(k)] + [cell(m, n, k, t) for m in METHODS]
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(body([216, 512, 1000, 1728]) + "\n")
ok = S[S.successes > 0]
dup = ok[ok.distinct < ok.successes]
print("methods", METHODS, "| cases", len(S), "with successes", len(ok), "cases with repeats", len(dup))
if len(dup):
    print(dup[["method", "sites", "partners", "target", "successes", "distinct"]].to_string())
print(OUT.read_text(encoding="utf-8"))
