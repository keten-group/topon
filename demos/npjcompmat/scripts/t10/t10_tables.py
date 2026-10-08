"""Table 1 body (Appendix B) from T10_OUT/t10_summary.csv, in seconds with three significant digits.

A cell gives the median time per network in seconds over the successful attempts, the number of networks found out
of 100 in parentheses when it is below 100, and a dash when none of the first 10 attempts succeeded (the case was
then stopped). Writes tables_checks/table_timing_t10.tex of this companion and prints it with the numbers the
Appendix B text quotes. T10_OUT is the folder of the outputs (default data/derived/generator_benchmark/).

    python t10_tables.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.normpath(os.path.join(HERE, "..", ".."))          # demos/npjcompmat
OUT = os.environ.get("T10_OUT", os.path.join(COMPANION, "data", "derived", "generator_benchmark"))
MS = os.path.join(COMPANION, "tables_checks")
S = pd.read_csv(os.path.join(OUT, "t10_summary.csv"))
COLS = [("Pruning", [("C", "C_pruning"), ("Python", "python_pruning")]),
        ("Exact matching", [("C", "C_exact"), ("Python", "python_exact")])]
METHODS = [m for _, sub in COLS for _, m in sub]


def sig3(t):
    """Three significant digits in seconds, no exponent."""
    if t >= 100:
        return f"{t:.0f}"
    for lim, dec in ((10, 1), (1, 2), (0.1, 3), (0.01, 4), (0.001, 5)):
        if t >= lim:
            return f"{t:.{dec}f}"
    return f"{t:.6f}"


def cell(method, sites, partners, target):
    r = S[(S.method == method) & (S.sites == sites) & (S.partners == partners) & (S.target == target)]
    assert len(r) == 1, (method, sites, partners, target)
    r = r.iloc[0]
    if r.successes == 0:
        return "-- & "
    return sig3(r.median_s) + " & " + (f"({int(r.successes)})" if r.successes < 100 else "")


def body(sizes=(216, 512, 1000, 1728)):
    # each method: the time right-aligned, then the count in a narrow left-aligned column
    m2 = lambda t: r"\multicolumn{2}{c}{" + t + "}"  # noqa: E731
    lines = [r"\begin{tabular}{cc" + r"r@{\,}l" * 4 + "}", r"\hline",
             r" & & \multicolumn{8}{c}{Median time per network (s)} \\",
             r"\cline{3-10}",
             r" & & \multicolumn{4}{c}{Pruning} & \multicolumn{4}{c}{Exact matching} \\",
             r"\cline{3-6}\cline{7-10}",
             "Target & Partners & " + " & ".join(m2(x) for x in ("C", "Python", "C", "Python")) + r" \\"]
    for n in sizes:
        lines += [r"\hline", rf"\multicolumn{{10}}{{l}}{{\textit{{{n:,} sites}}}} \\"]
        for t in ["A", "T", "B"]:
            for k in [6, 18, 26]:
                row = [t if k == 6 else "", str(k)] + [cell(m, n, k, t) for m in METHODS]
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def facts():
    p = S.pivot_table(index=["sites", "partners", "target"], columns="method", values="median_s")
    n = S.pivot_table(index=["sites", "partners", "target"], columns="method", values="successes")
    print("cells with networks:", {m: int((n[m] > 0).sum()) for m in METHODS}, "of", len(n))
    print("cells at 100/100:", {m: int((n[m] == 100).sum()) for m in METHODS})
    print("C exact max median:", sig3(p.C_exact.max()), "| Python exact cells under 1 s:", int((p.python_exact < 1).sum()))
    for a, b in (("python_pruning", "C_pruning"), ("python_exact", "C_exact")):
        r = (p[a] / p[b]).dropna()
        print(f"{a}/{b}: min {r.min():.2g} median {r.median():.2g} max {r.max():.2g}")
        for s in (216, 1728):
            rs = r.xs(s, level="sites")
            print(f"   at {s} sites: {rs.min():.2g} to {rs.max():.2g}")
    c216 = S[(S.method.str.startswith("C_")) & (S.sites == 216) & (S.successes == 100)].median_s
    print("C medians at 216 sites (fully reached cells):", sig3(c216.min()), "to", sig3(c216.max()))
    print("distinct = successes everywhere:", bool((S.distinct == S.successes).all()))


if __name__ == "__main__":
    txt = body()
    with open(os.path.join(MS, "table_timing_t10.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt + "\n")
    print(txt)
    facts()
