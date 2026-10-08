"""Print the Table 2 rows (Appendix C) from T10_OUT/t10_shells_summary.csv.
Mean ring size to one decimal and divergence to three, as before. T10_OUT is the folder of the outputs (default
data/derived/generator_benchmark/).

    python t10_shells_table.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.normpath(os.path.join(HERE, "..", ".."))          # demos/npjcompmat
OUT = os.environ.get("T10_OUT", os.path.join(COMPANION, "data", "derived", "generator_benchmark"))
S = pd.read_csv(os.path.join(OUT, "t10_shells_summary.csv")).set_index(["spec", "shells"])
PARTNERS = {1: 6, 2: 18, 3: 26, 4: 32, 5: 56, 6: 80, 8: 122}


def rows():
    out = []
    for k, z in PARTNERS.items():
        a, b = S.loc[("N20", k)], S.loc[("N100", k)]
        out.append(f"{k} & {z} & {a.mean_ring:.1f} & {a.js:.3f} & {b.mean_ring:.1f} & {b.js:.3f} \\\\")
    return "\n".join(out)


if __name__ == "__main__":
    print(rows())
