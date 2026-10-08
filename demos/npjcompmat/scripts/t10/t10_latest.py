"""Table 1 (Appendix B) rerun with the latest topon: both searches, in C and in Python, one code version for all columns.

Code:     main of the development repository at 54f2474 (the code of topon 0.4.5), exported with git archive to
          TOPON_SRC = <TOPON_BENCH>/src; the C generator built from the same source (gcc -O2) as
          <TOPON_BENCH>/generator.exe, see data/derived/generator_benchmark/t10_provenance.json.
Grid:     periodic SC lattices of 6^3, 8^3, 10^3 and 12^3 sites; one, two or three neighbor shells (6, 18 or 26
          candidate partners); targets A, T and B rescaled to the site count (TARGETS and rescale below, as in the
          t8 benchmark); max functionality 6.
Methods:  C_pruning      generator.exe ... --search=strict
          C_exact        generator.exe ... --search=exact --min-giant-fraction=1
          python_pruning PythonTopologyGenerator, search "strict", retried until a network or the cap
          python_exact   PythonTopologyGenerator, search "exact", min_giant_fraction 1, called again with new seeds
                         (seed*100003+k) until a network or the cap, as in the original runs
          Defaults of the current code otherwise (e.g., odd walks on in the exact search).
Attempts: 100 seeds per case, each limited to CAP = 120 s, connected network with the exact requested counts
          required (checked here independently, edge set hashed). A case whose first 10 attempts all fail is
          stopped there (reported as a dash, "none of the first 10").
Machine:  8 workers, each pinned to one performance core (logical CPUs 0, 2, ..., 14 of the i9-14900), nothing else
          heavy running. C times include starting the program and writing its files.
Setup:    TOPON_BENCH      folder with src/ (the topon source) and generator.exe, required
          T10_TMP          scratch folder of the C runs (default <TOPON_BENCH>/tmp)
          T10_OUT          folder of the outputs (default data/derived/generator_benchmark/ of this companion)
          T10_CORES        logical CPUs of the workers, comma-separated (default 0,2,4,6,8,10,12,14)

Writes T10_OUT/t10_timing.csv (one row per attempt, appended as it goes, so the run resumes) and
T10_OUT/t10_summary.csv.

    python t10_latest.py            (stage A = seeds 1-10 everywhere, then stage B = seeds 11-100 where A succeeded)
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BENCH_ROOT = os.environ.get("TOPON_BENCH")
if not BENCH_ROOT:
    raise SystemExit("Set TOPON_BENCH to the folder that holds src/ (the topon source to benchmark) and generator.exe"
                     " (see data/derived/generator_benchmark/README.md of the paper companion).")
BENCH_ROOT = os.path.abspath(BENCH_ROOT)
SRC = os.path.join(BENCH_ROOT, "src")
EXE = os.path.join(BENCH_ROOT, "generator.exe")
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True

import contextlib  # noqa: E402
import io  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

import topon  # noqa: E402
from topon.config.schema import GeneratorConfig  # noqa: E402
from topon.topology.generator_python import PythonTopologyGenerator  # noqa: E402

assert os.path.normcase(topon.__file__).startswith(os.path.normcase(SRC)), topon.__file__

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.normpath(os.path.join(HERE, "..", ".."))          # demos/npjcompmat
OUT = os.environ.get("T10_OUT", os.path.join(COMPANION, "data", "derived", "generator_benchmark"))
#: the graph metrics of the 327 deposited networks, read by the bias and degeneracy checks
DEPOSITED = os.path.join(COMPANION, "data", "derived", "generator_benchmark", "t0_deposited_graphs.csv")
TMP = os.environ.get("T10_TMP", os.path.join(BENCH_ROOT, "tmp"))
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
CAP = 120.0
N_SEEDS = 100
EARLY = 10
SIZES = [6, 8, 10, 12]
SHELLS = {1: 1.0, 2: 1.42, 3: 1.74}
PARTNERS = {1: 6, 2: 18, 3: 26}
TARGETS = {"A": [10, 0, 26, 75, 43, 53, 9],
           "T": [14, 23, 23, 20, 75, 41, 20],
           "B": [10, 44, 16, 35, 44, 13, 54]}
METHODS = ["C_pruning", "python_pruning", "C_exact", "python_exact"]


def cpu_list(var, default):
    """Logical CPUs to pin the workers to: the comma-separated list in environment variable var, else default."""
    value = os.environ.get(var, "").strip()
    return [int(c) for c in value.split(",")] if value else default


CORES = cpu_list("T10_CORES", [0, 2, 4, 6, 8, 10, 12, 14])
TIMING = os.path.join(OUT, "t10_timing.csv")
_core = None


def rescale(target, n):
    p = np.asarray(target, float) / 216.0
    c = np.rint(p * n).astype(int)
    c[0] = 0
    c[0] = n - c[1:].sum()
    if (c * np.arange(7)).sum() % 2:
        c[2] -= 1
        c[3] += 1
    assert c.sum() == n and c.min() >= 0 and (c * np.arange(7)).sum() % 2 == 0
    return c


def spec_of(counts):
    return ",".join(f"{d}:{int(c)}" for d, c in enumerate(counts))


def check_and_hash(G, target):
    deg = np.bincount([d for _, d in G.degree()], minlength=7)[:7]
    active = [v for v, d in G.degree() if d > 0]
    ok = bool((deg == np.asarray(target)).all()) and nx.is_connected(G.subgraph(active))
    edges = sorted(tuple(sorted((int(a), int(b)))) for a, b in G.edges())
    return ok, hashlib.md5(repr(edges).encode()).hexdigest()


def pin(queue):
    global _core
    _core = queue.get()
    psutil.Process().cpu_affinity([_core])


def read_graph(out):
    files = [f for f in os.listdir(out) if f.endswith(".edges")] if os.path.isdir(out) else []
    if not files:
        return None
    G = nx.Graph()
    with open(os.path.join(out, files[0][:-6] + ".nodes")) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                G.add_node(int(line.split()[0]))
    with open(os.path.join(out, files[0])) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                a, b = line.split()[:2]
                G.add_edge(int(a), int(b))
    return G


def run_c(search, size, shells, target, seed):
    d = tempfile.mkdtemp(prefix="t10c_", dir=TMP)
    out = os.path.join(d, "output")
    cmd = [EXE, f"{size}x{size}x{size}", "111", "6", "100000000", "1", spec_of(target), "0", "SC",
           f"{SHELLS[shells]}", f"--search={search}", "--min-giant-fraction=1", f"--seed={seed}", f"--output-dir={out}"]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if _core is not None:
            psutil.Process(p.pid).cpu_affinity([_core])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        p.wait(timeout=CAP)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    wall = time.perf_counter() - t0
    G = read_graph(out)
    shutil.rmtree(d, ignore_errors=True)
    return G, wall


def py_config(search, size, shells, target, seed):
    kw = dict(lattice_size=f"{size}x{size}x{size}", lattice_type="SC", periodicity="111", max_functionality=6,
              degree_distribution=spec_of(target), search=search, seed=int(seed))
    if shells > 1:
        kw["neighbour_shells"] = shells
    if search == "exact":
        kw["min_giant_fraction"] = 1.0
    return GeneratorConfig(**kw)


def gen_once(cfg, time_limit):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            graphs = PythonTopologyGenerator(cfg).generate(trials=10**9, max_saves=1, time_limit=time_limit)
        except Exception:  # noqa: BLE001 - a refusal or a give-up counts as no network
            graphs = []
    return graphs[0] if graphs else None


def run_py(search, size, shells, target, seed):
    t0 = time.perf_counter()
    if search == "strict":
        g = gen_once(py_config(search, size, shells, target, seed), CAP)
        wall = time.perf_counter() - t0
        return (nx.Graph(g) if g is not None and wall <= CAP * 1.05 else None), wall
    k = 0
    while True:   # the exact search gives up after its own attempts, so it gets new seeds until the cap
        remaining = CAP - (time.perf_counter() - t0)
        g = gen_once(py_config(search, size, shells, target, seed * 100003 + k), max(remaining, 1.0))
        k += 1
        wall = time.perf_counter() - t0
        if g is not None or wall >= CAP:
            return (nx.Graph(g) if g is not None and wall <= CAP * 1.05 else None), wall


def one(job):
    method, size, shells, tname, seed = job
    target = rescale(TARGETS[tname], size ** 3)
    search = "strict" if method.endswith("pruning") else "exact"
    G, wall = (run_c if method.startswith("C_") else run_py)(search, size, shells, target, seed)
    ok, h = check_and_hash(G, target) if G is not None else (False, "")
    return dict(method=method, size=size, sites=size ** 3, shells=shells, partners=PARTNERS[shells], target=tname,
                seed=seed, success=bool(G is not None and ok), returned=G is not None, wall_s=round(wall, 6),
                hash=h, core=_core)


def summarize(df):
    rows = []
    for (m, s, p, t), x in df.groupby(["method", "sites", "partners", "target"]):
        ok = x[x.success]
        rows.append(dict(method=m, sites=s, partners=p, target=t, attempts=len(x), successes=len(ok),
                         median_s=ok.wall_s.median() if len(ok) else np.nan,
                         q25_s=ok.wall_s.quantile(0.25) if len(ok) else np.nan,
                         q75_s=ok.wall_s.quantile(0.75) if len(ok) else np.nan,
                         p90_s=ok.wall_s.quantile(0.9) if len(ok) else np.nan,
                         distinct=ok.hash.nunique()))
    return pd.DataFrame(rows)


def run_jobs(jobs, ex, done):
    futures = [ex.submit(one, j) for j in jobs if j not in done]
    t0 = time.time()
    for i, f in enumerate(as_completed(futures), 1):
        r = f.result()
        pd.DataFrame([r]).to_csv(TIMING, mode="a", header=not os.path.exists(TIMING), index=False)
        done.add((r["method"], r["size"], r["shells"], r["target"], r["seed"]))
        if i % 100 == 0 or i == len(futures):
            print(f"  {i}/{len(futures)} in {time.time() - t0:7.0f} s", flush=True)


def main():
    import multiprocessing as mp
    done = set()
    if os.path.exists(TIMING):
        d0 = pd.read_csv(TIMING)
        done = set(zip(d0.method, d0["size"], d0.shells, d0.target, d0.seed))
        print(f"resuming: {len(done)} attempts already recorded")
    cases = [(m, s, k, t) for s in SIZES for k in SHELLS for t in TARGETS for m in METHODS]
    q = mp.Manager().Queue()
    for c in CORES:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(CORES), initializer=pin, initargs=(q,)) as ex:
        print("stage A: seeds 1-10 in every case", flush=True)
        # big lattices first, so the long attempts do not all land at the end
        stage_a = [(*c, seed) for c in sorted(cases, key=lambda c: -c[1]) for seed in range(1, EARLY + 1)]
        run_jobs(stage_a, ex, done)
        df = pd.read_csv(TIMING)
        first = df[df.seed <= EARLY].groupby(["method", "size", "shells", "target"]).success.agg(["sum", "count"])
        go = [c for c in cases if first.loc[c, "sum"] > 0]
        stopped = [c for c in cases if first.loc[c, "sum"] == 0]
        print(f"stage B: {len(go)} cases continue to {N_SEEDS} seeds, {len(stopped)} stopped after {EARLY}", flush=True)
        # the slowest cases (fewest successes in stage A) first
        go.sort(key=lambda c: first.loc[c, "sum"])
        stage_b = [(*c, seed) for c in go for seed in range(EARLY + 1, N_SEEDS + 1)]
        run_jobs(stage_b, ex, done)
    df = pd.read_csv(TIMING).drop_duplicates(["method", "size", "shells", "target", "seed"], keep="last")
    summ = summarize(df)
    summ.to_csv(os.path.join(OUT, "t10_summary.csv"), index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
