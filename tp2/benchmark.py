"""
Runs task1_oom.py, task2_chunking.py, task3_dask.py as subprocesses, samples their
RAM from the outside (RSS, including child processes), and produces:
  fig_peak_ram.png, fig_time.png, fig_ram_timeline.png, fig_dask_graph_*.png (if graphviz)
  results.json, results_table.tex, generated_vars.tex   (used by report.tex)

Usage (from the folder that contains tp2/dataset/itineraries.csv and the 3 scripts):
    python benchmark_and_plot.py
"""
import json
import os
import platform
import subprocess
import sys
import time
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil

CSV = "dataset/itineraries.csv"
TASKS = [
    ("Standard Pandas", "task1_oom.py", "OOM"),
    ("Pandas Chunking", "task2_chunking.py", "Chunk"),
    ("Dask DataFrame", "task3_dask.py", "Dask"),
]
INTERVAL = 0.1  # seconds


def tree_rss_mb(p):
    """RSS of a process + all its children, in MB."""
    try:
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total / 1024**2
    except psutil.Error:
        return None


def run_and_monitor(script):
    t0 = time.time()
    with tempfile.TemporaryFile(mode="w+") as out:
        proc = subprocess.Popen([sys.executable, script], stdout=out,
                                stderr=subprocess.STDOUT, text=True)
        ps = psutil.Process(proc.pid)
        ts, mem = [], []
        while proc.poll() is None:
            m = tree_rss_mb(ps)
            if m is not None:
                ts.append(time.time() - t0)
                mem.append(m)
            time.sleep(INTERVAL)
        out.seek(0)
        output = out.read()
    return {
        "wall_s": time.time() - t0,
        "peak_mb": max(mem) if mem else 0.0,
        "avg_mb": sum(mem) / len(mem) if mem else 0.0,
        "t": ts, "mem": mem,
        "returncode": proc.returncode,
        "stdout": output,
        "oom": "MemoryError" in output or "Killed" in output or proc.returncode in (-9, 137),
    }


def dask_graph_info():
    """Task counts + graph pictures (optimized vs not). Best-effort."""
    info = {"raw_tasks": "n/a", "opt_tasks": "n/a", "npart": "n/a"}
    try:
        import dask
        import dask.dataframe as dd
        ddf = dd.read_csv(CSV, usecols=["totalFare", "baseFare"], blocksize="128MB",
                          dtype={"totalFare": "float32", "baseFare": "float32"})
        info["npart"] = ddf.npartitions
        mean_fare, sum_fare = ddf["totalFare"].mean(), ddf["totalFare"].sum()
        info["raw_tasks"] = len(dict(mean_fare.__dask_graph__())) + len(dict(sum_fare.__dask_graph__()))
        opt_m, opt_s = dask.optimize(mean_fare, sum_fare)
        info["opt_tasks"] = len(dict(dask.base.collections_to_dsk([opt_m, opt_s])))
        # Pictures are drawn on a tiny 2-partition slice so they stay readable.
        small = ddf.partitions[:2]
        m, s = small["totalFare"].mean(), small["totalFare"].sum()
        for name, flag in (("raw", False), ("opt", True)):
            try:
                dask.visualize(m, s, filename=f"fig_dask_graph_{name}", format="png",
                               optimize_graph=flag)
            except Exception as e:  # graphviz not installed
                print(f"[graph] could not draw {name}: {e}")
    except Exception as e:
        print("[graph] skipped:", e)
    return info


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def main():
    if not os.path.exists(CSV):
        sys.exit(f"Cannot find {CSV}; run from the right folder.")
    size_gb = os.path.getsize(CSV) / 1024**3

    results = {}
    for label, script, _ in TASKS:
        print(f"\n=== {label} ({script}) ===")
        r = run_and_monitor(script)
        if r["returncode"] != 0 and not r["oom"]:
            sys.exit(f"{script} failed (return code {r['returncode']}); fix it before benchmarking.")
        print(r["stdout"])
        print(f"-> {r['wall_s']:.1f}s, peak {r['peak_mb']:.0f} MB")
        results[label] = r

    ginfo = dask_graph_info()

    # ---------------- save raw results
    with open("results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, f)

    labels = [t[0] for t in TASKS]
    colors = ["#c0392b", "#2980b9", "#27ae60"]

    # ---------------- Fig 1: peak RAM
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [results[l]["peak_mb"] for l in labels]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} MB", ha="center", va="bottom")
    ax.set_ylabel("Peak RAM (MB)")
    ax.set_title("Peak memory usage per method")
    fig.tight_layout(); fig.savefig("fig_peak_ram.png", dpi=200); plt.close(fig)

    # ---------------- Fig 2: time
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [results[l]["wall_s"] for l in labels]
    bars = ax.bar(labels, vals, color=colors)
    for b, l, v in zip(bars, labels, vals):
        txt = f"{v:.1f} s" + (" (crashed)" if results[l]["oom"] else "")
        ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center", va="bottom")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Execution time per method")
    fig.tight_layout(); fig.savefig("fig_time.png", dpi=200); plt.close(fig)

    # ---------------- Fig 3: RAM timeline
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for l, c in zip(labels, colors):
        ax.plot(results[l]["t"], results[l]["mem"], label=l, color=c, lw=1.6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("RAM (MB)")
    ax.set_title("Memory footprint over time")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig("fig_ram_timeline.png", dpi=200); plt.close(fig)

    # ---------------- LaTeX table
    rows = []
    for l in labels:
        r = results[l]
        status = "OOM / crash" if r["oom"] else "Success"
        rows.append(f"{tex_escape(l)} & {r['peak_mb']:.0f} & {r['avg_mb']:.0f} & "
                    f"{r['wall_s']:.1f} & {status} \\\\")
    with open("results_table.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrl}\n\\toprule\n"
                "Method & Peak RAM (MB) & Avg RAM (MB) & Time (s) & Outcome \\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- LaTeX variables
    vm = psutil.virtual_memory()
    try:
        import pandas, dask
        pv, dv = pandas.__version__, dask.__version__
    except Exception:
        pv = dv = "n/a"
    macros = {
        "DatasetSize": f"{size_gb:.2f}~GB",
        "TotalRAM": f"{vm.total / 1024**3:.1f}~GB",
        "CPUCount": psutil.cpu_count(logical=True),
        "OSName": tex_escape(platform.platform()),
        "PyVer": platform.python_version(),
        "PandasVer": pv, "DaskVer": dv,
        "NPart": ginfo["npart"], "RawTasks": ginfo["raw_tasks"], "OptTasks": ginfo["opt_tasks"],
    }
    for l, (_, _, k) in zip(labels, TASKS):
        macros[f"Time{k}"] = f"{results[l]['wall_s']:.1f}"
        macros[f"Peak{k}"] = f"{results[l]['peak_mb']:.0f}"
        macros[f"Avg{k}"] = f"{results[l]['avg_mb']:.0f}"
    with open("generated_vars.tex", "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    print("\nDone. Generated figures, results_table.tex and generated_vars.tex")


if __name__ == "__main__":
    main()