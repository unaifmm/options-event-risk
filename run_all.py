"""
Re-run the whole project with one command, in the right order.

    python run_all.py              # tests + every analysis script, on the data you have (about 3 minutes)
    python run_all.py --download   # first update the data (01_download_data.py; the option candles
                                   # take several days on the vendor's free plan - re-run until done)
    python run_all.py --from 05    # start at script 05 (e.g. after changing it)

It stops at the first error, so a broken step is never hidden by later output,
and prints how long each step took. Charts go to figures/, tables to results/.
Scripts whose data is not downloaded yet are skipped with a note.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

# (script, what it does, data it needs)
STEPS = [
    ("02_data_checks.py", "data checks: coverage, put-call parity, expiry prices, VIX1D vs straddles, clocks", config.CANDLES_DIR),
    ("03_long_sample.py", "VIX1D vs the S&P 500's actual moves, May 2022 - Nov 2026", config.CBOE_DIR),
    ("04_robustness.py", "is step 3 a handful of days? trims, medians, periods", config.CBOE_DIR),
    ("05_straddles.py", "the real trade: SPY straddles with costs and tail risk", config.CANDLES_DIR),
    ("06_black_scholes.py", "my pricer vs the vendor, early exercise, the smile, a rebuilt VIX", config.GREEKS_DIR),
    ("07_intraday.py", "implied variance minute by minute: overnight releases, the FOMC afternoon", config.CANDLES_DIR),
]


def has_data(path: Path) -> bool:
    return path.exists() and (path.is_file() or any(path.iterdir()))


def run(cmd: list[str], label: str, resume: str | None = None) -> float:
    print(f"\n{'=' * 78}\n>>> {label}\n{'=' * 78}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        hint = f" Fix it, then continue with:  python run_all.py --from {resume}" if resume else ""
        print(f"\n!! '{label}' failed (exit code {result.returncode}).{hint}")
        sys.exit(result.returncode)
    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="update the data first (01_download_data.py)")
    parser.add_argument("--from", dest="start", default="02", help="first script number to run, e.g. 05")
    parser.add_argument("--no-tests", action="store_true", help="skip the unit tests")
    args = parser.parse_args()
    args.start = args.start.zfill(2)

    timings = []
    if args.download:
        timings.append(("01_download_data.py", run([sys.executable, "01_download_data.py"], "01 download data")))
    if not args.no_tests:
        timings.append(("unit tests", run([sys.executable, "-m", "pytest", "-q"], "unit tests (pytest)")))

    for script, what, needs in STEPS:
        if script[:2] < args.start or not (ROOT / script).exists():
            continue
        if not has_data(needs):
            print(f"\n-- skipping {script}: its data is not downloaded yet (run with --download)")
            continue
        timings.append((script, run([sys.executable, script], f"{script[:2]} {what}", resume=script[:2])))

    print(f"\n{'=' * 78}\nAll done.")
    for name, secs in timings:
        print(f"  {name:<28} {secs:6.0f} s")
    print(f"  {'total':<28} {sum(s for _, s in timings):6.0f} s")
    print("Charts: figures/    Tables: results/")


if __name__ == "__main__":
    main()
