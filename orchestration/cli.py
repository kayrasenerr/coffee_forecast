"""
orchestration/cli.py
====================
Command-line interface for Coffee Quant.

Usage examples:
  python -m orchestration.cli run --variety arabica --backtest
  python -m orchestration.cli run --variety robusta --start 2018-01-01 --end 2024-01-01
  python -m orchestration.cli ingest --variety arabica
  python -m orchestration.cli features --variety arabica
  python -m orchestration.cli experiments list
  python -m orchestration.cli experiments compare --metric directional_accuracy
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> None:
    from contracts.schemas import CoffeeVariety, DataFrequency
    from orchestration.pipeline import CoffeeQuantPipeline

    pipeline = CoffeeQuantPipeline(
        start_date=args.start,
        end_date=args.end,
    )
    variety = CoffeeVariety(args.variety)
    result = pipeline.run(
        variety=variety,
        backtest=args.backtest,
        plots=args.plots,
    )
    print(f"\n✅  Run complete. ID: {result['run_id']}")
    print(f"    Models fitted: {list(result['models'].keys())}")
    if result["forecasts"]:
        for name, fc in result["forecasts"].items():
            print(f"    [{name}] prob_up={fc.prob_up:.2%}  mean_return={fc.mean_return:.4f}")


def cmd_features(args: argparse.Namespace) -> None:
    from contracts.schemas import CoffeeVariety, DataFrequency
    from orchestration.pipeline import CoffeeQuantPipeline

    pipeline = CoffeeQuantPipeline(start_date=args.start, end_date=args.end)
    variety = CoffeeVariety(args.variety)
    raw = pipeline.ingest(variety)
    frame = pipeline.build_features(raw, variety)
    print(f"\n✅  Features built: {frame.df.shape[0]} rows × {len(frame.feature_names)} features")
    print(f"    Saved to store as: {variety.value}_features_D")


def cmd_experiments(args: argparse.Namespace) -> None:
    from experiment.tracker import FileExperimentTracker

    tracker = FileExperimentTracker(settings.experiments_dir)

    if args.subcommand == "list":
        runs = tracker.list_runs()
        if not runs:
            print("No runs found.")
            return
        for r in runs[-20:]:
            print(f"  {r['run_id']}  [{r['status']}]  {r.get('start_time', '')[:19]}")

    elif args.subcommand == "compare":
        runs = tracker.list_runs()
        if not runs:
            print("No runs found.")
            return
        run_ids = [r["run_id"] for r in runs[-10:]]
        metrics = args.metric.split(",") if args.metric else ["directional_accuracy", "signal_sharpe"]
        df = tracker.compare_runs(run_ids, metrics)
        print("\n" + df.to_string())

    elif args.subcommand == "best":
        best = tracker.best_run(args.metric or "directional_accuracy")
        print(f"Best run: {best}")


def cmd_store(args: argparse.Namespace) -> None:
    from features.store import ParquetFeatureStore

    store = ParquetFeatureStore(settings.features_dir)
    if args.subcommand == "list":
        df = store.summary()
        if df.empty:
            print("Feature store is empty.")
        else:
            print(df.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="coffee_quant",
        description="Coffee Quant — Modular Coffee Market Forecasting System",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run full pipeline")
    run_p.add_argument("--variety", default="arabica", choices=["arabica", "robusta"])
    run_p.add_argument("--start", default=settings.default_start_date)
    run_p.add_argument("--end",   default=settings.default_end_date)
    run_p.add_argument("--backtest", action="store_true")
    run_p.add_argument("--plots",    action="store_true")

    # features
    feat_p = sub.add_parser("features", help="Build and cache feature matrix")
    feat_p.add_argument("--variety", default="arabica", choices=["arabica", "robusta"])
    feat_p.add_argument("--start", default=settings.default_start_date)
    feat_p.add_argument("--end",   default=settings.default_end_date)

    # experiments
    exp_p = sub.add_parser("experiments", help="Manage experiment runs")
    exp_sub = exp_p.add_subparsers(dest="subcommand")
    exp_sub.add_parser("list")
    cmp_p = exp_sub.add_parser("compare")
    cmp_p.add_argument("--metric", default="directional_accuracy")
    best_p = exp_sub.add_parser("best")
    best_p.add_argument("--metric", default="directional_accuracy")

    # store
    st_p = sub.add_parser("store", help="Inspect feature store")
    st_sub = st_p.add_subparsers(dest="subcommand")
    st_sub.add_parser("list")

    args = parser.parse_args()
    _setup_logging(args.log_level)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "features":
        cmd_features(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "store":
        cmd_store(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
