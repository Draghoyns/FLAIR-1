import argparse

from src.zone_detect.optimization.quantization.run import dry_run, run_pipeline
from src.zone_detect.utils import initial_log


def parse_args():
    parser = argparse.ArgumentParser(description="Quantization pipeline entrypoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the pipeline steps without executing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    initial_log()

    if args.dry_run:
        print("=== DRY RUN MODE ===")
        dry_run()
    else:
        print("=== EXECUTING PIPELINE ===")
        run_pipeline()


if __name__ == "__main__":
    main()


# command
# python src/zone_detect/quantization/main.py --dry-run
# python src/zone_detect/quantization/main.py
