"""Reserved command for the future SOL density analysis class."""

import argparse


PLACEHOLDER_MESSAGE = (
    "illiad-sol-density is reserved for a future SOL density analysis class; "
    "it is not implemented yet."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reserved command for the future ILLIAD SOL density analysis."
    )
    return parser.parse_args()


def main():
    """Report that the reserved command is not implemented."""
    parse_args()
    raise SystemExit(PLACEHOLDER_MESSAGE)


if __name__ == "__main__":
    main()
