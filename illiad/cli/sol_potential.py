"""Reserved command for the future SOL potential analysis class."""

import argparse


PLACEHOLDER_MESSAGE = (
    "illiad-sol-potential is reserved for a future SOL potential analysis "
    "class; it is not implemented yet."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reserved command for the future ILLIAD SOL potential analysis."
    )
    return parser.parse_args()


def main():
    """Report that the reserved command is not implemented."""
    parse_args()
    raise SystemExit(PLACEHOLDER_MESSAGE)


if __name__ == "__main__":
    main()
