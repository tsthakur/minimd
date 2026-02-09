"""Entry point: python -m minimd config.yaml"""

from __future__ import annotations

import sys

from minimd.config import Config
from minimd.simulation import run


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m minimd <config.yaml>", file=sys.stderr)
        sys.exit(1)
    config = Config.from_yaml(sys.argv[1])
    run(config)


if __name__ == "__main__":
    main()
