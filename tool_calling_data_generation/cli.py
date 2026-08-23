"""Command-line interface for validation, generation, and private upload."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

from tool_calling_data_generation.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tool-data-gen")
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a YAML config without loading ML models")
    validate.add_argument("--config", "-c", required=True)

    generate = commands.add_parser("generate", help="Generate and save a local dataset run")
    generate.add_argument("--config", "-c", required=True)

    upload = commands.add_parser("upload", help="Upload a generated run to a private Hub dataset")
    upload.add_argument("--config", "-c", required=True)
    upload.add_argument("--manifest", required=True)
    upload.add_argument(
        "--confirm-private-upload",
        action="store_true",
        help="Required acknowledgement that this mutates the configured private Hub repo",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)

    if args.command == "validate":
        print(json.dumps(config, indent=2))
        return 0

    if args.command == "generate":
        from tool_calling_data_generation.pipeline import run_generation

        artifacts = run_generation(config)
        print(json.dumps(artifacts.__dict__, indent=2))
        return 0

    if args.command == "upload":
        if not args.confirm_private_upload:
            raise SystemExit("Upload requires --confirm-private-upload.")
        from tool_calling_data_generation.hub import upload_private_run

        result = upload_private_run(config, args.manifest)
        print(json.dumps(result, indent=2))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


