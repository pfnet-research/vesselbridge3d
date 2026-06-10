"""
Shared YAML config support for train.py / inference.py.

Usage:
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument(...)  # define all other args as usual
    args = parse_args_with_config(ap)

Precedence: explicit CLI args > config file > argparse defaults.
"""

import argparse
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict (keys must match argparse dests)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Register the --config option on the parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config with preset values. Explicit CLI args override config values.",
    )


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args, applying a YAML config (if --config is given) as defaults.

    Config values override the parser's hardcoded defaults; any explicitly
    provided CLI argument still wins over the config.
    """
    # Pre-parse only --config so required args don't cause the pre-parse to fail.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    if pre_args.config:
        cfg = load_yaml_config(pre_args.config)
        valid = {a.dest for a in parser._actions}
        unknown = set(cfg) - valid
        if unknown:
            raise ValueError(
                f"Unknown config keys in {pre_args.config}: {sorted(unknown)}"
            )
        # config fills in defaults; CLI still overrides.
        parser.set_defaults(**cfg)
        # A required arg satisfied by the config no longer needs to be on the CLI.
        for action in parser._actions:
            if getattr(action, "required", False) and action.dest in cfg:
                action.required = False

    return parser.parse_args()
