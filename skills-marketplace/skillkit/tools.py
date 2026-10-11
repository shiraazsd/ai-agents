"""Turn a skill's scripts/tools.py into a CLI (for Claude Code / Copilot).

The same plain functions become ADK FunctionTools via skillkit.adk.
A tool = a public, top-level function with type hints and a docstring.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from types import ModuleType
from typing import Callable


def public_functions(module: ModuleType) -> list[Callable]:
    return [
        fn
        for name, fn in vars(module).items()
        if inspect.isfunction(fn) and not name.startswith("_") and fn.__module__ == module.__name__
    ]


def _arg_type(annotation):
    if annotation is bool:
        return lambda v: v.lower() in ("1", "true", "yes", "y")
    return annotation if annotation in (int, float, str) else json.loads


def run(argv: list[str] | None = None) -> None:
    """Call from tools.py:  if __name__ == "__main__": run()"""
    module = sys.modules["__main__"]
    parser = argparse.ArgumentParser(description=(module.__doc__ or "").strip())
    sub = parser.add_subparsers(dest="tool", required=True)
    fns = {fn.__name__: fn for fn in public_functions(module)}
    for name, fn in fns.items():
        p = sub.add_parser(name, help=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else None)
        for param in inspect.signature(fn, eval_str=True).parameters.values():
            required = param.default is inspect.Parameter.empty
            p.add_argument(
                f"--{param.name.replace('_', '-')}",
                dest=param.name,
                type=_arg_type(param.annotation),
                required=required,
                default=None if required else param.default,
            )
    args = vars(parser.parse_args(argv))
    result = fns[args.pop("tool")](**args)
    print(json.dumps(result, indent=2, default=str))
