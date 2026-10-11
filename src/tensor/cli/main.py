"""Lazy command dispatch."""


def main(argv=None):
    from importlib import import_module
    from tensor.cli.parser import make_parser

    parser = make_parser()
    args = parser.parse_args(argv)
    commands = {
        "add": "modules",
        "install": "modules",
        "pack": "modules",
        "publish": "modules",
        "resolve": "modules",
        "doctor": "doctor",
        "build": "build",
        "cache": "cache",
        "inspect": "inspect",
        "run": "run",
        "bench": "run",
    }
    handler = import_module("tensor.cli." + commands[args.command])
    return handler.execute_command(args, parser)
