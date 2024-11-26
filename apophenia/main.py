import argparse
import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logging(verbosity_level):
    """
    Configure the logging level based on the verbosity.
    verbosity_level: 0 (warnings only), 1 (-v, info level), 2 (-vv, debug level)
    """
    if verbosity_level == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity_level == 1:
        logging.basicConfig(level=logging.INFO)
        logger.info("Logging verbosity set INFO")
    elif verbosity_level >= 2:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Logging verbosity set DEBUG")


def discover_commands():
    commands = {}
    cmds_path = Path(__file__).parent / "cmds"
    for loader, module_name, is_pkg in pkgutil.iter_modules([str(cmds_path)]):
        module = importlib.import_module(f"apophenia.cmds.{module_name}")
        if hasattr(module, "add_arguments") and hasattr(module, "main"):
            commands[module_name] = module
    return commands


def main():
    parser = argparse.ArgumentParser(
        prog="apophenia", description="Apophenia command-line tool"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (-v for INFO, -vv for DEBUG)",
    )

    commands = discover_commands()

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for command_name, module in commands.items():
        command_help = getattr(module, "command_description", f"{command_name} command")
        command_parser = subparsers.add_parser(command_name, help=command_help)
        module.add_arguments(command_parser)

    args = parser.parse_args()

    configure_logging(args.verbose)

    if args.command in commands:
        commands[args.command].main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
