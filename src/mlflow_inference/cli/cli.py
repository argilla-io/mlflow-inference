import click
from click import Group

from .build_docker import build_docker
from .serve import serve

SUPPORTED_COMMANDS = [serve, build_docker]


def main():
    commands = Group(no_args_is_help=True)
    for command in SUPPORTED_COMMANDS:
        commands.add_command(command, command.name)
    click.CommandCollection(sources=[commands])()
