import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace


logger = logging.getLogger("gitcast.cli")


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Create a podcast episode from a github repo",
        formatter_class=ArgumentDefaultsHelpFormatter(),
    )
    parser.add_argument(
        "repo",
        type=str,
        help="The GitHub repository to use for generating the podcast",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path name (without suffix .mp3)",
        default=None,
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="The approach duration of the podcast episode in minutes",
        default=10,
    )
    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == "__main__":
    main()
