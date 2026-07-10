"""
BRMSD command-line interface.

Usage:
    BRMSD align    [options]   — average structure + per-atom weights
    BRMSD smooth   [options]   — trajectory smoothing
    BRMSD cluster  [options]   — clustering / cluster-center finding
    BRMSD domain   [options]   — rigid domain identification
    BRMSD focus    [options]   — focused alignment biased toward a domain
    BRMSD sweep    [options]   — parameter sweep for any module
"""

import argparse
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("MDAnalysis").setLevel(logging.WARNING)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='BRMSD',
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(
        title='subcommands',
        dest='subcommand',
        metavar='<subcommand>')
    subparsers.required = True

    from . import align, smooth, cluster, domain, focus, sweep
    align.add_subparser(subparsers)
    smooth.add_subparser(subparsers)
    cluster.add_subparser(subparsers)
    domain.add_subparser(subparsers)
    focus.add_subparser(subparsers)
    sweep.add_subparser(subparsers)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
