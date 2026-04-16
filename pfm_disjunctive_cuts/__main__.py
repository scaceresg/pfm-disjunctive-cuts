#!/usr/bin/env python3
"""
Entry point for pfm_disjunctive_cuts package.
"""

import argparse
import sys

from . import PFMDisjunctiveCuts

ALGORITHMS = {
    "2n_naive",
    "2n_cumulative",
    "further_improve",
    "mip_disjunctive",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PFM Disjunctive Cuts Algorithms")
    parser.add_argument(
        "--instance", required=True, help="Instance file name (e.g., tai20_5_1.txt)"
    )
    parser.add_argument(
        "--inst-type",
        choices=["taillard", "vallada"],
        required=True,
        help="Instance type: taillard or vallada",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        required=True,
        help="Disjunctive cuts algorithm to execute",
    )
    parser.add_argument(
        "--sorted-jobs",
        action="store_true",
        help="Sort jobs by total workload before applying cuts",
    )
    parser.add_argument(
        "--decreasing-order",
        action="store_true",
        help="Sort jobs in decreasing order when --sorted-jobs is enabled",
    )
    parser.add_argument(
        "--position-cuts-first",
        action="store_true",
        help="Generate position-based cuts before job-based cuts",
    )
    parser.add_argument(
        "--sparser-cuts",
        action="store_true",
        help="Reduce cut coefficients by subtracting the minimum bound",
    )
    parser.add_argument(
        "--priority-order",
        choices=["alpha", "fractional"],
        help="Priority order for mip_disjunctive only",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Show solver output during optimization",
    )
    parser.add_argument(
        "--obj-diff-rounding",
        type=float,
        default=0.002,
        help="Objective rounding tolerance",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of CPLEX threads to use",
    )
    parser.add_argument(
        "--mip-emphasis",
        type=int,
        default=0,
        help="CPLEX MIP emphasis parameter",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=10800,
        help="Time limit in seconds",
    )
    parser.add_argument(
        "--no-cuts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable CPLEX cuts during MIP solving",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.decreasing_order and not args.sorted_jobs:
        raise ValueError("--decreasing-order requires --sorted-jobs")

    if args.priority_order and args.algorithm != "mip_disjunctive":
        raise ValueError("--priority-order is only valid with --algorithm mip_disjunctive")


def format_result_value(value) -> str:
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def print_summary(pfm: PFMDisjunctiveCuts, algorithm: str, results: dict) -> None:
    print(f"Algorithm: {algorithm}")
    print(f"Instance: {pfm.data_file}")
    print(f"Type: {pfm.inst_name}")
    print(f"Jobs: {pfm.n}")
    print(f"Machines: {pfm.m}")
    print(f"Best Known UB: {pfm.best}")

    for key, value in results.items():
        print(f"{key}: {format_result_value(value)}")


def run_algorithm(pfm: PFMDisjunctiveCuts, args: argparse.Namespace) -> dict:
    if args.algorithm == "2n_naive":
        return pfm.run_2n_naive_disjunctive_cuts(
            sparser_cuts=args.sparser_cuts,
        )

    if args.algorithm == "2n_cumulative":
        return pfm.run_2n_cumulative_disjunctive_cuts(
            sorted_jobs=args.sorted_jobs,
            decreasing_order=args.decreasing_order,
            position_cuts_first=args.position_cuts_first,
            sparser_cuts=args.sparser_cuts,
        )

    if args.algorithm == "further_improve":
        return pfm.run_further_improve_disj_cuts(
            disj_cuts_approach="2n_cumulative",
            sorted_jobs=args.sorted_jobs,
            decreasing_order=args.decreasing_order,
            position_cuts_first=args.position_cuts_first,
            sparser_cuts=args.sparser_cuts,
        )

    return pfm.run_mip_disjunctive_cuts(
        mip_priority_order=args.priority_order,
        disj_cuts_approach="2n_naive",
        sorted_jobs=args.sorted_jobs,
        decreasing_order=args.decreasing_order,
        position_cuts_first=args.position_cuts_first,
        sparser_cuts=args.sparser_cuts,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)

        pfm = PFMDisjunctiveCuts(data_file=args.instance, inst_name=args.inst_type)

        pfm.set_cplex_params_user(
            show_output=args.show_output,
            obj_diff_rounding=args.obj_diff_rounding,
            n_threads=args.threads,
            mip_emphasis=args.mip_emphasis,
            time_limit=args.time_limit,
            no_cuts=args.no_cuts,
        )

        print(
            "===========================================\n"
            f"Executing {args.algorithm} for instance={args.instance}, inst_type={args.inst_type}\n"
            f"sorted_jobs={args.sorted_jobs}, decreasing_order={args.decreasing_order}\n"
            f"position_cuts_first={args.position_cuts_first}, sparser_cuts={args.sparser_cuts}\n"
            f"priority_order={args.priority_order}\n"
            "===========================================\n",
        )
        results = run_algorithm(pfm, args)
        print_summary(pfm, args.algorithm, results)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
