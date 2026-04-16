from copy import deepcopy
from time import time

import numpy as np

from .pfm_mip_model import PFMmip


# Class to generate disjunctive cuts for the PFM
class PFMDisjunctiveCuts(PFMmip):
    # Constructor
    def __init__(self, data_file=None, inst_name=None, n=None, m=None, proc_times=None, best=None):
        super().__init__(data_file, inst_name, n, m, proc_times, best)

    ############### MAIN ALGORITHMS TO GENERATE CUTS ###############
    # Run 2n cumulative disjunctive cuts algorithm (algorithms 1 and 2)
    def run_2n_cumulative_disjunctive_cuts(
        self,
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        position_cuts_first: bool = False,
        sparser_cuts: bool = False,
    ) -> dict[str, tuple[int | None, float | None]]:
        """
        Runs algorithm to generate and apply 2n cumulative disjunctive cuts
        for the PFM: job-based (1st kind) and position-based (2nd kind).

        Parameters:
        ---------
        sorted_jobs (bool, optional):
            Sorts jobs by their total workload. Defaults to False.
        decreasing_order (bool, optional):
            Sorts jobs in decreasing (non-increasing) order. It needs
            'sorted_jobs' to be True to work. Defaults to False.
        position_cuts_first (bool, optional):
            If True, generates and adds position-based cuts first before
            job-based cuts. Defaults to False.
        sparser_cuts (bool, optional):
            Makes the cuts sparser by subtracting the minimum bound value
            for a given job (j) or position (k) in the alpha matrix from each
            alpha_j_k in the cut. Defaults to False.

        Returns:
        ---------
        dict[str, tuple[int | None, float | None]]:
            A dictionary with tuples containing the lower bounds and runtimes
            of the LP relaxation and the two kinds of cuts:

                Example: {
                    "lb_lp": (lb_relaxation, runtime_relaxation),
                    "lb_n": (lb_after_first_n_cuts, runtime_after_first_n_cuts),
                    "lb_2n": (lb_after_2n_cuts, runtime_after_2n_cuts),
                }
        """
        self.start = time()

        self.jobs_array = None

        # Sort jobs in non-decreasing order by workload
        if sorted_jobs:
            self.jobs_array = self.sort_jobs(decreasing=decreasing_order)
        else:
            self.jobs_array = self.jobs

        # Initialise lbs matrix
        self.alpha = np.empty((self.n, self.n))

        # Build LP relaxation and get lower bound
        lb_lp = self.get_lp_relaxation_lb()
        runtime_lb_lp = round(time() - self.start, 3)

        # Generate first n cuts (cumulative)
        if position_cuts_first:
            # Generate position-based cuts first
            lb_n = self.generate_disj_cuts_pos_first(reduced_coeffs=sparser_cuts)
        else:
            # Generate job-based cuts first
            lb_n = self.generate_disj_cuts_jobs(reduced_coeffs=sparser_cuts)

        runtime_lb_n = round(time() - self.start, 3)

        if lb_n is None:
            return {
                "lb_lp": (lb_lp, runtime_lb_lp),
                "lb_n": (None, runtime_lb_n),
                "lb_2n": (None, None),
            }

        # Generate cuts the other n cuts (cumulative) + strengthen first n cuts
        if position_cuts_first:
            lb_2n = self.generate_disj_cuts_jobs_pos_first(reduced_coeffs=sparser_cuts)
        else:
            lb_2n = self.generate_disj_cuts_pos(reduced_coeffs=sparser_cuts)

        runtime_lb_2n = round(time() - self.start, 3)

        return {
            "lb_lp": (lb_lp, runtime_lb_lp),
            "lb_n": (lb_n, runtime_lb_n),
            "lb_2n": (lb_2n, runtime_lb_2n),
        }

    # Run naive 2n disjunctive cuts (non-cumulative)
    def run_2n_naive_disjunctive_cuts(
        self, sparser_cuts: bool = False
    ) -> dict[str, tuple[int | None, float | None]]:
        """
        Runs algorithm to generate and apply 2n non-cumulative disjunctive cuts
        in a naive way: job-based (1st kind) and position-based (2nd kind) for the PFM.

        Parameters:
        ---------
        sparser_cuts (bool, optional):
            Makes the cuts sparser by subtracting the minimum bound value for a given
            job (j) or position (k) in the alpha matrix from each alpha_j_k in the cut.
            Defaults to False.

        Returns:
        ---------
        dict[str, tuple[int | None, float | None]]:
            A dictionary with tuples containing the lower bounds and runtimes of the LP
            relaxation and the two kinds of cuts.

            Example {
                "lb_lp": (lb_relaxation, runtime_relaxation),
                "lb_2n": (lb_after_2n_cuts, runtime_after_2n_cuts),
            }
        """
        self.start = time()

        # Initialise lbs matrix
        self.alpha = np.empty((self.n, self.n))

        # Build LP relaxation and get lower bound
        lb_lp = self.get_lp_relaxation_lb()
        runtime_lb_lp = round(time() - self.start, 3)

        for j in self.jobs:
            for k in self.seq:
                disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                if disjunction_lb is None:
                    return {"lb_lp": (lb_lp, runtime_lb_lp), "lb_2n": (None, None)}

                self.alpha[j - 1, k - 1] = disjunction_lb

        lb_2n = self.add_naive_disjunctive_cuts(reduce_coeff=sparser_cuts)
        runtime_lb_2n = round(time() - self.start, 3)

        if lb_2n is None:
            return {
                "lb_lp": (lb_lp, runtime_lb_lp),
                "lb_2n": (None, runtime_lb_2n),
            }

        return {
            "lb_lp": (lb_lp, runtime_lb_lp),
            "lb_2n": (lb_2n, runtime_lb_2n),
        }

    # Run disjunctive cuts using fractional x_j_k variables only
    def run_fractional_var_disj_cuts(
        self,
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        reduced_alpha: bool = False,
        simultaneous_fract_cuts: bool = False,
        max_iters: int = 90,
        fractional_tol: float = 0.10,
    ) -> dict[str, tuple[int | None, float | None]]:
        """
        Run algorithm to generate and apply disjunctive cuts only to fractional x_j_k variables.

        Parameters:
        ---------
        sorted_jobs (bool, optional):
            Sorts jobs by their total workload. Defaults to False.
        decreasing_order (bool, optional):
            Sorts jobs in decreasing (non-increasing) order. It needs
            'sorted_jobs' to be True to work. Defaults to False.
        reduced_alpha (bool, optional):
            Makes the cuts sparser by subtracting the minimum bound value
            for a given job (j) or position (k) in the alpha matrix from each
            alpha_j_k in the cut. Defaults to False.
        simultaneous_fract_cuts (bool, optional):
            If True, generates simultaneous fractional cuts for jobs and positions.
            Defaults to False.
        max_iters (int, optional):
            Maximum number of iterations to perform when using fractional disjunctive cuts.
            Defaults to 90.
        fractional_tol (float, optional):
            Tolerance for considering a variable fractional. Defaults to 0.10.

        Returns:
        ---------
        dict[str, tuple[int | None, float | None]]:
            A dictionary with tuples containing the lower bounds and runtimes of the LP
            relaxation and the two sets of cuts (from fractional vars).

            Example {
                "lb_lp": (lb_relaxation, runtime_relaxation),
                "lb_1": (lb_after_first_set_of_cuts, runtime_after_first_set_of_cuts),
                "lb_2": (lb_after_second_set_of_cuts, runtime_after_second_set_of_cuts),
            }
        """
        self.start = time()

        self.jobs_array = None

        # Sort jobs in non-decreasing order by workload
        if sorted_jobs:
            self.jobs_array = self.sort_jobs(decreasing=decreasing_order)
        else:
            self.jobs_array = self.jobs

        # Build LP relaxation and get lower bound
        lb_lp = self.get_lp_relaxation_lb()
        runtime_lb_lp = round(time() - self.start, 3)

        # Initialise lbs matrix
        self.alpha = np.full((self.n, self.n), lb_lp)

        lb_2 = lb_lp
        lb_2_iter = 0
        num_iter = 0

        while lb_2_iter < lb_2 and num_iter < max_iters:
            lb_2_iter = lb_2
            num_iter += 1

            _, xs_fracs = self.get_fract_int_xs_vars(fract_tol=fractional_tol)

            if simultaneous_fract_cuts:
                # v1: Generate simultaneous job-based and pos-based disj. cuts for all fractional x vars
                lb_1, lb_2 = self.generate_simultaneous_fract_var_disj_cuts(
                    xs_fracts=xs_fracs, reduced_coeffs=reduced_alpha
                )
            else:
                # v2: Generate job-based first and then pos-based disj. cuts for all fractional x vars
                j_k_dict = self.get_jk_pairs_from_xs_fract_vars(xs_fract_vars=xs_fracs)

                print(len(j_k_dict["j"]), len(j_k_dict["k"]))

                lb_1 = self.generate_sequential_fract_disj_cuts_jobs(
                    j_fracts=j_k_dict["j"], reduced_coeffs=reduced_alpha
                )

                lb_2 = self.generate_sequential_fract_disj_cuts_pos(
                    k_fracts=j_k_dict["k"], reduced_coeffs=reduced_alpha
                )

            if lb_2 is None:
                return {
                    "lb_lp": (lb_lp, runtime_lb_lp),
                    "lb_1": (lb_1, round(time() - self.start, 3)),
                    "lb_2": (None, round(time() - self.start, 3)),
                }

        runtime_lb_2 = round(time() - self.start, 3)

        return {
            "lb_lp": (lb_lp, runtime_lb_lp),
            "lb_1": (lb_1, runtime_lb_2),
            "lb_2": (lb_2, runtime_lb_2),
        }

    # Run weak disjunctive cuts using fractional x_j_k variables only
    def run_weak_fractional_var_disj_cuts(
        self,
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        position_first: bool = False,
        reduced_alpha: bool = False,
        fractional_tol: float = 0.10,
    ):
        """
        Run algorithm to generate and apply disjunctive cuts using fractional x_j_k variables.

        Parameters:
        ---------
        sorted_jobs (bool, optional):
            Sorts jobs by their total workload. Defaults to False.
        decreasing_order (bool, optional):
            Sorts jobs in decreasing (non-increasing) order. It needs
            'sorted_jobs' to be True to work. Defaults to False.
        position_first (bool, optional):
            If True, generates and adds position-based cuts first before
            job-based cuts. Defaults to False.
        reduced_alpha (bool, optional):
            Makes the cuts sparser by subtracting the minimum bound value
            for a given job (j) or position (k) in the alpha matrix from each
            alpha_j_k in the cut. Defaults to False.
        fractional_tol (float, optional):
            Tolerance for considering a variable fractional. Defaults to 0.10.

        Returns:
        ---------
        dict[str, tuple[int | None, float | None]]:
            A dictionary with tuples containing the lower bounds and runtimes of the LP
            relaxation and the two kinds of cuts.

            Example {
                "lb_lp": (lb_relaxation, runtime_relaxation),
                "lb_1": (lb_after_first_set_of_cuts, runtime_after_first_set_of_cuts),
                "lb_2": (lb_after_second_set_of_cuts, runtime_after_second_set_of_cuts),
            }
        """
        self.start = time()

        self.jobs_array = None

        # Sort jobs in non-decreasing order by workload
        if sorted_jobs:
            self.jobs_array = self.sort_jobs(decreasing=decreasing_order)
        else:
            self.jobs_array = self.jobs

        # Build LP relaxation and get lower bound
        lb_lp = self.get_lp_relaxation_lb()
        runtime_lb_lp = round(time() - self.start, 3)

        # Initialise lbs matrix
        self.alpha = np.full((self.n, self.n), lb_lp)

        if position_first:
            lb_1 = self.generate_fract_var_disj_cuts_pos_first(
                reduced_coeffs=reduced_alpha, fractional_tol=fractional_tol
            )
        else:
            lb_1 = self.generate_fract_var_disj_cuts_jobs(
                reduced_coeffs=reduced_alpha, fractional_tol=fractional_tol
            )

        runtime_lb_1 = round(time() - self.start, 3)

        if lb_1 is None:
            return {
                "lb_lp": (lb_lp, runtime_lb_lp),
                "lb_1": (None, runtime_lb_1),
                "lb_2": (None, None),
            }

        if position_first:
            lb_2 = self.generate_fract_var_disj_cuts_jobs_pos_first(
                reduced_coeffs=reduced_alpha, fractional_tol=fractional_tol
            )
        else:
            lb_2 = self.generate_fract_var_disj_cuts_pos(
                reduced_coeffs=reduced_alpha, fractional_tol=fractional_tol
            )

        runtime_lb_2 = round(time() - self.start, 3)

        return {
            "lb_lp": (lb_lp, runtime_lb_lp),
            "lb_1": (lb_1, runtime_lb_1),
            "lb_2": (lb_2, runtime_lb_2),
        }

    # Run further improve disjunctive cuts (algorithm 6)
    def run_further_improve_disj_cuts(
        self,
        disj_cuts_approach: str,
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        position_cuts_first: bool = False,
        sparser_cuts: bool = False,
    ):
        """
        Run algorithm to further improve initial disjunctive cuts by revising each (j, k) pair.

        Parameters:
        ---------
        disj_cuts_approach (str, optional):
            Initial disjunctive cuts approach used to generate and apply the first cuts.
            It must be one in {'2n_cumulative', '2n_naive'}.
        sorted_jobs (bool, optional):
            Sorts jobs by their total workload. Defaults to False.
        decreasing_order (bool, optional):
            Sorts jobs in decreasing (non-increasing) order. It needs
            'sorted_jobs' to be True to work. Defaults to False.
        position_cuts_first (bool, optional):
            If True, generates and adds position-based cuts first before
            job-based cuts. Defaults to False.
        sparser_cuts (bool, optional):
            Makes the cuts sparser by subtracting the minimum bound value
            for a given job (j) or position (k) in the alpha matrix from each
            alpha_j_k in the cut. Defaults to False.

        Returns:
        ---------
        dict[str, tuple[int | None, float | None]]:
            A dictionary with tuples containing the lower bounds and runtimes of the LP
            relaxation, the two kinds of cuts and the further improved lower bounds,
            its runtime, and the total runtime.

            Example {
                "lb_lp": (lb_relaxation, runtime_relaxation),
                "lb_2n": (lb_after_2n_cuts, runtime_after_2n_cuts),
                "lb_fi": (lb_after_further_impr_cuts, runtime_after_further_impr_cuts, total_runtime),
            }
        """
        # Validate that disj_cuts_approach is correct
        if disj_cuts_approach not in {"2n_cumulative", "2n_naive"}:
            raise ValueError(
                "Invalid disj_cuts_approach parameter value. Must be one in {'2n_cumulative', '2n_naive'}."
            )

        lb_2n = None
        if disj_cuts_approach == "2n_cumulative":
            # Run 2n Cumulative Cuts
            results_2n_cuts = self.run_2n_cumulative_disjunctive_cuts(
                sorted_jobs=sorted_jobs,
                decreasing_order=decreasing_order,
                position_cuts_first=position_cuts_first,
                sparser_cuts=sparser_cuts,
            )
            lb_2n, lb_2n_time = results_2n_cuts["lb_2n"]

        elif disj_cuts_approach == "2n_naive":
            # Run 2n Naive Cuts
            results_2n_cuts = self.run_2n_naive_disjunctive_cuts(
                sparser_cuts=sparser_cuts,
            )
            lb_2n, lb_2n_time = results_2n_cuts["lb_2n"]

        # Check if lb_2n is None
        if lb_2n is None:
            return {
                "lb_lp": results_2n_cuts["lb_lp"],
                "lb_2n": (None, lb_2n_time),
                "lb_fi": (None, None, None),
            }

        lb_fi_start = time()

        # Set initial values for lb_fi and iteration count
        lb_fi = lb_2n
        lb_fi_iter = 0

        # Improve lb_fi (lb_2n) for all pairs of j,k
        while lb_fi_iter < lb_fi:
            lb_fi_iter = lb_fi

            # Generate further improve cuts for j, k pairs
            lb_fi = self.further_improve_disj_cuts(reduced_coeffs=sparser_cuts)

            if lb_fi is None:
                return {
                    "lb_lp": results_2n_cuts["lb_lp"],
                    "lb_2n": (lb_2n, lb_2n_time),
                    "lb_fi": (None, None, None),
                }

        runtime_lb_fi = round(time() - lb_fi_start, 3)
        runtime_full = round(time() - self.start, 3)

        return {
            "lb_lp": results_2n_cuts["lb_lp"],
            "lb_2n": (lb_2n, lb_2n_time),
            "lb_fi": (lb_fi, runtime_lb_fi, runtime_full),
        }

    # Run further improve disjunctive cuts by using fractional xs first
    def run_further_impr_disj_cuts_fract_first(
        self,
        pos_first: bool = False,
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        reduced_coeff: bool = False,
        fractional_tolerance: float = 0.25,
    ):
        """
        Run algorithm to further improve disjunctive cuts by revising each (j, k) pair using fractional x_j_k
        variables first.

        Args:
            pos_first (bool, optional): If True, applies position-based cuts before job-based cuts.
                                        Defaults to False.
            sorted_jobs (bool, optional): Sorts jobs by workload. Defaults to False.
            decreasing_order (bool, optional): Sorts jobs in decreasing (non-increasing) order. It needs
                                            'sorted_jobs' to be True to work. Defaults to False.
            reduced_coeff (bool, optional): Subtracts the minimum lower bound value from the alpha matrix.
                                        Defaults to False.
            fractional_tolerance (float, optional): Tolerance for considering a variable as fractional.
                                                    Defaults to 0.25.

        Returns:
            dict: A dictionary with tuples containing the lower bounds and runtimes of the LP relaxation,
                the two kinds of cuts, the further improved lower bound, its runtime, the number of iterations,
                and the total runtime.
        """
        # Run 2n Cuts
        results_2n_cuts = self.run_2n_disjunctive_cuts(
            sorted_jobs=sorted_jobs,
            decreasing_order=decreasing_order,
            position_first=pos_first,
            reduced_alpha=reduced_coeff,
        )

        # Check if lb_3 is None
        lb_3 = results_2n_cuts["lb_3"][0]

        if lb_3 is None:
            return {
                "lb_1": results_2n_cuts["lb_1"],
                "lb_2": results_2n_cuts["lb_2"],
                "lb_3": (None, results_2n_cuts["lb_3"][1]),
                "lb_4": (None, None, None, None),
            }

        lb_4_start = time()

        lb_4 = lb_3
        lb_4_iter = 0
        num_iter = 0

        while lb_4_iter < lb_4:
            # Get fractional and integer x vars
            xs_ints, xs_fracs = self.get_fract_int_xs_vars(fract_tol=fractional_tolerance)

            lb_4_iter = lb_4
            num_iter += 1

            # Generate further improve cuts for fractional x vars
            lb_4 = self.further_improve_disj_cuts_xs_array(
                xs_array=xs_fracs, reduced_coeffs=reduced_coeff
            )

            if lb_4 is None:
                return {
                    "lb_1": results_2n_cuts["lb_1"],
                    "lb_2": results_2n_cuts["lb_2"],
                    "lb_3": results_2n_cuts["lb_3"],
                    "lb_4": (None, None, num_iter, None),
                }

            elif lb_4 > lb_4_iter:
                continue
            else:
                # Generate further improve cuts for integer x vars
                lb_4 = self.further_improve_disj_cuts_xs_array(
                    xs_array=xs_ints, reduced_coeffs=reduced_coeff
                )

        runtime_lb_4 = round(time() - lb_4_start, 3)
        runtime_full = round(time() - self.start, 3)

        return {
            "lb_1": results_2n_cuts["lb_1"],
            "lb_2": results_2n_cuts["lb_2"],
            "lb_3": results_2n_cuts["lb_3"],
            "lb_4": (lb_4, runtime_lb_4, num_iter, runtime_full),
        }

    # Run MIP model + further improved disjunctive cuts
    def run_mip_further_improve_disj_cuts(
        self,
        priority_order: str = None,
        fract_first: bool = False,
        pos_first: bool = False,
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        reduced_coeff: bool = False,
        fractional_tolerance: float = 0.25,
    ):
        """
        Run MIP model after further improving disjunctive cuts.

        Args:
            priority_order (str, optional): If 'alpha', uses priority order based on alpha values.
                                            If 'fractional', uses priority order based on fractional x_j_k values.
                                            If None, does not use priority order. Defaults to None.
            fract_first (bool, optional): If True, uses fractional x_j_k variables first to further improve
                                        disjunctive cuts first, else further improves each (j, k) pairs. Defaults to False.
            pos_first (bool, optional): If True, applies position-based cuts before job-based cuts.
                                        Defaults to False.
            sorted_jobs (bool, optional): Sorts jobs by workload. Defaults to False.
            decreasing_order (bool, optional): Sorts jobs in decreasing (non-increasing) order. It needs
                                            'sorted_jobs' to be True to work. Defaults to False.
            reduced_coeff (bool, optional): Subtracts the minimum lower bound value from the alpha matrix.
                                        Defaults to False.
            fractional_tolerance (float, optional): Tolerance for considering a variable as fractional.
                                                    Only used if 'fract_first' is True. Defaults to 0.25.

        Returns:
            dict: A dictionary with tuples containing the lower bounds and runtimes of the LP relaxation,
                the two kinds of cuts, the further improved lower bound, its runtime, the makespan found by the MIP,
                its runtime, and the total runtime.
        """

        if priority_order not in {"alpha", "fractional", None}:
            raise ValueError(
                "Invalid priority_order parameter value. Must be one in {'alpha', 'fractional', None}."
            )

        # Run further improve Cuts
        func = (
            self.run_further_impr_disj_cuts_fract_first
            if fract_first
            else self.run_further_improve_disj_cuts
        )

        kwargs = dict(
            pos_first=pos_first,
            sorted_jobs=sorted_jobs,
            decreasing_order=decreasing_order,
            reduced_coeff=reduced_coeff,
        )

        if fract_first:
            kwargs["fractional_tolerance"] = fractional_tolerance

        results_impr = func(**kwargs)

        lb_4 = results_impr["lb_4"][0]

        if lb_4 is None:
            return {
                "lb_1": results_impr["lb_1"],
                "lb_2": results_impr["lb_2"],
                "lb_3": results_impr["lb_3"],
                "lb_4": (None, None, None),
            }

        mip_start = time()

        if priority_order:
            # Solve MIP using priority order based on lower bounds
            makespan_mip = self.solve_mip_disj_cuts_priority_ord(priority_by=priority_order)
        else:
            # Solve MIP without priority order
            makespan_mip = self.solve_mip_disj_cuts()

        runtime_mip = round(time() - mip_start, 3)
        runtime_full = round(time() - self.start, 3)

        return {
            "lb_1": results_impr["lb_1"],
            "lb_2": results_impr["lb_2"],
            "lb_3": results_impr["lb_3"],
            "lb_4": results_impr["lb_4"],
            "mip": (makespan_mip, runtime_mip, runtime_full),
        }

    # Run disjunctive cuts (2n, naive or fractional) + MIP model (with or without priority order)
    def run_mip_disjunctive_cuts(
        self,
        mip_priority_order: str = None,
        disj_cuts_approach: str = "2n_cumulative",
        sorted_jobs: bool = False,
        decreasing_order: bool = False,
        position_cuts_first: bool = False,
        sparser_cuts: bool = False,
        simultaneous_fract_cuts: bool = False,
        max_iters: int = 90,
        fractional_tol: float = 0.10,
    ):
        """
        Run MIP model after applying 2n disjunctive cuts directly.

        Parameters:
        ---------
        mip_priority_order (str, optional):
            If 'alpha', uses priority order based on alpha values.
            If 'fractional', uses priority order based on fractional x_j_k values.
            If None, does not use priority order. Defaults to None.
        disj_cuts_approach (str, optional):
            Initial disjunctive cuts approach used to generate and apply the 2n cuts.
            It must be one in {'2n_cumulative', '2n_naive'}.
        sorted_jobs (bool, optional):
            Sorts jobs by their total workload. Defaults to False.
        decreasing_order (bool, optional):
            Sorts jobs in decreasing (non-increasing) order. It needs
            'sorted_jobs' to be True to work. Defaults to False.
        position_cuts_first (bool, optional):
            If True, generates and adds position-based cuts first before
            job-based cuts. Defaults to False.
        sparser_cuts (bool, optional):
            Makes the cuts sparser by subtracting the minimum bound value
            for a given job (j) or position (k) in the alpha matrix from each
            alpha_j_k in the cut. Defaults to False.
        simultaneous_fract_cuts (bool, optional):
            If True, generates simultaneous fractional cuts for jobs and positions.
            Defaults to False.
        max_iters (int, optional):
            Maximum number of iterations to perform when using fractional disjunctive cuts.
            Defaults to 90.
        fractional_tol (float, optional):
            Tolerance for considering a variable fractional. Defaults to 0.10.

        Returns:
        ---------
        dict[str, tuple[int | None, float | None]]:
            A dictionary with tuples containing the lower bounds and runtimes of the LP
            relaxation, the lower bounds and running times after adding cuts, and the
            MIP solution, total nodes processed, MIP running times, and the total runtime.

            Example {
                "lb_lp": (lb_relaxation, runtime_relaxation),
                "lb_2n": (lb_after_2n_cuts, runtime_after_2n_cuts),
                "mip": (makespan_mip, nodes_processed, runtime_mip, runtime_full),
            }
        """
        if disj_cuts_approach not in {"2n_cumulative", "2n_naive", "fractional", "fractional_weak"}:
            raise ValueError(
                "Invalid disj_cuts_approach parameter value. Must be one in {'2n_cumulative', '2n_naive', 'fractional', 'fractional_weak'}."
            )

        if mip_priority_order not in {"alpha", "fractional", None}:
            raise ValueError(
                "Invalid mip_priority_order parameter value. Must be one in {'alpha', 'fractional', None}."
            )

        if disj_cuts_approach == "2n_naive":
            func = self.run_2n_naive_disjunctive_cuts
        elif disj_cuts_approach == "2n_cumulative":
            func = self.run_2n_cumulative_disjunctive_cuts
        elif disj_cuts_approach == "fractional":
            func = self.run_fractional_var_disj_cuts
        elif disj_cuts_approach == "fractional_weak":
            func = self.run_weak_fractional_var_disj_cuts

        kwargs = {}

        if disj_cuts_approach == "2n_naive":
            kwargs["sparser_cuts"] = sparser_cuts
        elif disj_cuts_approach == "2n_cumulative":
            kwargs["sorted_jobs"] = sorted_jobs
            kwargs["decreasing_order"] = decreasing_order
            kwargs["position_cuts_first"] = position_cuts_first
            kwargs["sparser_cuts"] = sparser_cuts
        elif disj_cuts_approach == "fractional":
            kwargs["sorted_jobs"] = sorted_jobs
            kwargs["decreasing_order"] = decreasing_order
            kwargs["reduced_alpha"] = sparser_cuts
            kwargs["simultaneous_fract_cuts"] = simultaneous_fract_cuts
            kwargs["max_iters"] = max_iters
            kwargs["fractional_tol"] = fractional_tol
        elif disj_cuts_approach == "fractional_weak":
            kwargs["sorted_jobs"] = sorted_jobs
            kwargs["decreasing_order"] = decreasing_order
            kwargs["position_first"] = position_cuts_first
            kwargs["reduced_alpha"] = sparser_cuts
            kwargs["fractional_tol"] = fractional_tol

        # Run disjunctive cuts
        results_disj_cuts = func(**kwargs)

        # Get the last key from the results dictionary (lb_2n for 2n approaches, lb_2 for fractional)
        last_key = list(results_disj_cuts.keys())[-1]
        lb_final = results_disj_cuts[last_key][0]

        if lb_final is None:
            return {
                **results_disj_cuts,
                "mip": (None, None, None),
            }

        mip_start = time()

        if mip_priority_order:
            # Solve MIP using priority order based on lower bounds
            makespan_mip = self.solve_mip_disj_cuts_priority_ord(priority_by=mip_priority_order)
        else:
            # Solve MIP without priority order
            makespan_mip = self.solve_mip_disj_cuts()

        nodes_processed = self.get_num_nodes_processed()

        runtime_mip = round(time() - mip_start, 3)
        runtime_full = round(time() - self.start, 3)

        return {
            **results_disj_cuts,
            "mip": (makespan_mip, nodes_processed, runtime_mip, runtime_full),
        }

    # Run MIP model without disjunctive cuts
    def run_mip_model(self) -> dict:
        """
        Run the MIP model without any disjunctive cuts.

        Returns
        -------
        dict:
            A dictionary containing the makespan found by the MIP,
            total nodes processed, and the MIP running time.
        """
        mip_start = time()

        # Change xs variable types to binary
        self.change_xs_var_types(var_type="binary")

        makespan_mip = self.solve_model(show_log=True, rounding_thrsh=self.obj_diff_rounding)

        nodes_processed = self.get_num_nodes_processed()

        runtime_mip = round(time() - mip_start, 3)

        return {
            "makespan_mip": makespan_mip,
            "nodes_processed": nodes_processed,
            "runtime_mip": runtime_mip,
        }

    # Run further cumulative disjunctive cuts algorithm (algorithms 4 and 5)
    def run_further_disjunctive_cuts(
        self, sorted_jobs: bool = False, decreasing_order: bool = False
    ):
        # Stopped coding here as the further cuts did not get improved solutions
        pass

    ############### METHODS FOR 2n DISJ. CUTS APPROACHES ###############
    # Generate cumulative disj. cuts of the 1st kind (job-based)
    def generate_disj_cuts_jobs(self, reduced_coeffs: bool):
        for t in self.jobs_array:
            for k in self.seq:
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                if disjunction_lb is None:
                    return None

                # Save disjunction lb
                self.alpha[t - 1, k - 1] = disjunction_lb

            # Add disjunctive cut for job
            self.add_disj_cuts_jobs(j=t, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate cumulative disj. cuts of the 1st kind using vars reduced cost
    def generate_disj_cuts_jobs_quickrc(self, reduced_coeffs: bool, curr_bound: float):
        # Get xs var values and reduced costs
        xs_vars = self.get_x_var_vals()
        xs_reduced_costs = self.get_reduced_costs()

        for t in self.jobs_array:
            for k in self.seq:
                if xs_vars[f"x_{t}_{k}"] <= 0.0001:
                    # Get lb from disjunction
                    disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                    if disjunction_lb is None:
                        return None

                    # Save disjunction lb
                    self.alpha[t - 1, k - 1] = disjunction_lb
                else:
                    # Add var reduced costs to the current bound
                    self.alpha[t - 1, k - 1] = np.ceil(curr_bound + xs_reduced_costs[f"x_{t}_{k}"])

            # Add disjunctive cut for job
            self.add_disj_cuts_jobs(j=t, reduce_coeffs=reduced_coeffs)

            # Update current bound
            curr_bound = self.solve_model(
                show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding
            )

        return curr_bound

    # Generate cumulative disj. cuts of the 1st kind job-based with pos-based first
    def generate_disj_cuts_jobs_pos_first(self, reduced_coeffs: bool):
        for t in self.jobs_array:
            for k in self.seq:
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                if disjunction_lb is None:
                    return None

                # Strenghten current lb and update k-th cut
                if disjunction_lb > self.alpha[t - 1, k - 1]:
                    self.alpha[t - 1, k - 1] = disjunction_lb
                    self.update_disj_cut_pos(k=k, reduce_coeffs=reduced_coeffs)

            # Add disjunctive cut for job
            self.add_disj_cuts_jobs(j=t, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate cumulative disj. cuts of the 2nd kind (pos-based) + strenghten 1st kind
    def generate_disj_cuts_pos(self, reduced_coeffs: bool):
        for k in self.seq:
            for j in self.jobs:
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                if disjunction_lb is None:
                    return None

                # Strenghten current lb and update j-th cut
                if disjunction_lb > self.alpha[j - 1, k - 1]:
                    self.alpha[j - 1, k - 1] = disjunction_lb
                    self.update_disj_cut_job(j=j, reduce_coeffs=reduced_coeffs)

            # Add disjunctive cut for position
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate cumulative disj. cuts of the 2nd kind position-based first
    def generate_disj_cuts_pos_first(self, reduced_coeffs: bool):
        for k in self.seq:
            for t in self.jobs:
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                if disjunction_lb is None:
                    return None

                self.alpha[t - 1, k - 1] = disjunction_lb

            # Add disjunctive cut for position
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate cumulative disj. cuts of the 2nd kind using vars reduced costs + strengthen 1st kind
    def generate_disj_cuts_pos_quickrc(self, reduced_coeffs: bool, curr_bound: float):
        # Get xs var values and reduced costs
        xs_vars = self.get_x_var_vals()
        xs_reduced_costs = self.get_reduced_costs()

        for k in self.seq:
            for j in self.jobs:
                if xs_vars[f"x_{j}_{k}"] <= 0.001:
                    # Get lb from disjunction
                    disjunction_lb = self.get_disjunction_lb(j=j, k=k)
                else:
                    # Add reduced costs for the var to the current bound
                    disjunction_lb = np.ceil(curr_bound + xs_reduced_costs[f"x_{j}_{k}"])

                if disjunction_lb is None:
                    return None

                # Strenghten current lb and update j-th cut
                if disjunction_lb > self.alpha[j - 1, k - 1]:
                    self.alpha[j - 1, k - 1] = disjunction_lb
                    self.update_disj_cut_job(j=j, reduce_coeffs=reduced_coeffs)

            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

            curr_bound = self.solve_model(
                show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding
            )

        return curr_bound

    ############### METHODS FOR FRACTIONAL VARIABLES DISJ. CUTS APPROACH ###############
    # Generate both job-based and pos-based disj. cuts for all fractional x vars
    def generate_simultaneous_fract_var_disj_cuts(
        self, xs_fracts: np.ndarray, reduced_coeffs: bool
    ):
        # Initialize return variables at the start
        lb_2 = None
        lb_3 = None

        for x_fract in xs_fracts:
            parts = x_fract.split("_")
            j = int(parts[1])
            k = int(parts[2])

            lb_2 = self.generate_simultaneous_fract_disj_cuts_jobs(
                j=j, reduced_coeffs=reduced_coeffs
            )

            lb_3 = self.generate_simultaneous_fract_disj_cuts_pos(
                k=k, reduced_coeffs=reduced_coeffs
            )

        return lb_2, lb_3

    # Generate simultaneous job-based disj. cuts for fract. vars
    def generate_simultaneous_fract_disj_cuts_jobs(self, j: int, reduced_coeffs: bool):
        for h in self.seq:
            disjunction_lb = self.get_disjunction_lb(j=j, k=h)

            if disjunction_lb is None:
                print("None disjunction lb for j,k:", j, h)
                return None

            self.alpha[j - 1, h - 1] = max(self.alpha[j - 1, h - 1], disjunction_lb)

        if self.has_constraint(f"disj_cut_job_{j}"):
            self.pfm.remove_constraint(f"disj_cut_job_{j}")
            self.add_disj_cuts_jobs(j=j, reduce_coeffs=reduced_coeffs)
        else:
            self.add_disj_cuts_jobs(j=j, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate simultaneous pos-based disj. cuts for fract. vars
    def generate_simultaneous_fract_disj_cuts_pos(self, k: int, reduced_coeffs: bool):
        for t in self.jobs:
            disjunction_lb = self.get_disjunction_lb(j=t, k=k)

            if disjunction_lb is None:
                print("None disjunction lb for j,k:", t, k)
                return None

            self.alpha[t - 1, k - 1] = max(self.alpha[t - 1, k - 1], disjunction_lb)

        if self.has_constraint(f"disj_cut_pos_{k}"):
            self.pfm.remove_constraint(f"disj_cut_pos_{k}")
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)
        else:
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate job-based disj. cuts first and then pos-based disj. cuts for all fractional x vars
    def generate_sequential_fract_disj_cuts_jobs(self, j_fracts: set[int], reduced_coeffs: bool):
        for j in self.jobs_array:
            if j in j_fracts:
                for h in self.seq:
                    disjunction_lb = self.get_disjunction_lb(j=j, k=h)

                    if disjunction_lb is None:
                        print("None disjunction lb for j,k:", j, h)
                        return None

                    self.alpha[j - 1, h - 1] = max(self.alpha[j - 1, h - 1], disjunction_lb)

                if self.has_constraint(f"disj_cut_job_{j}"):
                    self.pfm.remove_constraint(f"disj_cut_job_{j}")
                    self.add_disj_cuts_jobs(j=j, reduce_coeffs=reduced_coeffs)
                else:
                    self.add_disj_cuts_jobs(j=j, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    def generate_sequential_fract_disj_cuts_pos(self, k_fracts: set[int], reduced_coeffs: bool):
        for k in self.seq:
            if k in k_fracts:
                for t in self.jobs:
                    disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                    if disjunction_lb is None:
                        print("None disjunction lb for j,k:", t, k)
                        return None

                    self.alpha[t - 1, k - 1] = max(self.alpha[t - 1, k - 1], disjunction_lb)

                if self.has_constraint(f"disj_cut_pos_{k}"):
                    self.pfm.remove_constraint(f"disj_cut_pos_{k}")
                    self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)
                else:
                    self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Generate disj. cuts using fractional x_j_k variables
    def generate_fract_var_disj_cuts_jobs(self, reduced_coeffs: bool, fractional_tol: float):
        # Get fractional x vars
        _, xs_fracs = self.get_fract_int_xs_vars(fract_tol=fractional_tol)

        for t in self.jobs_array:
            for k in self.seq:
                x_var = f"x_{t}_{k}"

                if x_var in xs_fracs:
                    # Get lb from disjunction
                    disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                    if disjunction_lb is None:
                        return None

                    # Save disjunction lb
                    if disjunction_lb > self.alpha[t - 1, k - 1]:
                        self.alpha[t - 1, k - 1] = disjunction_lb

            # Add disjunctive cut for job
            self.add_disj_cuts_jobs(j=t, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    def generate_fract_var_disj_cuts_jobs_pos_first(
        self, reduced_coeffs: bool, fractional_tol: float
    ):
        # Get fractional x vars
        _, xs_fracs = self.get_fract_int_xs_vars(fract_tol=fractional_tol)

        for t in self.jobs_array:
            for k in self.seq:
                x_var = f"x_{t}_{k}"

                if x_var in xs_fracs:
                    # Get lb from disjunction
                    disjunction_lb = self.get_disjunction_lb(j=t, k=k)

                    if disjunction_lb is None:
                        return None

                    # Strenghten current lb and update k-th cut
                    if disjunction_lb > self.alpha[t - 1, k - 1]:
                        self.alpha[t - 1, k - 1] = disjunction_lb
                        self.update_disj_cut_pos(k=k, reduce_coeffs=reduced_coeffs)

            # Add disjunctive cut for job
            self.add_disj_cuts_jobs(j=t, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    def generate_fract_var_disj_cuts_pos(self, reduced_coeffs: bool, fractional_tol: float):
        # Get fractional x vars
        _, xs_fracs = self.get_fract_int_xs_vars(fract_tol=fractional_tol)

        for k in self.seq:
            for j in self.jobs:
                x_var = f"x_{j}_{k}"

                if x_var in xs_fracs:
                    # Get lb from disjunction
                    disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                    if disjunction_lb is None:
                        return None

                    # Strenghten current lb and update j-th cut
                    if disjunction_lb > self.alpha[j - 1, k - 1]:
                        self.alpha[j - 1, k - 1] = disjunction_lb
                        self.update_disj_cut_job(j=j, reduce_coeffs=reduced_coeffs)

            # Add disjunctive cut for position
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    def generate_fract_var_disj_cuts_pos_first(self, reduced_coeffs: bool, fractional_tol: float):
        # Get fractional x vars
        _, xs_fracs = self.get_fract_int_xs_vars(fract_tol=fractional_tol)

        for k in self.seq:
            for j in self.jobs:
                x_var = f"x_{j}_{k}"

                if x_var in xs_fracs:
                    # Get lb from disjunction
                    disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                    if disjunction_lb is None:
                        return None

                    # Save disjunction lb
                    if disjunction_lb > self.alpha[j - 1, k - 1]:
                        self.alpha[j - 1, k - 1] = disjunction_lb

            # Add disjunctive cut for position
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduced_coeffs)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    ############### METHODS FOR FURTHER IMPROVED DISJ. CUTS APPROACH ###############
    # Further improve disjunctive cuts for all pairs of j, k
    def further_improve_disj_cuts(self, reduced_coeffs: bool) -> int | None:
        for k in self.seq:
            for j in self.jobs:
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                if disjunction_lb is None:
                    return None

                # Strenghten current lb and update j-th job-based and k-th pos-based cuts
                if disjunction_lb > self.alpha[j - 1, k - 1]:
                    self.alpha[j - 1, k - 1] = disjunction_lb
                    self.update_disj_cut_job(j=j, reduce_coeffs=reduced_coeffs)
                    self.update_disj_cut_pos(k=k, reduce_coeffs=reduced_coeffs)

        # Return re-optimised LP
        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Further improve cuts using a specific array of xs vars (p.e., fractional xs)
    def further_improve_disj_cuts_xs_array(self, xs_array: np.ndarray, reduced_coeffs: bool):
        for x_var in xs_array:
            j = int(x_var.split("_")[1])
            k = int(x_var.split("_")[2])

            # Get lb from disjunction
            disjunction_lb = self.get_disjunction_lb(j=j, k=k)

            if disjunction_lb is None:
                return None

            # Strenghten current lb and update j-th job-based and k-th pos-based cuts
            if disjunction_lb > self.alpha[j - 1, k - 1]:
                self.alpha[j - 1, k - 1] = disjunction_lb
                self.update_disj_cut_job(j=j, reduce_coeffs=reduced_coeffs)
                self.update_disj_cut_pos(k=k, reduce_coeffs=reduced_coeffs)

        # Return re-optimised LP
        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Solve MIP model with disjunctive cuts by setting priority order
    def solve_mip_disj_cuts_priority_ord(self, priority_by: str = "alpha"):
        # Change xs variable types to binary
        self.change_xs_var_types(var_type="binary")

        # Set priority order
        if priority_by == "alpha":
            self.set_priority_by_alpha_bounds()
        elif priority_by == "fractional":
            self.set_priority_by_fract_var()

        # Solve MIP model with disjunctive cuts
        improved_lb = self.solve_model(
            show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding
        )

        # Unset priority order
        self.unset_priority_order()

        return improved_lb

    # Solve MIP model with disjunctive cuts
    def solve_mip_disj_cuts(self):
        # Change xs variable types to binary
        self.change_xs_var_types(var_type="binary")

        # Solve MIP model with disjunctive cuts
        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    ############### METHODS FOR FURTHER DISJ. CUTS APPROACHES ###############
    # Generate further cumulative disj. cuts of the 1st kind
    def generate_further_disj_cuts_jobs(self, s: int, t: int, jobs: np.ndarray):
        for j in jobs:
            for k in range(1, t + 1):
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                if disjunction_lb is None:
                    return None

                # Save disjunction lb
                self.alpha[j - 1, k - 1] = disjunction_lb

            # Get lb after adding sum xs[j,k] = 0
            self.add_sum_xs_eq_zero(j=j, t=t)

            lbound = self.solve_model(
                show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding
            )

            if lbound is None:
                return None

            self.remove_sum_xs_eq_zero(j=j)

            # Store lb in alpha[j, t+1]
            self.alpha[j - 1, t] = lbound

            self.add_further_disj_cuts_jobs(j=j, s=s, t=t)

    # Generate further cumulative disj. cuts of the 2nd kind
    def generate_further_disj_cuts_pos(self, s: int, t: int, jobs: np.ndarray):
        for k in range(1, t + 1):
            for j in jobs:
                # Get lb from disjunction
                disjunction_lb = self.get_disjunction_lb(j=j, k=k)

                if disjunction_lb is None:
                    return None

                # Strenghten current lb and update j-th cut
                if disjunction_lb > self.alpha[j - 1, k - 1]:
                    self.alpha[j - 1, k - 1] = disjunction_lb
                    self.update_further_disj_cut_job(j=j, k=k, s=s, t=t)

            self.add_further_disj_cuts_pos(k=k, s=s, t=t)

    # Add sum of xs = 0 constraint
    def generate_further_sum_xs_eq_zero(self, t: int, jobs_array: np.ndarray):
        for j in jobs_array:
            # Get lb after adding sum xs[j,k] = 0
            self.add_sum_xs_eq_zero(j=j, t=t)

            lbound = self.solve_model(
                show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding
            )

            if lbound is None:
                return None

            self.remove_sum_xs_eq_zero(j=j)

            if lbound > self.alpha[j - 1, t]:
                self.alpha[j - 1, t] = lbound
                self.update_further_disj_cut_job(
                    j=j,
                )

    # Add sum_{k=1}^t x[j,k] = 0
    def add_sum_xs_eq_zero(self, j: int, t: int):
        # Create sum of xs = 0 constraint
        expr = self.pfm.linear_expr()

        for k in range(1, t + 1):
            expr.add_term(self.xs[j, k], 1)

        # Add constraint
        self.pfm.add(self.pfm.eq_constraint(expr, 0, name=f"sum_xs_eq_zero_{j}"))

    # Remove sum_{k=1}^t x[j,k] = 0
    def remove_sum_xs_eq_zero(self, j: int):
        self.pfm.remove(self.pfm.get_constraint_by_name(f"sum_xs_eq_zero_{j}"))

    # Add further disjunctive cut for JOBS (s, t)
    def add_further_disj_cuts_jobs(self, j: int, s: int, t: int):
        # Create disjunctive cut
        expr = self.pfm.linear_expr()
        expr.add_term(self.fs[s, t], 1)

        for k in range(1, t + 1):
            expr.add_term(self.xs[j, k], -(self.alpha[j - 1, k - 1] - self.alpha[j - 1, t]))

        self.pfm.add(
            self.pfm.ge_constraint(
                expr, self.alpha[j - 1, t], name=f"further_disj_cut_job_{j}_{s}_{t}"
            )
        )

    # Add further disjunctive cut for POSITIONS (s, t)
    def add_further_disj_cuts_pos(self, k: int, s: int, t: int):
        # Create disjunctive cut
        expr = self.pfm.linear_expr()
        expr.add_term(self.fs[s, t], 1)

        for j in self.jobs:
            expr.add_term(self.xs[j, k], -self.alpha[j - 1, k - 1])

        self.pfm.add(self.pfm.ge_constraint(expr, 0, name=f"further_disj_cut_pos_{k}_{s}_{t}"))

    # Update lower bound coefficient for variable in further cuts
    def update_further_disj_cut_job(self, j: int, k: int, s: int, t: int):
        # Get LHS of disjunctive cut constraint
        lhs_j = self.pfm.get_constraint_by_name(f"further_disj_cut_job_{j}_{s}_{t}").lhs

        # Remove coeff * x_jk term from constraint
        lhs_j.remove_term(self.xs[j, k])

        # Add updated coeff * x_jk term
        lhs_j.add_term(self.xs[j, k], -(self.alpha[j - 1, k - 1] - self.alpha[j - 1, t]))

    ############### METHODS FOR GETTING DISJUNCTIONS/ADDING DISJ. CUTS ###############
    # Add disjunctive cut for JOBS
    def add_disj_cuts_jobs(self, j: int, reduce_coeffs: bool = False):
        # Reduce the magnitude of xs coefficients (lbs)
        min_lb = 0
        if reduce_coeffs:
            min_lb = np.min(self.alpha[j - 1, :])

        # Create disjunctive cut
        expr = self.pfm.linear_expr()
        expr.add_term(self.fs[self.m, self.n], 1)

        for k in self.seq:
            expr.add_term(self.xs[j, k], min_lb - self.alpha[j - 1, k - 1])

        # Add constraint
        self.pfm.add(self.pfm.ge_constraint(expr, min_lb, name=f"disj_cut_job_{j}"))

    # Add disjunctive cut for POSITIONS
    def add_disj_cuts_pos(self, k: int, reduce_coeffs: bool = False):
        # Reduce the magnitude of xs coefficients (lbs)
        min_lb = 0
        if reduce_coeffs:
            min_lb = np.min(self.alpha[:, k - 1])

        # Create disjunctive cut
        expr = self.pfm.linear_expr()
        expr.add_term(self.fs[self.m, self.n], 1)

        for j in self.jobs:
            expr.add_term(self.xs[j, k], min_lb - self.alpha[j - 1, k - 1])

        # Add constraint
        self.pfm.add(self.pfm.ge_constraint(expr, min_lb, name=f"disj_cut_pos_{k}"))

    # Add naive disjunctive cuts from alpha array
    def add_naive_disjunctive_cuts(self, reduce_coeff: bool = False):
        for j in self.jobs:
            self.add_disj_cuts_jobs(j=j, reduce_coeffs=reduce_coeff)

        for k in self.seq:
            self.add_disj_cuts_pos(k=k, reduce_coeffs=reduce_coeff)

        return self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

    # Get disjunction lower bound by fixing x_{j, k} = 1
    def get_disjunction_lb(self, j: int, k: int):
        # Fix variable in j, k
        self.xs[(j, k)].lb = 1
        self.xs[(j, k)].ub = 1

        # Solve relaxed model with fixed var
        disj_lb = self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

        # Unfix variable
        self.xs[(j, k)].lb = 0
        self.xs[(j, k)].ub = 1e20

        return disj_lb

    # Update lower bound coefficient for variable in job-based cut
    def update_disj_cut_job(self, j: int, reduce_coeffs: bool):

        self.pfm.remove_constraint(f"disj_cut_job_{j}")
        self.add_disj_cuts_jobs(j=j, reduce_coeffs=reduce_coeffs)

    # Update lower bound coefficient for variable in pos-based cut
    def update_disj_cut_pos(self, k: int, reduce_coeffs: bool):

        self.pfm.remove_constraint(f"disj_cut_pos_{k}")
        self.add_disj_cuts_pos(k=k, reduce_coeffs=reduce_coeffs)

    ############### OTHER USEFUL METHODS ###############
    # Sort jobs by workload
    def sort_jobs(self, decreasing: bool = False):
        sum_ptimes = [(self.proc_times[:, j - 1].sum(), j) for j in self.jobs]
        sum_ptimes = sorted(sum_ptimes, reverse=decreasing)
        jobs_array = [tp[1] for tp in sum_ptimes]

        return jobs_array

    # Split x variable into fractional and integer
    def get_fract_int_xs_vars(self, fract_tol: float):
        xs_vars = self.get_x_var_vals()

        x_var_keys = np.array(list(xs_vars.keys()))
        x_var_vals = np.fromiter(xs_vars.values(), dtype=float)

        mask_int = np.isclose(x_var_vals, 0.0, atol=fract_tol) | np.isclose(
            x_var_vals, 1.0, atol=fract_tol
        )

        return x_var_keys[mask_int], x_var_keys[~mask_int]

    # Get j,k pairs for fractional x variables
    def get_jk_pairs_from_xs_fract_vars(self, xs_fract_vars: np.ndarray):
        j_set = {int(parts[1]) for parts in (name.split("_") for name in xs_fract_vars)}
        k_set = {int(parts[2]) for parts in (name.split("_") for name in xs_fract_vars)}

        return {"j": j_set, "k": k_set}

    # Set priority order using the alpha bounds
    def set_priority_by_alpha_bounds(self):
        # Get (j, k), lb pairs from alpha
        xs_bound_pairs = [((j, k), self.alpha[j - 1, k - 1]) for j in self.jobs for k in self.seq]

        # Create ORD triples to set priority order
        ord_triples = [
            (f"x_{j}_{k}", int(alpha), 0) for ((j, k), alpha) in xs_bound_pairs
        ]  # ("x_j_k", alpha, default)

        # Set priority order
        self.set_priority_order(triples_lst=ord_triples)

    # Set priority order using fractional vars
    def set_priority_by_fract_var(self):
        xs_vars = self.get_x_var_vals()

        ord_triples = [
            (x_var, int(round(x_val * 1000, 0)), 0)
            for (x_var, x_val) in xs_vars.items()
            if x_val < 1
        ]

        # Set priority order
        self.set_priority_order(triples_lst=ord_triples)

    # Get LP Relaxation lower bound
    def get_lp_relaxation_lb(self):
        # Build relaxed model and set CPLEX parameters
        self.build_model(lp_relaxed=True)
        self.set_model_parameters(
            n_threads=self.n_threads,
            num_emphasis=self.num_emphasis,
            markowitz_tol=self.markowitz_tol,
            scaling=self.scaling,
            feasibility_tol=self.feasibility_tol,
            mip_emphasis=self.mip_emphasis,
            mip_solution_limit=self.mip_sol_limit,
            time_limit=self.time_limit,
            workmem=self.workmem,
            node_file_strategy=self.node_file_strategy,
            workdir=self.workdir,
            no_cuts=self.no_cuts,
        )

        # Solve relaxed model and save lb
        lp_lb = self.solve_model(show_log=self.show_output, rounding_thrsh=self.obj_diff_rounding)

        return lp_lb

    ############### CPLEX METHODS ###############
    # Set CPLEX parameter values by the user before building the model
    def set_cplex_params_user(
        self,
        show_output: bool = False,
        obj_diff_rounding: float = 0.1,
        n_threads: int = None,
        num_emphasis: int = 0,
        markowitz_tol: float = 0.01,
        scaling: int = 0,
        feasibility_tol: float = 1e-6,
        mip_emphasis: int = 0,
        mip_sol_lim: int = None,
        time_limit: int = None,
        workmem: int = None,
        node_file_strategy: int = 1,
        workdir: str = None,
        no_cuts: bool = False,
    ):
        """Set CPLEX parameters before building the model.

        Args:
            show_output (bool, optional): Shows the output log when calling the solve() method. Defaults to False.
            obj_diff_rounding (float, optional): Tolerance for the difference applied to the objective values when
                                    rounding up to the nearest integer. Defaults to 0.1.
            n_threads (int, optional): CPLEX Global Thread Count. Sets CPLEX parallel optimisation mode and the default
                                    number of threads that will be invoked. The number of threads is limited by available
                                    processors and Processor Value Units (PVU). Defaults to None (Automatic: let CPLEX
                                    decide).
            num_emphasis (int, optional): CPLEX Numerical Emphasis. Emphasizes precision in numerically unstable or
                                    difficult problems. Possible values include: 0 (Do not emphasize numerical precision),
                                    1 (Exercise extreme caution in computation). Defaults to 0.
            markowitz_tol (float, optional): CPLEX Markowitz Tolerance influences pivot selection during basis factoring.
                                    The value should be between 0.0001 and 0.99999. Defaults to 0.01.
            scaling (int, optional): CPLEX Scale parameter. Decides how to scale the problem matrix. Possible values
                                    include 0 (Equilibration scaling), 1 (More agressive scaling) and -1 (No scaling).
                                    Defaults to 0.
            feasibility_tol (float, optional): CPLEX Feasibility Tolerance. Specifies the maximum primal and dual
                                    infeasibilities allowed in a solution. Any values from 1e-9 to 1e-1 are allowed.
                                    Defaults to 1e-6.
            mip_emphasis (int, optional): CPLEX MIP Emphasis. Controls trade-offs between speed, feasibility, optimality,
                                    and moving bounds in MIP. Possible alternatives include: 0: BALANCED (Default),
                                    1: FEASIBILITY, 2: OPTIMALITY, 3: BESTBOUND, 4: HIDDENFEAS.
            mip_sol_lim (int, optional): CPLEX MIP Integer Solution Limit. Sets the number of MIP solutions to be found
                                    before stopping. Defaults to None (9223372036800000000 solutions).
            time_limit (int, optional): CPLEX Optimiser Time Limit in Seconds. Sets the maximum time, in seconds, for
                                    a call to an optimizer. Clock type for computation time is set to Wall Clock Time
                                    (total physical time elapsed). Defaults to None.
            workmem (int, optional): Working memory limit in MB. Controls memory available for working storage.
                                    When exceeded, CPLEX may use node files. Defaults to None (CPLEX default ~128MB).
            node_file_strategy (int, optional): Node file storage strategy. 0: No node files, 1: Compressed in memory,
                                    2: Uncompressed to disk, 3: Compressed to disk. Defaults to 1.
            workdir (str, optional): Working directory for temporary files. Used when node_file_strategy > 1.
                                    Defaults to None (system temp directory).
            no_cuts (bool, optional): If True, disables all cutting planes in MIP. Defaults to False.
        """
        # Logging and objective function tolerance
        self.show_output = show_output
        self.obj_diff_rounding = obj_diff_rounding

        # Parallelism
        self.n_threads = n_threads

        # Simplex parameters
        if num_emphasis not in {0, 1}:
            raise ValueError("Invalid num_emphasis parameter value. Must be one in {0, 1}.")
        else:
            self.num_emphasis = num_emphasis

        if markowitz_tol < 0.0001 or markowitz_tol > 0.99999:
            raise ValueError(
                "Invalid markowitz_tol parameter value. Must be between 0.0001 and 0.99999."
            )
        else:
            self.markowitz_tol = markowitz_tol

        if scaling not in {0, 1, -1}:
            raise ValueError("Invalid scaling parameter value. Must be one in {0, 1, -1}.")
        else:
            self.scaling = scaling

        if feasibility_tol < 1e-9 or feasibility_tol > 1e-1:
            raise ValueError(
                "Invalid feasibility_tol parameter value. Must be between 1e-9 and 1e-1."
            )
        else:
            self.feasibility_tol = feasibility_tol

        # MIP parameters
        if mip_emphasis not in {0, 1, 2, 3, 4}:
            raise ValueError(
                "Invalid mip_emphasis parameter value. Must be one in {0, 1, 2, 3, 4}."
            )
        else:
            self.mip_emphasis = mip_emphasis

        self.mip_sol_limit = mip_sol_lim
        self.time_limit = time_limit

        # Memory management parameters
        self.workmem = workmem

        if node_file_strategy not in {0, 1, 2, 3}:
            raise ValueError(
                "Invalid node file strategy parameter value. Must be one in {0, 1, 2, 3}."
            )
        else:
            self.node_file_strategy = node_file_strategy

        if workdir is None:
            self.workdir = "."
        else:
            self.workdir = workdir

        self.no_cuts = no_cuts

    ############### TIMER DECORATOR ###############
    # def timer(func):
    #     def timer_wrapper(self, *args, **kwargs):

    #         start_timer = time()
    #         result = func(self, *args, **kwargs)
    #         elapsed_time = time() - start_timer

    #         if self.time_keeper is None:
    #             print(f'Time for {func.__name__} = {elapsed_time:.3f}')
    #         else:
    #             self.time_keeper.append(f'Time for {func.__name__} = {elapsed_time:.3f}\n')

    #         return result
    #     return timer_wrapper
