import os
import sys
from time import time

import docplex.mp.model as cpx
import numpy as np

from .pfm_problem_definition import PFMproblem


# Class to build PFM MIP formulation model using docplex
class PFMmip(PFMproblem):
    # Constructor
    def __init__(self, data_file=None, inst_name=None, n=None, m=None, proc_times=None, best=None):
        super().__init__(data_file, inst_name, n, m, proc_times, best)

    # Build MIP or LP relaxation
    def build_model(self, lp_relaxed: bool = False):
        self.pfm = cpx.Model("Stafford_MIP")
        self.add_variables(lp_relaxed)
        self.add_obj_function()
        self.add_job_assignment_const()
        self.add_pos_assignment_const()
        self.add_finishing_first_const()
        self.add_pos_finishing_const()
        self.add_mach_finishing_const()

        self.cpx = self.pfm.get_cplex()

    # Add variables
    def add_variables(self, lp_relaxed):
        if lp_relaxed:
            self.xs = {
                (j, k): self.pfm.continuous_var(name=f"x_{j}_{k}", lb=0)
                for j in self.jobs
                for k in self.seq
            }
        else:
            self.xs = {
                (j, k): self.pfm.binary_var(name=f"x_{j}_{k}") for j in self.jobs for k in self.seq
            }

        self.fs = {
            (i, k): self.pfm.continuous_var(name=f"f_{i}_{k}", lb=0)
            for i in self.machines
            for k in self.seq
        }

    # Add objective function: makespan
    def add_obj_function(self):
        self.pfm.minimize(self.fs[self.m, self.n])

    # Add job assignment constraints
    def add_job_assignment_const(self):
        for j in self.jobs:
            expr = self.pfm.linear_expr()
            for k in self.seq:
                expr.add_term(self.xs[j, k], 1)

            self.pfm.add(self.pfm.eq_constraint(expr, 1))

    # Add position assignment constraints
    def add_pos_assignment_const(self):
        for k in self.seq:
            expr = self.pfm.linear_expr()
            for j in self.jobs:
                expr.add_term(self.xs[j, k], 1)

            self.pfm.add(self.pfm.eq_constraint(expr, 1))

    # Add finishing first position constraints
    def add_finishing_first_const(self):
        expr = self.pfm.linear_expr()
        expr.add_term(self.fs[1, 1], 1)
        for j in self.jobs:
            expr.add_term(self.xs[j, 1], -self.proc_times[0, j - 1])

        self.pfm.add(self.pfm.eq_constraint(expr, 0))

    # Add position-based finishing times constraints
    def add_pos_finishing_const(self):
        for i in self.machines:
            for k in self.seq[:-1]:
                expr = self.pfm.linear_expr()
                expr.add_term(self.fs[i, k + 1], 1)
                expr.add_term(self.fs[i, k], -1)

                for j in self.jobs:
                    expr.add_term(self.xs[j, k + 1], -self.proc_times[i - 1, j - 1])

                self.pfm.add(self.pfm.ge_constraint(expr, 0))

    # Add machine-based finishing times constraints
    def add_mach_finishing_const(self):
        for i in self.machines[:-1]:
            for k in self.seq:
                expr = self.pfm.linear_expr()
                expr.add_term(self.fs[i + 1, k], 1)
                expr.add_term(self.fs[i, k], -1)

                for j in self.jobs:
                    expr.add_term(self.xs[j, k], -self.proc_times[i, j - 1])

                self.pfm.add(self.pfm.ge_constraint(expr, 0))

    # Set CPLEX model parameters
    def set_model_parameters(
        self,
        n_threads: int = None,
        num_emphasis: int = 0,
        markowitz_tol: float = 0.01,
        scaling: int = 0,
        feasibility_tol: float = 1e-6,
        mip_emphasis: int = 0,
        mip_solution_limit: int = None,
        time_limit: int = None,
        workmem: int = None,
        node_file_strategy: int = 1,
        workdir: str = None,
        no_cuts: bool = False,
    ):
        # Change to parallelisation and set number of threads
        if n_threads:
            self.pfm.parameters.parallel = -1
            self.pfm.parameters.threads = n_threads

        # Set numerical emphasis
        if num_emphasis != 0:
            self.pfm.parameters.emphasis.numerical = num_emphasis

        # Set Markowitz tolerance
        if markowitz_tol != 0.01:
            self.pfm.parameters.simplex.tolerances.markowitz = markowitz_tol

        # Set scaling
        if scaling != 0:
            self.pfm.parameters.read.scale = scaling

        # Set feasibility tolerance
        if feasibility_tol != 1e-6:
            self.pfm.parameters.simplex.tolerances.feasibility = feasibility_tol

        # Set the feasibility-optimality MIP emphasis
        if mip_emphasis != 0:
            self.pfm.parameters.emphasis.mip = mip_emphasis

        # Set the number of MIP solutions limit
        if mip_solution_limit:
            self.pfm.parameters.mip.limits.solutions = mip_solution_limit

        # Set the time limit (real-elapsed time)
        if time_limit:
            self.pfm.parameters.clocktype = 2  # Wall clock time
            self.pfm.parameters.timelimit = time_limit

        # Set working storage memory
        if workmem:
            self.pfm.parameters.workmem = workmem

        # Set node file switch strategy
        if node_file_strategy != 1:
            self.pfm.parameters.mip.strategy.file = node_file_strategy

        # Set working directory for node files
        if workdir:
            if not os.path.exists(workdir):
                try:
                    os.makedirs(workdir, exist_ok=True)
                    print(f"Created working directory: {workdir}")
                except Exception as e:
                    print(f"Error creating directory {workdir}: {e}")

            self.pfm.parameters.workdir = workdir

        if no_cuts:
            self.pfm.parameters.mip.cuts.gomory = -1
            self.pfm.parameters.mip.cuts.mircut = -1
            self.pfm.parameters.mip.cuts.covers = -1
            self.pfm.parameters.mip.cuts.flowcovers = -1
            self.pfm.parameters.mip.cuts.cliques = -1
            self.pfm.parameters.mip.cuts.disjunctive = -1
            self.pfm.parameters.mip.cuts.zerohalfcut = -1
            self.pfm.parameters.mip.cuts.liftproj = -1
            self.pfm.parameters.mip.cuts.bqp = -1
            self.pfm.parameters.mip.cuts.mcfcut = -1
            self.pfm.parameters.mip.limits.cutpasses = -1
            self.pfm.parameters.mip.cuts.gubcovers = -1
            self.pfm.parameters.mip.cuts.implied = -1
            self.pfm.parameters.mip.cuts.localimplied = -1
            self.pfm.parameters.mip.cuts.pathcut = -1
            self.pfm.parameters.mip.cuts.nodecuts = -1

    # Solve MIP/LP model.
    def solve_model(self, show_log: bool = False, rounding_thrsh: float = 0.1) -> int | None:
        if not hasattr(self, "pfm") or self.pfm is None:
            raise ValueError("The model needs to be defined before solving it!")

        self.pfm_sol = self.pfm.solve(clean_before_solve=False, log_output=show_log)

        c_max = None
        # Check if the solution exists
        if self.pfm_sol is None:
            # print the pfm_sol status
            print(
                f"PFM solution not found for {self.data_file}! Solution status: {self.pfm.solve_status}, {self.pfm.solve_details}"
            )

            try:
                cpx = self.pfm.get_engine().get_cplex()
                c_max = float(cpx.solution.get_objective_value())

                # Quantify infeasibility (quality metrics)
                q = cpx.solution.get_quality_metrics()

                print(q)

            except Exception as e:
                print(f"CPLEX fallback failed to provide an objective: {e}")
                return None
                # raise ValueError('PFM solution is infeasible or a Null value!')

        # Return the best bound if the solution hits a time limit
        elif self.pfm.solve_details.has_hit_limit():
            c_max = self.pfm.solve_details.best_bound

        else:
            c_max = self.pfm_sol.get_objective_value()

        fract_part, int_part = np.modf(c_max)

        if fract_part < rounding_thrsh:
            c_max = int(int_part)
        else:
            c_max = int(int_part + 1.0)

        return c_max

    # Get variable values
    def get_var_values(self):
        try:
            self.pfm_sol
        except NameError:
            print("The model needs to be solved first!")
            sys.exit()

        xs_vars = self.get_x_var_vals()
        fs_vars = self.get_f_var_vals()

        return {"x_vars": xs_vars, "f_vars": fs_vars}

    # Get x variable values
    def get_x_var_vals(self):
        xs_vars = {}
        for x_var in self.xs.values():
            var_val = self.pfm_sol.get_value(x_var)
            xs_vars[x_var.name] = var_val

        return xs_vars

    # Get f variable values
    def get_f_var_vals(self):
        fs_vars = {}
        for f_var in self.fs.values():
            var_val = self.pfm_sol.get_value(f_var)
            fs_vars[f_var.name] = var_val

        return fs_vars

    # Check if a constraint exists
    def has_constraint(self, constraint_name: str):
        return self.pfm.get_constraint_by_name(constraint_name) is not None

    # Add fixed variable constraints for the 'relax-and-fix' algorithm -> X_{jk} = 1
    def add_fixed_var_const(self, j: int, k: int, const_name: str = None):
        expr = self.pfm.linear_expr()
        expr.add_term(self.xs[j, k], 1)
        self.pfm.add(self.pfm.eq_constraint(expr, 1), const_name)

    # Add fixed variable constraints for the local search algorithm -> X_{jk} = 0
    def add_fixed_var_const_zero(self, j: int, k: int, const_name: str = None):
        expr = self.pfm.linear_expr()
        expr.add_term(self.xs[j, k], 1)
        self.pfm.add(self.pfm.eq_constraint(expr, 0), const_name)

    # Add MIP start (warm start)
    def add_mip_solution(self, xs_vars: dict):
        self.pfm.clear_mip_starts()
        mip_start = self.pfm.new_solution(var_value_dict=xs_vars)
        self.pfm.add_mip_start(mip_start_sol=mip_start)

    # Get reduced costs of xs vars for LPs
    def get_reduced_costs(self):
        xs_red_costs = {}
        for x_var in self.xs.values():
            rc = x_var.reduced_cost
            xs_red_costs[x_var.name] = rc

        return xs_red_costs

    # Change variable types
    def change_xs_var_types(self, var_type: str = "binary"):
        if var_type not in {"binary", "continuous"}:
            raise ValueError("Variable type must be 'binary', or 'continuous'.")

        try:
            self.pfm.change_var_types(self.xs, var_type)
        except:
            self.pfm.change_var_types(list(self.xs.values()), var_type)

    # Set CPLEX priority order for MIP solver
    def set_priority_order(self, triples_lst: list):
        # Set the order parameter
        self.pfm.parameters.mip.strategy.order = 1

        self.cpx.order.set(triples_lst)
        # self.cpx.order.write('dc_mip_order.ord')

    # Unset CPLEX priority order
    def unset_priority_order(self):
        # Set the order parameter
        self.pfm.parameters.mip.strategy.order = 0

    # Get CPLEX model number of nodes processed
    def get_num_nodes_processed(self) -> int:
        if self.pfm.solve_details is None:
            raise ValueError("The model needs to be solved first!")

        return self.pfm.solve_details.nb_nodes_processed
