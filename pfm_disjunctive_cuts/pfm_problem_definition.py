import os

import numpy as np
import pandas as pd


# Class for defining the the PFM problem instance
class PFMproblem:
    # Constructor
    def __init__(
        self,
        data_file: str = None,
        inst_name: str = None,
        n: int = None,
        m: int = None,
        proc_times: list = None,
        best: int = None,
    ):
        if data_file == None:
            self.n = n
            self.m = m
            self.jobs = np.arange(1, n + 1)
            self.machines = np.arange(1, m + 1)
            self.seq = np.arange(1, n + 1)
            self.proc_times = np.array(proc_times)
            self.best = best
        else:
            self.data_file = data_file
            self.inst_name = inst_name

            if inst_name not in {"taillard", "vallada"}:
                raise ValueError("Argument inst_name should be either 'taillard' or 'vallada'")
            elif inst_name == "taillard":
                self.get_taillard()
            elif inst_name == "vallada":
                self.get_vallada()

    @staticmethod
    def _get_data_root() -> str:
        package_dir = os.path.dirname(os.path.realpath(__file__))
        repo_dir = os.path.dirname(package_dir)
        return os.path.join(repo_dir, "data")

    def _get_instance_path(self, subdir: str) -> str:
        path = os.path.join(self._get_data_root(), subdir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory does not exist: {path}")
        return path

    # Method for reading Taillard data instances
    def get_taillard(self):
        # Get Taillard instances file
        path = self._get_instance_path("taillard_instances")
        file_path = os.path.join(path, self.data_file)

        with open(file_path) as f:
            lines = f.readlines()

        prob_info = []
        line1 = lines[0].split()
        for val in line1:
            prob_info.append(int(val))

        self.n = prob_info[0]
        self.m = prob_info[1]
        self.best = prob_info[3]

        self.jobs = np.arange(1, self.n + 1)
        self.machines = np.arange(1, self.m + 1)
        self.seq = np.arange(1, self.n + 1)

        p_times = []
        for l in lines[1:]:
            ln = [int(x) for x in l.split()]
            p_times.append(ln)

        self.proc_times = np.array(p_times)

        if self.proc_times.shape[0] != self.m or self.proc_times.shape[1] != self.n:
            raise ValueError("Procesing times must have (m, n) shape")

    # Method for reading Vallada et al. data instances
    def get_vallada(self):
        # Get Vallada instances file
        path = self._get_instance_path("vallada_etal_instances")

        vallada_ubs = pd.read_csv(os.path.join(path, "Vallada-bounds.csv"), index_col=0).to_dict()[
            "ub"
        ]

        file_path = os.path.join(path, self.data_file)

        with open(file_path) as f:
            lines = f.readlines()

        prob_info = []
        line1 = lines[0].split()
        for val in line1:
            prob_info.append(int(val))

        self.n = prob_info[0]
        self.m = prob_info[1]
        self.best = vallada_ubs[self.data_file]

        self.jobs = np.arange(1, self.n + 1)
        self.machines = np.arange(1, self.m + 1)
        self.seq = np.arange(1, self.n + 1)

        p_times = []
        for l in lines[1:]:
            ln = [int(l.split()[i]) for i in range(1, self.m * 2, 2)]
            p_times.append(ln)

        self.proc_times = np.array(p_times).transpose()

        if self.proc_times.shape[0] != self.m or self.proc_times.shape[1] != self.n:
            raise ValueError("Procesing times must have (m, n) shape")
