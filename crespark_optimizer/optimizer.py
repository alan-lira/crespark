from configparser import ConfigParser
from gurobipy import Env, GRB, Model
from math import ceil, inf, log
from pathlib import Path
from time import time


class Optimizer:

    def __init__(self) -> None:
        self.optimizer_config_file_path = Path("config/optimizer.cfg")
        self.optimizer_config_parser = None
        self.beta_zero = 0.0
        self.beta_one = 0.0
        self.beta_two = 0.0
        self.beta_three = 0.0
        self.beta_four = 0.0
        self.beta_five = 0.0
        self.beta_six = 0.0
        self.beta_seven = 0.0
        self.M = 0
        self.R = 0
        self.gamma = 0
        self.upsilon = 0.0
        self.phi = 0.0
        self.tau = 0.0
        self.Tc = 0
        self.Tc_lower_bound = 0
        self.Tc_upper_bound = 0
        self.monetary_unit = None
        self.time_unit = None
        self.alfa_zero = 0.0
        self.alfa_one = 0.0
        self.alfa_two = 0.0
        self.alfa_three = 0.0
        self.optimization_problem = 0
        self.optimization_problem_description = None
        self.optimization_modes = None
        self.T = inf
        self.C = inf
        self.nu = 0

    @staticmethod
    def __load_config_parser(config_file: Path) -> ConfigParser:
        config_parser = ConfigParser()
        # Case Preservation of Each Option Name
        config_parser.optionxform = str
        # Load config_parser
        config_parser.read(config_file,
                           encoding="utf-8")
        return config_parser

    def __load_optimizer_config_parser(self) -> ConfigParser:
        return self.__load_config_parser(self.optimizer_config_file_path)

    def __set_optimizer_config_parser(self,
                                      optimizer_config_parser: ConfigParser) -> None:
        self.optimizer_config_parser = optimizer_config_parser

    def __get_optimizer_config_parser(self) -> ConfigParser:
        return self.optimizer_config_parser

    def __get_beta_i(self,
                     optimizer_config_parser: ConfigParser,
                     beta_i: str) -> float:
        exception_message = "{0}: '{1}' must be a float value equal or higher than zero!" \
            .format(self.optimizer_config_file_path, beta_i)
        try:
            beta_i_value = float(optimizer_config_parser.get("Beta Coefficients", beta_i))
            if beta_i_value < 0.0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return beta_i_value

    def __load_beta_coefficients(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        self.beta_zero = self.__get_beta_i(optimizer_config_parser, "beta_zero")
        self.beta_one = self.__get_beta_i(optimizer_config_parser, "beta_one")
        self.beta_two = self.__get_beta_i(optimizer_config_parser, "beta_two")
        self.beta_three = self.__get_beta_i(optimizer_config_parser, "beta_three")
        self.beta_four = self.__get_beta_i(optimizer_config_parser, "beta_four")
        self.beta_five = self.__get_beta_i(optimizer_config_parser, "beta_five")
        self.beta_six = self.__get_beta_i(optimizer_config_parser, "beta_six")
        self.beta_seven = self.__get_beta_i(optimizer_config_parser, "beta_seven")

    def __get_M(self,
                optimizer_config_parser: ConfigParser) -> int:
        exception_message = "{0}: 'M' (Number of Map Tasks) must be a integer value higher than zero!" \
            .format(self.optimizer_config_file_path)
        try:
            M = int(optimizer_config_parser.get("Input Parameters", "M"))
            if M <= 0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return M

    def __get_R(self,
                optimizer_config_parser: ConfigParser) -> int:
        exception_message = "{0}: 'R' (Number of Reduce Tasks) must be a integer value higher than zero!" \
            .format(self.optimizer_config_file_path)
        try:
            R = int(optimizer_config_parser.get("Input Parameters", "R"))
            if R <= 0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return R

    def __get_gamma(self,
                    optimizer_config_parser: ConfigParser) -> int:
        exception_message = "{0}: 'gamma' (Number of Cores per Worker) must be a integer value higher than zero!" \
            .format(self.optimizer_config_file_path)
        try:
            gamma = int(optimizer_config_parser.get("Input Parameters", "gamma"))
            if gamma <= 0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return gamma

    def __get_upsilon(self,
                      optimizer_config_parser: ConfigParser) -> float:
        exception_message = "{0}: 'upsilon' (Monetary Cost, per hour, to rent one Worker) " \
                            "must be a float value higher than zero!" \
            .format(self.optimizer_config_file_path)
        try:
            upsilon = float(optimizer_config_parser.get("Input Parameters", "upsilon"))
            if upsilon <= 0.0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return upsilon

    def __get_phi(self,
                  optimizer_config_parser: ConfigParser) -> float:
        exception_message = "{0}: 'phi' (Monetary Budget Constraint) " \
                            "must be a float value higher than zero!" \
            .format(self.optimizer_config_file_path)
        try:
            phi = float(optimizer_config_parser.get("Input Parameters", "phi"))
            if phi <= 0.0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return phi

    def __get_tau(self,
                  optimizer_config_parser: ConfigParser) -> float:
        exception_message = "{0}: 'tau' (Deadline Constraint in Hours) " \
                            "must be a float value higher than zero!" \
            .format(self.optimizer_config_file_path)
        try:
            tau = float(optimizer_config_parser.get("Input Parameters", "tau"))
            if tau <= 0.0:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return tau

    def __load_input_parameters(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        self.M = self.__get_M(optimizer_config_parser)
        self.R = self.__get_R(optimizer_config_parser)
        self.gamma = self.__get_gamma(optimizer_config_parser)
        self.upsilon = self.__get_upsilon(optimizer_config_parser)
        self.phi = self.__get_phi(optimizer_config_parser)
        self.tau = self.__get_tau(optimizer_config_parser)

    def __get_Tc_bounds(self,
                        optimizer_config_parser: ConfigParser) -> list:
        exception_message = "{0}: 'Tc' (Total Number of Available Cores) must have valid integer bounds!" \
            .format(self.optimizer_config_file_path)
        Tc_bounds = []
        try:
            Tc_bounds_list = optimizer_config_parser.get("Decision Variables Bounds", "Tc").split("...")
            Tc_lower = int(Tc_bounds_list[0])
            if Tc_lower <= 0:
                raise ValueError(exception_message)
            else:
                Tc_bounds.append(Tc_lower)
            Tc_higher = int(Tc_bounds_list[1])
            if Tc_higher <= 0 or Tc_higher < Tc_lower:
                raise ValueError(exception_message)
            else:
                Tc_bounds.append(Tc_higher)
        except ValueError:
            raise ValueError(exception_message)
        return Tc_bounds

    def __load_Tc_bounds(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        Tc_bounds = self.__get_Tc_bounds(optimizer_config_parser)
        self.Tc_lower_bound = Tc_bounds[0]
        self.Tc_upper_bound = Tc_bounds[1]

    def __get_monetary_unit(self,
                            optimizer_config_parser: ConfigParser) -> str:
        valid_monetary_unit_list = ["USD"]
        exception_message = "{0}: supported 'monetary_unit' values: {1}." \
            .format(self.optimizer_config_file_path, " | ".join(valid_monetary_unit_list))
        try:
            monetary_unit = str(optimizer_config_parser.get("General Settings", "monetary_unit"))
            if monetary_unit not in valid_monetary_unit_list:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return monetary_unit

    def __load_monetary_unit(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        self.monetary_unit = self.__get_monetary_unit(optimizer_config_parser)

    def __get_time_unit(self,
                        optimizer_config_parser: ConfigParser) -> str:
        valid_time_unit_list = ["second", "minute", "hour"]
        exception_message = "{0}: supported 'time_unit' values: {1}." \
            .format(self.optimizer_config_file_path, " | ".join(valid_time_unit_list))
        try:
            time_unit = str(optimizer_config_parser.get("General Settings", "time_unit"))
            if time_unit not in valid_time_unit_list:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return time_unit

    def __load_time_unit(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        self.time_unit = self.__get_time_unit(optimizer_config_parser)

    def __convert_time_unit_dependent_parameters(self) -> None:
        if self.time_unit == "minute":
            self.upsilon = self.upsilon / 60  # Price of renting one VM instance per minute
            self.tau = self.tau * 60  # Maximum amount of time τ, in minutes, for finishing the job (Time constraint)
        elif self.time_unit == "second":
            self.upsilon = self.upsilon / 3600  # Price of renting one VM instance per second
            self.tau = self.tau * 3600  # Maximum amount of time τ, in seconds, for finishing the job (Time constraint)

    def __calculate_alfa_constants(self) -> None:
        self.alfa_zero = self.beta_zero + \
                         self.beta_four * ((self.M * log(self.M)) / self.R) + \
                         self.beta_five * (self.M / self.R) + \
                         self.beta_six * self.M + \
                         self.beta_seven * self.R
        self.alfa_one = self.beta_one * self.M
        self.alfa_two = self.beta_two * self.M * self.R
        self.alfa_three = self.beta_three / self.R

    def __get_optimization_problem(self,
                                   optimizer_config_parser: ConfigParser) -> int:
        valid_optimization_problem_list = ["1", "2", "3"]
        exception_message = "{0}: supported 'optimization_problem' values: {1}." \
            .format(self.optimizer_config_file_path, " | ".join(valid_optimization_problem_list))
        try:
            optimization_problem = int(optimizer_config_parser.get("General Settings", "optimization_problem"))
            if str(optimization_problem) not in valid_optimization_problem_list:
                raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return optimization_problem

    def __load_optimization_problem(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        self.optimization_problem = self.__get_optimization_problem(optimizer_config_parser)
        if self.optimization_problem == 1:
            self.optimization_problem_description = \
                "Minimize the job time, given a maximum monetary cost φ = " \
                + str(self.phi) + " " + str(self.monetary_unit)
        elif self.optimization_problem == 2:
            self.optimization_problem_description = \
                "Minimize the monetary cost, given a maximum amount of time τ = " \
                + str(self.tau) + " " + str(self.time_unit) + "(s)"
        elif self.optimization_problem == 3:
            self.optimization_problem_description = "Minimize the monetary cost (without deadline constraint)"

    def __get_optimization_modes(self,
                                 optimizer_config_parser: ConfigParser) -> list:
        valid_optimization_mode_list = ["brute_force", "gurobi"]
        exception_message = "{0}: supported 'optimization_modes' values: {1}." \
            .format(self.optimizer_config_file_path, " | ".join(valid_optimization_mode_list))
        try:
            optimization_modes = str(optimizer_config_parser.get("General Settings", "optimization_modes")).split(", ")
            for optimization_mode in optimization_modes:
                if optimization_mode not in valid_optimization_mode_list:
                    raise ValueError(exception_message)
        except ValueError:
            raise ValueError(exception_message)
        return optimization_modes

    def __load_optimization_modes(self) -> None:
        optimizer_config_parser = self.__get_optimizer_config_parser()
        self.optimization_modes = self.__get_optimization_modes(optimizer_config_parser)

    def __calculate_time_cost_function(self,
                                       Tc: int) -> float:
        T = self.alfa_zero + self.alfa_one / Tc + self.alfa_two / Tc + self.alfa_three * Tc
        if self.time_unit == "hour":
            T = T / 3600
        elif self.time_unit == "minute":
            T = T / 60
        return T

    def __calculate_monetary_cost(self,
                                  T: float,
                                  Tc: int) -> float:
        C = self.upsilon * T * Tc / self.gamma
        return C

    def __is_constraint_not_violated(self,
                                     C: float,
                                     T: float) -> bool:
        if self.optimization_problem == 1:
            return C <= self.phi
        if self.optimization_problem == 2:
            return T <= self.tau

    def __reset_model_results(self) -> None:
        self.Tc = 0
        self.T = inf
        self.C = inf
        self.nu = 0

    def __optimize_model_with_brute_force(self) -> None:
        if self.optimization_problem == 1:
            # Minimize T(M, R, Tc)
            # Subject To
            # upsilon * [T(M, R, Tc)] * [Tc / gamma] <= phi
            for Tc_candidate in range(self.Tc_lower_bound, self.Tc_upper_bound + 1):
                T_candidate = self.__calculate_time_cost_function(Tc_candidate)
                C_candidate = self.__calculate_monetary_cost(T_candidate, Tc_candidate)
                if self.__is_constraint_not_violated(C_candidate, T_candidate):
                    if T_candidate < self.T:
                        self.Tc = Tc_candidate
                        self.T = T_candidate
                        self.C = C_candidate
        elif self.optimization_problem == 2:
            # Minimize upsilon * [T(M, R, Tc)] * [Tc / gamma]
            # Subject To
            # T(M, R, Tc) <= tau
            for Tc_candidate in range(self.Tc_lower_bound, self.Tc_upper_bound + 1):
                T_candidate = self.__calculate_time_cost_function(Tc_candidate)
                C_candidate = self.__calculate_monetary_cost(T_candidate, Tc_candidate)
                if self.__is_constraint_not_violated(C_candidate, T_candidate):
                    if C_candidate < self.C:
                        self.Tc = Tc_candidate
                        self.T = T_candidate
                        self.C = C_candidate
        elif self.optimization_problem == 3:
            # Minimize upsilon * [T(M, R, Tc)] * [Tc / gamma]
            for Tc_candidate in range(self.Tc_lower_bound, self.Tc_upper_bound + 1):
                T_candidate = self.__calculate_time_cost_function(Tc_candidate)
                C_candidate = self.__calculate_monetary_cost(T_candidate, Tc_candidate)
                if C_candidate < self.C:
                    self.Tc = Tc_candidate
                    self.T = T_candidate
                    self.C = C_candidate

    def __optimize_model_with_gurobi(self) -> None:
        with Env() as env, Model(name="Spark Application Time Cost Model Optimization on Gurobi for Python",
                                 env=env) as model:
            # Set Model Parameters
            model.setParam("NonConvex", 2)
            # Set Model Decision Variables
            Tc = model.addVar(name="Tc",
                              vtype=GRB.INTEGER,
                              lb=self.Tc_lower_bound,
                              ub=self.Tc_upper_bound)
            z0 = model.addVar(name="z0",
                              vtype=GRB.CONTINUOUS)  # z0 = α0
            z1 = model.addVar(name="z1",
                              vtype=GRB.CONTINUOUS)  # z1 = α1 / Tc -> z1 * Tc = α1
            z2 = model.addVar(name="z2",
                              vtype=GRB.CONTINUOUS)  # z2 = α2 / Tc -> z2 * Tc = α2
            z3 = model.addVar(name="z3",
                              vtype=GRB.CONTINUOUS)  # z3 = α3 * Tc
            # Set Time Cost Function: T(M, R, Tc) = α0 + α1 / Tc + α2 / Tc + α3 * Tc
            T = z0 + z1 + z2 + z3
            if self.time_unit == "hour":
                T = T / 3600
            elif self.time_unit == "minute":
                T = T / 60
            # Set Model Objective and Constraints
            if self.optimization_problem == 1:
                # Minimize T(M, R, Tc)
                # Subject To
                # upsilon * [T(M, R, Tc)] * [Tc / gamma] <= phi
                model.setObjective(T, GRB.MINIMIZE)
                model.addConstr(self.upsilon * T * (Tc / self.gamma) <= self.phi, "c0")
            elif self.optimization_problem == 2:
                # Minimize upsilon * [T(M, R, Tc)] * [Tc / gamma]
                # Subject To
                # T(M, R, Tc) <= tau
                model.setObjective(self.upsilon * T * (Tc / self.gamma), GRB.MINIMIZE)
                model.addConstr(T <= self.tau, "c0")
            elif self.optimization_problem == 3:
                # Minimize upsilon * [T(M, R, Tc)] * [Tc / gamma]
                model.setObjective(self.upsilon * T * (Tc / self.gamma), GRB.MINIMIZE)
            model.addConstr(self.alfa_zero == z0, "c1")  # z0 = α0
            model.addConstr(self.alfa_one == z1 * Tc, "c2")  # z1 = α1 / Tc -> z1 * Tc = α1
            model.addConstr(self.alfa_two == z2 * Tc, "c3")  # z2 = α2 / Tc -> z2 * Tc = α2
            model.addConstr(self.alfa_three * Tc == z3, "c4")  # z3 = α3 * Tc
            # Optimize Model
            model.optimize()
            # If Model is Feasible (Found Optimal Value, GRB.OPTIMAL)...
            if model.status == 2:
                for v in model.getVars():
                    if str(v.varName) == "Tc":
                        self.Tc = ceil(v.x)
                self.T = T.getValue()
                self.C = self.__calculate_monetary_cost(self.T, self.Tc)
            del env
            del model

    def __calculate_optimal_number_of_worker_nodes(self) -> None:
        self.nu = ceil(self.Tc / self.gamma)

    def __print_optimization_results(self,
                                     optimization_mode: str,
                                     optimization_time_in_seconds: time) -> None:
        print("-------------------- {0} ({1} seconds) -------------------".format(optimization_mode.upper(),
                                                                                  optimization_time_in_seconds))
        if self.Tc > 0:
            print("Total Number of Worker Nodes: {0}".format(self.nu))
            print("Total Number of Available Cores (Tc): {0}".format(self.Tc))
            print("Estimated Time Cost: {0} {1}(s)".format(self.T, self.time_unit))
            print("Estimated Monetary Cost: {0} {1}".format(round(self.C, 4), self.monetary_unit))
        else:
            print("MODEL IS INFEASIBLE!")

    def __optimize_model_with_available_optimization_modes(self) -> None:
        print("Spark Application Optimization Problem \"{0}\" - {1}:"
              .format(self.optimization_problem,
                      self.optimization_problem_description))
        for mode in self.optimization_modes:
            self.__reset_model_results()
            optimization_start_time = time()
            if mode == "brute_force":
                self.__optimize_model_with_brute_force()
            if mode == "gurobi":
                self.__optimize_model_with_gurobi()
            optimization_end_time = time() - optimization_start_time
            self.__calculate_optimal_number_of_worker_nodes()
            self.__print_optimization_results(mode, optimization_end_time)

    def optimize(self):

        # Load and Set Optimizer Config Parser
        optimizer_config_parser = self.__load_optimizer_config_parser()
        self.__set_optimizer_config_parser(optimizer_config_parser)

        # Load and Set Beta Coefficients (β0, β1, β2, β3, β4, β5, β6, β7)
        self.__load_beta_coefficients()

        # Load and Set Input Parameters (M, R, gamma, upsilon, phi, tau)
        # M: Total Number of Map Tasks
        # R: Total Number of Reduce Tasks
        # gamma: Fixed Number of Cores (vCPUs) per Node for a Specific Cluster Setup (γ)
        # upsilon: Price of Renting One Node for an Hour (υ)
        # phi: Monetary Budget Constraint (φ)
        # tau: Deadline Constraint (τ)
        self.__load_input_parameters()

        # Load Tc Bounds (Lower and Upper)
        self.__load_Tc_bounds()

        # Load Monetary Unit [USD]
        self.__load_monetary_unit()

        # Load Time Unit [second | minute | hour]
        self.__load_time_unit()

        # Convert Time Unit Dependent Parameters (υ, τ)
        self.__convert_time_unit_dependent_parameters()

        # Calculate and Set Alfa Constants (α0, α1, α2, α3)
        self.__calculate_alfa_constants()

        # Load Optimization Problem [1 | 2 | 3]
        # 1: Given a Maximum Monetary Cost φ for Finishing the Job (Monetary Budget Constraint),
        #    Find the Best Resource Allocation to Minimize the Job Time
        # 2: Given a Maximum Amount of Time τ for Finishing the Job (Deadline Constraint),
        #    Find the Best Resource Allocation to Minimize the Monetary Cost
        # 3: Find the Most Economical Solution for the Job Without the Deadline Constraint
        self.__load_optimization_problem()

        # Load Optimization Modes [brute_force | gurobi]
        self.__load_optimization_modes()

        # Optimize Model With Available Optimization Modes
        self.__optimize_model_with_available_optimization_modes()
