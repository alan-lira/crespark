from configparser import ConfigParser
from math import e, log
from numpy import array, ndarray
from pathlib import Path
from scipy.optimize import nnls
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Learner:

    def __init__(self,
                 beta_coefficients_learner_config_file: Path,
                 crespark_optimizer_config_file: Path) -> None:
        self.beta_coefficients_learner_config_file = beta_coefficients_learner_config_file
        self.beta_coefficients_learner_config_parser = None
        self.crespark_optimizer_config_file = crespark_optimizer_config_file
        self.crespark_optimizer_config_parser = None
        self.training_dataset_input_file_path = None
        self.training_dataset_input_parser = None
        self.testing_dataset_input_file_path = None
        self.testing_dataset_input_parser = None

    @staticmethod
    def __load_config_parser(config_file: Path) -> ConfigParser:
        config_parser = ConfigParser()
        # Case Preservation of Each Option Name
        config_parser.optionxform = str
        # Load config_parser
        config_parser.read(config_file,
                           encoding="utf-8")
        return config_parser

    def __load_beta_coefficients_learner_config_parser(self) -> ConfigParser:
        return self.__load_config_parser(self.beta_coefficients_learner_config_file)

    def __set_beta_coefficients_learner_config_parser(self,
                                                      beta_coefficients_learner_config_parser: ConfigParser) -> None:
        self.beta_coefficients_learner_config_parser = beta_coefficients_learner_config_parser

    def __get_beta_coefficients_learner_config_parser(self) -> ConfigParser:
        return self.beta_coefficients_learner_config_parser

    def __load_crespark_optimizer_config_parser(self) -> ConfigParser:
        return self.__load_config_parser(self.crespark_optimizer_config_file)

    def __set_crespark_optimizer_config_parser(self,
                                               crespark_optimizer_config_parser: ConfigParser) -> None:
        self.crespark_optimizer_config_parser = crespark_optimizer_config_parser

    def __get_crespark_optimizer_config_parser(self) -> ConfigParser:
        return self.crespark_optimizer_config_parser

    def __get_training_dataset_input_file_path(self) -> Path:
        exception_message = "{0}: 'training_dataset_input_file' must be a valid path file!" \
            .format(self.beta_coefficients_learner_config_file)
        try:
            training_dataset_input_file_path = \
                Path(self.beta_coefficients_learner_config_parser.get("Datasets Input Settings",
                                                                      "training_dataset_input_file"))
        except ValueError:
            raise ValueError(exception_message)
        return training_dataset_input_file_path

    def __get_testing_dataset_input_file_path(self) -> Path:
        exception_message = "{0}: 'testing_dataset_input_file' must be a valid path file!" \
            .format(self.beta_coefficients_learner_config_file)
        try:
            testing_dataset_input_file_path = \
                Path(self.beta_coefficients_learner_config_parser.get("Datasets Input Settings",
                                                                      "testing_dataset_input_file"))
        except ValueError:
            raise ValueError(exception_message)
        return testing_dataset_input_file_path

    def __load_datasets_input_settings(self) -> None:
        self.training_dataset_input_file_path = self.__get_training_dataset_input_file_path()
        self.testing_dataset_input_file_path = self.__get_testing_dataset_input_file_path()

    def __load_training_dataset_input_parser(self) -> ConfigParser:
        return self.__load_config_parser(self.training_dataset_input_file_path)

    def __set_training_dataset_input_parser(self,
                                            training_dataset_input_parser: ConfigParser) -> None:
        self.training_dataset_input_parser = training_dataset_input_parser

    def __get_training_dataset_input_parser(self) -> ConfigParser:
        return self.training_dataset_input_parser

    def __load_testing_dataset_input_parser(self) -> ConfigParser:
        return self.__load_config_parser(self.testing_dataset_input_file_path)

    def __set_testing_dataset_input_parser(self,
                                           testing_dataset_input_parser: ConfigParser) -> None:
        self.testing_dataset_input_parser = testing_dataset_input_parser

    def __get_testing_dataset_input_parser(self) -> ConfigParser:
        return self.testing_dataset_input_parser

    @staticmethod
    def __calculate_x0() -> float:
        return 1

    @staticmethod
    def __calculate_x1(M: int,
                       Tc: int) -> float:
        return M / Tc

    @staticmethod
    def __calculate_x2(M: int,
                       R: int,
                       Tc: int) -> float:
        return (M * R) / Tc

    @staticmethod
    def __calculate_x3(R: int,
                       Tc: int) -> float:
        return Tc / R

    @staticmethod
    def __calculate_x4(M: int,
                       R: int) -> float:
        return (M * log(M, e)) / R

    @staticmethod
    def __calculate_x5(M: int,
                       R: int) -> float:
        return M / R

    @staticmethod
    def __calculate_x6(M: int) -> float:
        return M

    @staticmethod
    def __calculate_x7(R: int) -> float:
        return R

    def __load_actual_y(self) -> list:
        actual_y = []
        number_of_testing_experiments = 0
        for section in self.testing_dataset_input_parser.sections():
            if "Experiment Index" in section:
                number_of_testing_experiments = number_of_testing_experiments + 1
                exception_message = "Please fill all the '{0}' fields of '{1}' file!" \
                    .format("runtime_in_seconds", self.testing_dataset_input_file_path)
                try:
                    runtime_in_seconds = \
                        float(self.testing_dataset_input_parser.get(section,
                                                                    "runtime_in_seconds"))
                except ValueError:
                    raise ValueError(exception_message)
                actual_y.append(runtime_in_seconds)
        print("Number of testing experiments: {0}".format(number_of_testing_experiments))
        loaded_actual_y_message = "Successfully loaded 'actual_y' using the experiments from the '{0}' file.\n-------" \
            .format(self.testing_dataset_input_file_path)
        print(loaded_actual_y_message)
        return actual_y

    def __load_a_matrix(self) -> array:
        a_matrix = []
        number_of_training_experiments = 0
        for section in self.training_dataset_input_parser.sections():
            if "Experiment Index" in section:
                number_of_training_experiments = number_of_training_experiments + 1
                M = int(self.training_dataset_input_parser.get(section,
                                                               "M"))
                R = int(self.training_dataset_input_parser.get(section,
                                                               "R"))
                iota_w = int(self.training_dataset_input_parser.get(section,
                                                                    "iota_w"))
                gamma_w = int(self.training_dataset_input_parser.get(section,
                                                                     "gamma_w"))
                Tc = iota_w * gamma_w
                a_matrix.append([self.__calculate_x0(),
                                 self.__calculate_x1(M, Tc),
                                 self.__calculate_x2(M, R, Tc),
                                 self.__calculate_x3(R, Tc),
                                 self.__calculate_x4(M, R),
                                 self.__calculate_x5(M, R),
                                 self.__calculate_x6(M),
                                 self.__calculate_x7(R)])
        print("Number of training experiments: {0}".format(number_of_training_experiments))
        loaded_a_matrix_message = "Successfully loaded 'A' matrix using the experiments from the '{0}' file." \
            .format(self.training_dataset_input_file_path)
        print(loaded_a_matrix_message)
        return array(a_matrix)

    def __load_b_vector(self) -> array:
        b_vector = []
        for section in self.training_dataset_input_parser.sections():
            if "Experiment Index" in section:
                exception_message = "Please fill all the '{0}' fields of '{1}' file!" \
                    .format("runtime_in_seconds", self.training_dataset_input_file_path)
                try:
                    runtime_in_seconds = \
                        float(self.training_dataset_input_parser.get(section,
                                                                     "runtime_in_seconds"))
                except ValueError:
                    raise ValueError(exception_message)
                b_vector.append(runtime_in_seconds)
        loaded_b_vector_message = "Successfully loaded 'b' vector using the experiments from the '{0}' file.\n-------" \
            .format(self.training_dataset_input_file_path)
        print(loaded_b_vector_message)
        return array(b_vector)

    @staticmethod
    def __solve_non_negative_least_squares_problem(a_matrix: array,
                                                   b_vector: array) -> ndarray:
        beta_coefficients = nnls(a_matrix, b_vector)[0]
        beta_coefficients_list = list(beta_coefficients)
        print("NNLS problem solved!\nBETA COEFFICIENTS:")
        for i in range(len(beta_coefficients_list)):
            print("   - β{0}: {1}".format(i,
                                     str(beta_coefficients_list[i])))
        print("-------")
        return beta_coefficients

    def __calculate_time_cost_function(self,
                                       beta_coefficients: ndarray,
                                       M: int,
                                       R: int,
                                       Tc: int) -> float:
        beta0 = beta_coefficients[0]
        beta1 = beta_coefficients[1]
        beta2 = beta_coefficients[2]
        beta3 = beta_coefficients[3]
        beta4 = beta_coefficients[4]
        beta5 = beta_coefficients[5]
        beta6 = beta_coefficients[6]
        beta7 = beta_coefficients[7]
        x0 = self.__calculate_x0()
        x1 = self.__calculate_x1(M, Tc)
        x2 = self.__calculate_x2(M, R, Tc)
        x3 = self.__calculate_x3(R, Tc)
        x4 = self.__calculate_x4(M, R)
        x5 = self.__calculate_x5(M, R)
        x6 = self.__calculate_x6(M)
        x7 = self.__calculate_x7(R)
        return (beta0 * x0) + \
               (beta1 * x1) + \
               (beta2 * x2) + \
               (beta3 * x3) + \
               (beta4 * x4) + \
               (beta5 * x5) + \
               (beta6 * x6) + \
               (beta7 * x7)

    def __load_predicted_y(self,
                           beta_coefficients: ndarray) -> list:
        predicted_y = []
        for section in self.testing_dataset_input_parser.sections():
            if "Experiment Index" in section:
                M = int(self.testing_dataset_input_parser.get(section,
                                                              "M"))
                R = int(self.testing_dataset_input_parser.get(section,
                                                              "R"))
                iota_w = int(self.testing_dataset_input_parser.get(section,
                                                                   "iota_w"))
                gamma_w = int(self.testing_dataset_input_parser.get(section,
                                                                    "gamma_w"))
                Tc = iota_w * gamma_w
                T = self.__calculate_time_cost_function(beta_coefficients, M, R, Tc)
                predicted_y.append(T)
        loaded_predicted_y_message = "Successfully calculated 'predicted_y' applying the learned Beta coefficients " \
                                     "into the experiments from the '{0}' file.\n-------" \
            .format(self.testing_dataset_input_file_path)
        print(loaded_predicted_y_message)
        return predicted_y

    @staticmethod
    def __get_and_print_regression_metrics_scores(actual_y: list,
                                                  predicted_y: list) -> None:
        print("REGRESSION MODEL METRICS:")
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual_y,
                                  predicted_y)
        print("1) Mean Absolute Error (MAE): {0} (Best: 0.0)".format(str(mae)))
        # Mean Squared Error (MSE)
        mse = mean_squared_error(actual_y,
                                 predicted_y)
        print("2) Mean Squared Error (MSE): {0} (Best: 0.0)".format(str(mse)))
        # Root Mean Squared Error (RMSE)
        rmse = mean_squared_error(actual_y,
                                  predicted_y,
                                  squared=False)
        print("3) Root Mean Squared Error (RMSE): {0} (Best: 0.0)".format(str(rmse)))
        # R² Score (Coefficient of Determination)
        r2 = r2_score(actual_y,
                      predicted_y)
        print("4) R² Score (Coefficient of Determination): {0} (Best: 1.0)\n-------".format(str(r2)))

    def __update_beta_coefficients_on_crespark_optimizer_config_file(self,
                                                                     beta_coefficients: ndarray) -> None:
        section_name = "Beta Coefficients"
        self.crespark_optimizer_config_parser.set(section_name, "beta_zero", str(beta_coefficients[0]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_one", str(beta_coefficients[1]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_two", str(beta_coefficients[2]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_three", str(beta_coefficients[3]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_four", str(beta_coefficients[4]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_five", str(beta_coefficients[5]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_six", str(beta_coefficients[6]))
        self.crespark_optimizer_config_parser.set(section_name, "beta_seven", str(beta_coefficients[7]))
        with open(self.crespark_optimizer_config_file, "w", encoding="utf-8") as crespark_optimizer_config_file:
            self.crespark_optimizer_config_parser.write(crespark_optimizer_config_file)
        print("Updated the '{0}' file with the learned Beta coefficients.\n-------"
              .format(self.crespark_optimizer_config_file))

    def learn(self):
        # Load and Set Beta Coefficients Learner Config Parser
        beta_coefficients_learner_config_parser = self.__load_beta_coefficients_learner_config_parser()
        self.__set_beta_coefficients_learner_config_parser(beta_coefficients_learner_config_parser)
        # Load and Set CRESPark Optimizer Config Parser
        crespark_optimizer_config_parser = self.__load_crespark_optimizer_config_parser()
        self.__set_crespark_optimizer_config_parser(crespark_optimizer_config_parser)
        # Load Datasets Input Settings
        self.__load_datasets_input_settings()
        # Load and Set Training Dataset Input Parser
        training_dataset_input_parser = self.__load_training_dataset_input_parser()
        self.__set_training_dataset_input_parser(training_dataset_input_parser)
        # Load and Set Testing Dataset Input Parser
        testing_dataset_input_parser = self.__load_testing_dataset_input_parser()
        self.__set_testing_dataset_input_parser(testing_dataset_input_parser)
        # Load Actual Y (y_true)
        actual_y = self.__load_actual_y()
        # Load "A" Matrix, i.e., Train X (Independent Variables)
        a_matrix = self.__load_a_matrix()
        # Load "b" Vector, i.e., Train Y (Dependent Variable)
        b_vector = self.__load_b_vector()
        # Solve the Non-Negative Least Squares (NNLS) Problem
        beta_coefficients = self.__solve_non_negative_least_squares_problem(a_matrix,
                                                                            b_vector)
        # Load Predicted Y (y_pred)
        predicted_y = self.__load_predicted_y(beta_coefficients)
        # Get and Print Regression Metrics Scores (MAE, MSE, RMSE & R²)
        self.__get_and_print_regression_metrics_scores(actual_y,
                                                       predicted_y)
        # Update the Beta Coefficients on "crespark_optimizer.cfg" File
        self.__update_beta_coefficients_on_crespark_optimizer_config_file(beta_coefficients)
