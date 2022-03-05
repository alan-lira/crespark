from beta_coefficients_learner.learner import Learner
from crespark_optimizer.optimizer import Optimizer
from pathlib import Path
from sys import argv


def check_if_has_valid_number_of_arguments(argv_list: list) -> None:
    number_of_arguments_expected = 2
    arguments_expected_list = ["beta_coefficients_learner_config_file",
                               "crespark_optimizer_config_file"]
    number_of_arguments_provided = len(argv_list) - 1
    if number_of_arguments_provided != number_of_arguments_expected:
        number_of_arguments_expected_message = \
            "".join([str(number_of_arguments_expected),
                     " arguments were" if number_of_arguments_expected > 1 else " argument was"])
        number_of_arguments_provided_message = \
            "".join([str(number_of_arguments_provided),
                     " arguments were" if number_of_arguments_provided > 1 else " argument was"])
        invalid_number_of_arguments_message = \
            "Invalid number of arguments provided!\n" \
            "{0} expected: {1}\n" \
            "{2} provided: {3}".format(number_of_arguments_expected_message,
                                       ", ".join(arguments_expected_list),
                                       number_of_arguments_provided_message,
                                       ", ".join(argv_list[1:]))
        raise ValueError(invalid_number_of_arguments_message)


def check_if_file_exists(file_path: Path) -> None:
    if not file_path.is_file():
        file_not_found_message = "'{0}' not found. The application will halt!".format(str(file_path))
        raise FileNotFoundError(file_not_found_message)


def main(argv_list: list):
    # Begin
    # Print Application Start Notice
    print("CRESPark Started!\n-------")
    # Check If Has Valid Number of Arguments
    check_if_has_valid_number_of_arguments(argv_list)
    # Read Beta Coefficients Learner Config File
    beta_coefficients_learner_config_file = Path(argv_list[1])
    # Check If Beta Coefficients Learner Config File Exists
    check_if_file_exists(beta_coefficients_learner_config_file)
    # Read CRESPark Optimizer Config File
    crespark_optimizer_config_file = Path(argv_list[2])
    # Check If CRESPark Optimizer Config File Exists
    check_if_file_exists(crespark_optimizer_config_file)
    # Init 'Learner' Object
    bcl = Learner(beta_coefficients_learner_config_file,
                  crespark_optimizer_config_file)
    # Learn Beta Coefficients
    bcl.learn()
    # Delete 'Learner' Object
    del bcl
    # Init 'Optimizer' Object
    o = Optimizer(crespark_optimizer_config_file)
    # Optimize
    o.optimize()
    # Delete 'Optimizer' Object
    del o
    # Print Application End Notice
    print("-------\nCRESPark Finished Successfully!")
    # End
    exit(0)


if __name__ == "__main__":
    main(argv)
