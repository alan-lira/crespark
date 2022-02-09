from beta_coefficients_learner.learner import Learner
from crespark_optimizer.optimizer import Optimizer


def main():
    # Begin

    # Print Application Start Notice
    print("CRESPark Started!\n-------")

    # Init 'Learner' Object
    bcl = Learner()

    # Learn Beta Coefficients
    bcl.learn()

    # Delete 'Learner' Object
    del bcl

    # Init 'Optimizer' Object
    o = Optimizer()

    # Optimize
    o.optimize()

    # Delete 'Optimizer' Object
    del o

    # Print Application End Notice
    print("-------\nCRESPark Finished Successfully!")

    # End
    exit(0)


if __name__ == "__main__":
    main()
