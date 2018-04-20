import sys

import matplotlib.pyplot as plt


def parselog(l):
    date, time, error = l.split(' ')
    error = float(error.strip())
    return date, time, error


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py path/training_error.log")
        sys.exit(-1)
    logpath = sys.argv[1]

    with open(logpath, 'r') as log:
        y = []
        for l in log:
            _, _, error = parselog(l)
            y.append(error)

    plt.plot(y)
    plt.show()
