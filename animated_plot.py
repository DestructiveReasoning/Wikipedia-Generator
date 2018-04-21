import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read(logfile):
    with open(logfile) as f:
        data = []
        while True:
            line = f.readline()
            time.sleep(0.1)
            if line:
                _, _, error = line.split(' ')
                error = float(error.strip())
                data.append(error)
                yield data


def animate(values):
    x = list(range(len(values)))
    line.set_data(x, values)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(values), max(values))
    return line,


fig, ax = plt.subplots()
line, = ax.plot([])


ani = FuncAnimation(fig, animate,
                    frames=read('models/experiment1/training_error.log'),
                    interval=10, blit=True)
plt.show()
