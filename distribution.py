import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from datetime import datetime


class Distribution:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        self.data = []

    def read_data(self, filename, sample=True):
        with open(filename) as file:
            data_list = file.read()
            data_list = [float(i) for i in data_list.split('\n')]
        file.close()

        self.data = data_list


class Gaussian(Distribution):
    def __init__(self, mean=0, std=1):
        Distribution.__init__(self, mean, std)

    def calculate_mean(self):
        self.mean = np.mean(self.data)

        return(self.mean)

        pass

    def calculate_std(self, sample=True):
        self.std = np.std(self.data)

        return(self.std)

        pass

    def plot_histo(self, n_spaces=50):
        mean = self.mean
        std = self.std
        minimum = min(self.data)
        maximum = max(self.data)
        interval = (maximum - minimum) / n_spaces

        distribution = norm(mean, std)

        x = []
        y = []

        for i in range(n_spaces):
            tmp = minimum + interval * i
            x.append(tmp)
            y.append(distribution.pdf(tmp))

        fig, axes = plt.subplots(2, sharex=True)
        fig.subplots_adjust(hspace=0.5)
        axes[0].hist(self.data, density=True)
        axes[1].plot(x, y)

        plt.show()

        return(x, y)

    def __repr__(self):
        return("mean {}, standard deviation {}".format(self.mean, self.std))
