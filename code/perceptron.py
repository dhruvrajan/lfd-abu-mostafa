import numpy as np
import matplotlib.pyplot as plt


def plot_function(f, low=0, high=10):
    fx = np.random.uniform(low, high, 100)
    fy = list(map(f, fx))

    plt.plot(fx, fy)

class Perceptron:
    def __init__(self, dimension, initialize_weights=None, pocket=False, max_iter = 10000):
        self.weights = initialize_weights \
            if initialize_weights \
            else np.zeros(dimension + 1)
        # self.weights = []
        self.pocket = pocket
        self.best_weights = self.weights
        self.best_count = float('inf')
        self.data = []
        self.max_iter = max_iter


    def _augment(self, X):
        return np.insert(X, 0, 1)

    def _signal(self, X):
        augmented = self._augment(X)
        # print(augmented)
        # print(self.weights)
        return np.dot(self.weights, augmented)

    def plot(self):
        a = -self.weights[1] / self.weights[2]
        b = - self.weights[0] / self.weights[2]

        f = lambda x: a * x + b

        plot_function(f)

    def best_plot(self):
        a = -self.best_weights[1] / self.best_weights[2]
        b = - self.best_weights[0] / self.best_weights[2]

        f = lambda x: a * x + b

        plot_function(f)

    def _classify(self, X):

        return np.sign(self._signal(X))

    def _misclassified(self):
        for X, y in self.data:
            if not self._classify(X) == y:
                yield X, y

    def train(self, x_data, y_labels):
        """
        Update the model with new data
        :param y_labels:
        :param x_data:
        :return:
        """
        print(x_data, y_labels)
        self.data.extend(zip(x_data, y_labels))
        iter = 0
        while True:
            ex = False
            count = 0
            for X, y in self._misclassified():
                self.weights += y * self._augment(X)
                ex = True
                count += 1

                iter += 1


            if count < self.best_count:
                pocket_weights = self.weights
                self.best_count = count


            if not ex or iter > self.max_iter:
                # no misclassified points
                break



    def test(self, X, y):
        return self._classify(X), y


def signal(X, w):
    return np.dot(X, w)


def get_linearly_separable_2d(w):
    clf = lambda x: signal(x, w)



def linearly_separable_2d(f, low=0, high=10, size=100, margin=15, spread=50):
    """
    Create a 2-D linearly separable dataset around the function f
    :param f: The separating line
    :return:
    """


    x1_data = np.random.uniform(low, high, size)

    def transform(x):
        # random value on interval [-1.0, 1.0)
        r = (np.random.rand() - 0.5) * 2 * spread
        noise = np.sign(r) * margin + r

        return f(x) + noise

    x2_data = np.array(list(map(transform, x1_data)))

    def clf(x1, x2):
        return np.sign(x2 - f(x1))

    clf_data = [clf(x1, x2) for x1, x2 in zip(x1_data, x2_data)]


    data = list(zip(x1_data, x2_data, clf_data))
    positive = list(filter(lambda x: x[2] >= 0, data))
    negative = list(filter(lambda x: x[2] < 0, data))

    positive_x1 = [x1 for x1, x2, clf in positive]
    positive_x2 = [x2 for x1, x2, clf in positive]
    plt.plot(positive_x1, positive_x2, "ro")


    negative_x1 = [x1 for x1, x2, clf in negative]
    negative_x2 = [x2 for x1, x2, clf in negative]
    plt.plot(negative_x1, negative_x2, "bo")

    # plot_function(f)

    # fx = np.random.uniform(low, high, size)
    # fy = list(map(f, fx))
    #
    # plt.plot(fx, fy)


    return x1_data, x2_data, clf_data


def linear_with_overlap(f, low=0, high=10, size=100, margin=0.3, spread=5):
    """
    Create a 2-D linearly separable dataset around the function f
    :param f: The separating line
    :return:
    """


    x1_data = np.random.uniform(low, high, size)

    def transform(x):
        # random value on interval [-1.0, 1.0)
        r = (np.random.rand() - 0.5) * 2 * spread
        noise = np.sign(r) * margin + r

        return f(x) + r

    x2_data = np.array(list(map(transform, x1_data)))

    def clf(x1, x2):
        return np.sign(x2 - f(x1))

    clf_data = [clf(x1, x2) for x1, x2 in zip(x1_data, x2_data)]


    data = list(zip(x1_data, x2_data, clf_data))
    positive = list(filter(lambda x: x[2] >= 0, data))
    negative = list(filter(lambda x: x[2] < 0, data))

    positive_x1 = [x1 for x1, x2, clf in positive]
    positive_x2 = [x2 for x1, x2, clf in positive]
    plt.plot(positive_x1, positive_x2, "ro")


    negative_x1 = [x1 for x1, x2, clf in negative]
    negative_x2 = [x2 for x1, x2, clf in negative]
    plt.plot(negative_x1, negative_x2, "bo")

    plot_function(f, low, high)

    # fx = np.random.uniform(low, high, size)
    # fy = list(map(f, fx))
    #
    # plt.plot(fx, fy)


    return x1_data, x2_data, clf_data

# w0 * 1 + w1 * x1 + w2 * x2 = 0
# x2 = -w1 * x1 / w2 - w0 / w2


if __name__ == "__main__":
    target = lambda x:  -10 * x ** -5
    x1, x2, clf = linearly_separable_2d(lambda x:  3 * x  + 4)
    p = Perceptron(2)
    p.train(np.array(list(zip(x1, x2))), clf)
    # plot_function(target)
    p.plot()




    # plot_function(f)
    # p.plot()
    # a = -p.best_weights[1] / p.best_weights[2]
    # b = - p.best_weights[0] / p.best_weights[2]
    #
    # g = lambda x: a * x + b
    # plot_function(g)

    # print(p.weights)
    plt.show()
