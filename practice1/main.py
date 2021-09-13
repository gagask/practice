import matplotlib.pyplot as plt
from scipy import stats
import numpy


class taskOne:
    def __init__(self):
        self.X = [69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76]
        self.Y = [153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140,  150, 165, 185, 210, 220]

    def taskA(self):
        print("Mean:", numpy.mean(self.X))
        print("Median:", numpy.median(self.X))
        print("Mode:", stats.mode(self.X).mode[0])

    def taskB(self):
        # print("Dispersion:", numpy.mean([i ** 2 for i in self.Y]) - numpy.mean(self.Y) ** 2)
        print("Dispersion:", numpy.var(self.Y))

    def taskC(self):
        plt.plot(sorted(self.X), stats.norm.pdf(sorted(self.X), numpy.mean(self.X), numpy.std(self.X)))
        plt.show()

    def taskD(self):
        print("P(Age > 80) =", sum(stats.norm.pdf(list(range(81, 200)), numpy.mean(self.X), numpy.std(self.X))))

    def taskE(self):
        print("Two-dimensional expectation:", [numpy.mean(self.X), numpy.mean(self.Y)])
        print("Covariance matrix:\n", numpy.cov(self.X, self.Y, ddof=0))

    def taskF(self):
        print("Correlation coefficient", numpy.corrcoef(self.X, self.Y)[0][1])

    def taskG(self):
        plt.plot(self.X, self.Y, 'o')
        plt.show()


class taskTwo:
    def __init__(self):
        pass


class taskThree:
    def __init__(self):
        self.m = [4, 8]
        self.std = [1, 2]
        self.vars = [5, 6, 7]

    def taskA(self):
        for i in self.vars:
            if stats.norm.pdf(i, self.m[0], self.std[0]) >= stats.norm.pdf(i, self.m[1], self.std[1]):
                print(i, "- a")
            else:
                print(i, "- b")

    def taskB(self):
        N = 10000
        for i in range(5*N, 6*N):
            if abs(stats.norm.pdf(i/N, self.m[0], self.std[0]) - stats.norm.pdf(i/N, self.m[1], self.std[1])) < 0.1/N:
                print(i/N)
                print("P for a:", stats.norm.pdf(i/10000, self.m[0], self.std[0]))
                print("P for b:", stats.norm.pdf(i/10000, self.m[1], self.std[1]))
                return


def main():
    taskOne().taskC()


if __name__ == '__main__':
    main()
