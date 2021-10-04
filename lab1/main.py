import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn import preprocessing


class PredObr:
    def __init__(self):
        self.df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
        self.df = self.df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
        self.data = self.df.to_numpy(dtype='float')
        # print("data:\n", self.data)

    def before_after(self):
        print("Before standardization:")
        print("Math:", [numpy.mean(i) for i in self.data.T])
        print("SKO:", [numpy.std(i) for i in self.data.T])
        print("After standardization:")

        scaler = preprocessing.StandardScaler().fit(self.data)
        data = scaler.transform(self.data)
        print("Math:", [numpy.mean(i) for i in data.T])
        print("SKO:", [numpy.std(i) for i in data.T])

        print("\nScaler: ")
        print(scaler.mean_)
        print(scaler.var_)

    @staticmethod
    def plot(data):
        n_bins = 20
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].hist(data[:, 0], bins=n_bins)
        axs[0, 0].set_title('age')
        axs[0, 1].hist(data[:, 1], bins=n_bins)
        axs[0, 1].set_title('creatinine_phosphokinase')
        axs[0, 2].hist(data[:, 2], bins=n_bins)
        axs[0, 2].set_title('ejection_fraction')
        axs[1, 0].hist(data[:, 3], bins=n_bins)
        axs[1, 0].set_title('platelets')
        axs[1, 1].hist(data[:, 4], bins=n_bins)
        axs[1, 1].set_title('serum_creatinine')
        axs[1, 2].hist(data[:, 5], bins=n_bins)
        axs[1, 2].set_title('serum_sodium')
        plt.show()

    def plot_MinMax(self):
        minMaxScaler = preprocessing.MinMaxScaler().fit(self.data)
        data = minMaxScaler.transform(self.data)

        PredObr.plot(data)

        print(minMaxScaler.data_max_)
        print(minMaxScaler.data_min_)

    def plot_MaxAbs(self):
        maxAbsScaler = preprocessing.MaxAbsScaler().fit(self.data)
        data = maxAbsScaler.transform(self.data)

        PredObr.plot(data)

        print(maxAbsScaler.max_abs_)

    def plot_Robust(self):
        robus = preprocessing.RobustScaler().fit(self.data)
        data = robus.transform(self.data)

        PredObr.plot(data)

        print(robus.center_)

    def plot_signs(self):
        PredObr.plot(self.data)

    def plot_scaled(self):
        scaler = preprocessing.StandardScaler().fit(self.data)
        data = scaler.transform(self.data)

        PredObr.plot(data)

    def plot_custom(self, range=(-5, 10)):
        minMaxScaler = preprocessing.MinMaxScaler(range).fit(self.data)
        data = minMaxScaler.transform(self.data)

        PredObr.plot(data)

    def plot_transformation(self):
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution="normal", n_quantiles=20, random_state=0)
        power_transformer = preprocessing.PowerTransformer()

        data = quantile_transformer.fit_transform(self.data)
        # data = numpy.random.normal(data)

        PredObr.plot(data)

    def plot_discretiz(self):
        discretizer = preprocessing.KBinsDiscretizer((3, 4, 3, 10, 2, 4), encode='ordinal')
        data = discretizer.fit_transform(self.data)

        PredObr.plot(data)

        print(discretizer.bin_edges_)

def main():
    PredObr().plot_discretiz()


if __name__ == '__main__':
    main()
