import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt


class ChA:
    def __init__(self):
        all_data = pd.read_csv('dataset_group.csv', header=None)
        unique_id = list(set(all_data[1]))
        # print("Count(ids):", len(unique_id))  # Выведем количество id
        items = list(set(all_data[2]))
        # print("Count(items):", len(items))  # Выведем количество товаров
        dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]

        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        self.df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(self.df)

        results = apriori(self.df, min_support=0.38, use_colnames=True, max_len=1)
        new_items = [list(elem)[0] for elem in results['itemsets']]
        new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in new_items] for id in unique_id]
        te = TransactionEncoder()
        te_ary = te.fit(new_dataset).transform(new_dataset)
        self.indf = pd.DataFrame(te_ary, columns=te.columns_)

        new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem not in new_items] for id in unique_id]
        te = TransactionEncoder()
        te_ary = te.fit(new_dataset).transform(new_dataset)
        self.outdf = pd.DataFrame(te_ary, columns=te.columns_)



    def apriori_minSup03(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        results = apriori(data, min_support=0.3, use_colnames=True)
        results['length'] = results['itemsets'].apply(lambda x: len(x))  # добавление размера набора
        print(results)

    def apriori_minSup03_maxLen1(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        results = apriori(data, min_support=0.3, use_colnames=True, max_len=1)
        print(results)

    def apriori_minSup03_len2(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        results = apriori(data, min_support=0.3, use_colnames=True)
        results['length'] = results['itemsets'].apply(lambda x: len(x))
        results = results[results['length'] == 2]
        print(results)
        print('\nCount of result itemstes = ', len(results))

    def graph_itemsets(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        arr = []
        for minSup in range(5, 100):
            results = apriori(data, min_support=minSup/100, use_colnames=True)
            arr.append(len(results))
        plt.plot([i/100 for i in range(5, 100)], arr)
        plt.show()

    def pred(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        for i in range(1, 39):
            for minSup in range(5, 100):
                results = apriori(data, min_support=minSup/100, use_colnames=True)
                results['length'] = results['itemsets'].apply(lambda x: len(x))
                if len(results[results['length'] == i]) == 0:
                    print(i, ":", minSup/100)
                    break

    def apriori_minSup015_lenGt2_have(self, have=set(["yogurt", "waffles"])):
        data = self.indf

        results = apriori(data, min_support=0.15, use_colnames=True)
        results['length'] = results['itemsets'].apply(lambda x: len(x))
        results = results[results['length'] > 1]
        arr = []
        for i in results["itemsets"]:
            if i & have != set():
                arr.append(i)
                print(i)
        print("Количество элементов:", len(arr))

    @staticmethod
    def CountS(data):
        count = 0
        for i in data:
            if i[0] == "s":
                count += 1
        return count

    def BeginOnSgte2(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        arr = []
        results = apriori(data, min_support=0.1, use_colnames=True)

        for i in results['itemsets']:
            if ChA.CountS(i) > 1:
                arr.append(i)
                print(i)
        print(len(arr))

    def aprioriBetwen01and025(self, new=None):
        if new is None:
            data = self.df
        else:
            data = self.indf if new else self.outdf

        results1 = set(apriori(data, min_support=0.1, use_colnames=True)["itemsets"])
        results2 = set(apriori(data, min_support=0.25, use_colnames=True)["itemsets"])
        results = results1.difference(results2)
        print(results)
        print("Кол-во элементов:", len(results))

def main():
    ChA().aprioriBetwen01and025()


if __name__ == '__main__':
    main()
