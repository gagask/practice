import matplotlib.pyplot as plt
import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, fpmax, association_rules
from collections import Counter
import networkx as nx


class AA:
    def __init__(self):
        all_data = pd.read_csv('groceries - groceries.csv')
        # print(all_data)  # Видно, что датафрейм содержит NaN значения
        np_data = all_data.to_numpy()
        np_data = [[elem for elem in row[1:] if isinstance(elem, str)] for row in
                   np_data]
        unique_items = set()
        self.all_items = []
        for row in np_data:
            for elem in row:
                self.all_items.append(elem)
                unique_items.add(elem)

        te = TransactionEncoder()
        te_ary = te.fit(np_data).transform(np_data)
        self.data = pd.DataFrame(te_ary, columns=te.columns_)
        #print(self.data)
        items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags','sausage','whipped/sour cream',
                 'rolls/buns', 'other vegetables', 'root vegetables','pork', 'bottled water', 'pastry', 'citrus fruit',
                 'canned beer', 'bottled beer']
        np_data = all_data.to_numpy()
        np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in
                    items] for row in np_data]
        te = TransactionEncoder()
        te_ary = te.fit(np_data).transform(np_data)
        self.new_data = pd.DataFrame(te_ary, columns=te.columns_)

        np_data = all_data.to_numpy()
        np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in
                    items] for row in np_data]
        np_data = [row for row in np_data if len(row) > 1]
        te = TransactionEncoder()
        te_ary = te.fit(np_data).transform(np_data)
        self.data_2 = pd.DataFrame(te_ary, columns=te.columns_)


    def FPGrowth003(self, flag=True):
        if flag:
            data = self.data
        else:
            data = self.new_data

        result = fpgrowth(data, min_support=0.03, use_colnames=True)
        print(result)
        result['length'] = result['itemsets'].apply(lambda x: len(x))
        for i in range(1, 30):
            tmp = result[result['length'] == i]
            if len(tmp['support']) == 0:
                continue
            print("Max for len:", i, "is:", max(tmp['support']))
            print("Min for len:", i, "is:", min(tmp['support']))

    def FPMax003(self, flag=True):
        if flag:
            data = self.data
        else:
            data = self.new_data

        result = fpmax(data, min_support=0.03, use_colnames=True)
        print(result)
        result['length'] = result['itemsets'].apply(lambda x: len(x))
        for i in range(1, 30):
            tmp = result[result['length'] == i]
            if len(tmp['support']) == 0:
                continue
            print("Max for len:", i, "is:", max(tmp['support']))
            print("Min for len:", i, "is:", min(tmp['support']))

    def histo(self):
        data = Counter(self.all_items)

        x = []
        y = []
        for k in sorted(data, key=data.get, reverse=True)[:10]:
            x.append(k)
            y.append(data[k])

        plt.bar(x, y, align='center')
        plt.show()

    def graphic(self, flag=True):
        if flag:
            meth = fpgrowth
        else:
            meth = fpmax

        colors = ['r', 'c', 'y', 'g', 'b']
        for i in range(1, 6):
            arr = []
            for minSup in np.linspace(0.005, 1.0, 500):
                results = meth(self.data, min_support=minSup, use_colnames=True, max_len=i)
                results['length'] = results['itemsets'].apply(lambda x: len(x))
                results = results[results['length'] == i]
                arr.append(len(results))
            plt.plot(np.linspace(0.005, 1, 500), arr, colors[i - 1])
        plt.show()

    def aa(self):
        result = fpgrowth(self.data_2, min_support=0.05, use_colnames=True)
        metrics = ["support", "confidence", "lift", "leverage", "conviction"]

        for i in metrics:
            print("For", i, ":")
            rules = association_rules(result, min_threshold=0.01, metric=i)
            print("Mean:", rules[i].mean())
            print("Median:", rules[i].median())
            print("Std:", rules[i].std())

    def graph(self):
        result = fpgrowth(self.data_2, min_support=0.05, use_colnames=True)
        rules = association_rules(result, min_threshold=0.4, metric='confidence')

        G = nx.DiGraph()

        for index, row in rules.iterrows():
            l = list(row['antecedents'])[0]
            r = list(row['consequents'])[0]
            w = row['support']*25
            label = round(row['confidence'], 4)
            G.add_edge(l, r, label=label, weight=w)
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw_networkx(G, pos, with_labels=True)

        nx.draw_networkx_edges(G, pos,
                               width=list([G[n1][n2]['weight'] for n1, n2 in G.edges])
                               )
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=dict([((n1, n2), f'{G[n1][n2]["label"]}') for n1, n2 in G.edges]),
                                     font_color='red'
                                     )
        plt.show()


def main():
    AA().graphic()


if __name__ == '__main__':
    main()
