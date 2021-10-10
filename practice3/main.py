from apriori_python import apriori
from fpgrowth_py import fpgrowth

print("\nTask 1\n")
itemset = [
    ['A', 'B', 'C', 'D'],
    ['A', 'C', 'D', 'F'],
    ['A', 'C', 'D', 'E', 'G'],
    ['A', 'B', 'D', 'F'],
    ['B', 'C', 'G'],
    ['D', 'F', 'G'],
    ['A', 'B', 'G'],
    ['C', 'D', 'F', 'G']]

freqItems, rules = apriori(itemset, minSup=3/8, minConf=0)
print("Apriori:")
print(freqItems)
print(rules)

freqItems, rules = fpgrowth(itemset, minSupRatio=2/8, minConf=0)
print("FPGrowth:")
print(freqItems)
print(rules)

itemset = [
    [2, 3, 6, 7, 12, 14, 15],
    [1, 3, 4, 8, 11, 12, 13, 14, 15],
    [3, 9, 11, 12, 13, 14, 15],
    [1, 5, 6, 7, 14, 15],
    [1, 3, 8, 10, 11, 12, 13, 14, 15],
    [3, 5, 7, 9, 11, 12, 13, 14, 15],
    [4, 6, 8, 10, 11, 12, 14, 13, 15],
    [1, 3, 5, 8, 11, 12, 13, 14, 15]]

print("\nTask 2\n")
freqItems, _ = apriori(itemset, 0, 0)
print(f"A: 2^{len(freqItems[1])}")

basic = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
freqItems, _ = fpgrowth(itemset, 7/8, 0)
print("B:", list(filter(lambda el: el & basic == set(), freqItems)))
