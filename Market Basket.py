#Data Preprocessing
import pandas as pd
data=pd.read_csv("C:\\Users\exam.SBS\Desktop\Market_Basket.csv",header=None)

transactions=[]

for i in range(0,7500):
    transactions.append([str(data.values[i, j]) for j in range(0, 15)])

#Training apriori on the dataset

from apyori import apriori
rules = apriori(transactions,min_support = 0.003,min_confidence = 0.2,min_lift = 2)

results = list(rules)
results[0]


##### to visualize the final results
for item in results:
    #first index of the inner list
    # Contains base item and add item
    pair=item[0]
    items=[x for x in pair]
    print("Rule:"+ items[0] + "->" + items[1])

    #second index of the inner list
    print("Support:" + str (item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence:" + str(item[2][0][2]))
    print("Lift:"+str(item[2][0][3]))
    print("==================================")

######
