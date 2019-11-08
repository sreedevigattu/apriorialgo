from apyori import apriori
import pandas as pd
import sys, datetime

# Implementation of apriori algorithm - finding associations - using apyori module
# (data) csv --> list of lists of RelationRecord --> dataframe --> csv (associations)

# python aa_apyori.py
# python aa_apyori.py store_data.csv 0.0045 0.2 3 2

datafile = "default"
outfile = "rules_a.csv"
if len(sys.argv) > 1:
    datafile = sys.argv[1]
    SUPPORT = float(sys.argv[2])
    CONFIDENCE = float(sys.argv[3])
    LIFT = int(sys.argv[4])
    LENGTH = int(sys.argv[5])

if datafile == "default":
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
    SUPPORT = 0.6
    CONFIDENCE = 0.75
    LIFT = 1.2
    LENGTH = 2
else:
    data = pd.read_csv("data\\" + datafile, header=None)
    print(data.size)
    dataset = []
    for i in range(0, 7501):
        dataset.append([str(data.values[i,j]) for j in range(0, 20)])
    outfile = datafile.split('.')[0] + "_" + outfile

association_rules = apriori(dataset, min_support=SUPPORT, min_confidence=CONFIDENCE, min_lift=LIFT, min_length=LENGTH) #, use_colnames=True)
association_results = list(association_rules)
print(type(association_rules), type(association_results[0]))
print(len(association_results))

results = []
for aresult in association_results:
    print(aresult)
    pair = aresult[0] 
    items = [x for x in pair]
    support = aresult[1]
    ordstats = aresult[2]   

    # ordered statistics
    for ordstat in ordstats:
        itembase    = [i for i in ordstat[0]]
        itemadd     = [i for i in ordstat[1]]
        result = [field for field in [  items, itembase, itemadd,
                                        #ordstat[0] ,+ ' --> ' + str(ordstat[1]), # itembase --> itemadd
                                        "{:.4f}%".format(support),                  # support 
                                        "{:.4f}%".format(ordstat[2]),               # confidence
                                        "{:.4f}%".format(ordstat[3])                # lift
                                    ] 
                ]       
        results.append(result)
df = pd.DataFrame(results,columns =['Items','Rule-X', 'Rule-Y', 'Support','Confidence', 'Lift'])
df = df.sort_values(by = ['Items'], ascending = False)
df.to_csv("out\\apyori\\" + outfile, index=False)
print(df.head())