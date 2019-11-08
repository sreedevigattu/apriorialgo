from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import sys, datetime

# Implementation of apriori algorithm - finding associations - using mlxtend module
# (data) csv --> dataframe --> csv (associations)

# python aa_mlxtend.py
# python aa_mlxtend.py store_data.csv 0.0045 0.2 3 2
datafile = "default"
outfile = "rules.csv"
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

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)
frequent_itemsets = apriori(df, min_support=SUPPORT, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=LIFT)
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
for column in ['antecedents','consequents']:
    rules[column] = rules[column].apply(lambda x: [i for i in x])
for column in ['support','leverage']:
    rules[column] = rules[column].map('{:,.4f}'.format) 
for column in ['antecedent support','consequent support','confidence','lift','conviction']:
    rules[column] = rules[column].map('{:,.2f}'.format) 
rules[['support', 'leverage','antecedent support','consequent support','confidence','lift','conviction']] = \
    rules[['support', 'leverage','antecedent support','consequent support','confidence','lift','conviction']].apply(pd.to_numeric)

# filter
rules = rules[ (rules['antecedent_len'] >= LENGTH) & (rules['confidence'] > CONFIDENCE) & (rules['lift'] > LIFT) ]
print(rules)
rules.to_csv("out\\mlxtend\\" + outfile ,index=False)