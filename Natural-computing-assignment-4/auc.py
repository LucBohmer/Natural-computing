import pandas as pd
from sklearn import metrics
import sys
table = pd.read_table(sys.stdin, header=None, delim_whitespace=True)

table.columns = ['score', 'label']

table = table.sort_values(by=['score'])
normal_length = len(table.loc[table['label'] == 0])
anomalous_length = len(table.loc[table['label'] == 1])
sensitivity = []
specificity = []

anomalous = table.loc[table['label'] == 1]
normal = table.loc[table['label'] == 0]
for index, value in table.iterrows():
    sensitivity.append(anomalous[anomalous > value['score']].count()[0]/anomalous_length)
    specificity.append(normal[normal < value['score']].count()[0]/normal_length)
print(metrics.auc(sensitivity, specificity))
