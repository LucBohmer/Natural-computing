import pandas as pd
import sys
from sklearn import metrics

def generate_N_grams(string, ngram=3):
    chars = [char for char in string]
    temp = zip(*[chars[i:] for i in range(0,ngram)])
    ans = [''.join(ngram) for ngram in temp]
    return ans

test_file = sys.argv[1]
labels_file = test_file[:-4]+'labels'
ngram = int(sys.argv[3])
test = pd.read_table(test_file, header=None)
labels = pd.read_table(labels_file, header=None)

anomaly_scores = pd.read_table(sys.argv[2], header=None)

test_chunks = [generate_N_grams(''.join(x), ngram ) for x in test[0]]

counter = 0
combined_scores = []
for sequence in test_chunks:
    n_chunks = counter+len(sequence)
    combined_scores.append(anomaly_scores.iloc[counter:n_chunks].mean()[0])
    counter += len(sequence)

table = pd.DataFrame()
table['score'] = combined_scores
table['label'] = labels
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