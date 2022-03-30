import pandas as pd
import sys

def generate_N_grams(string, ngram=3):
    chars = [char for char in string]
    temp = zip(*[chars[i:] for i in range(0,ngram)])
    ans = [''.join(ngram) for ngram in temp]
    return ans

training_file = sys.argv[1]
test_file = sys.argv[2]
ngram = int(sys.argv[3])
training_out = training_file+'.out'
test_out = test_file+'.out'
train = pd.read_table(training_file, header=None)
test = pd.read_table(test_file, header=None)
training_chunks = generate_N_grams(''.join(train[0]), ngram)
test_chunks = generate_N_grams(''.join(test[0]), ngram)
outputstring = '\n'.join(training_chunks)
with open(training_out, 'w') as output:
    output.write(outputstring)

outputstring = '\n'.join(test_chunks)
with open(test_out, 'w') as output:
    output.write(outputstring)