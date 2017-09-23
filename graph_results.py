import matplotlib
matplotlib.use('TkAgg')

import sys
import glob

import gensim, logging
import numpy as np
import time

import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize

import matplotlib.pyplot as plt

def get_dirs(filepath):
    return [dirs.replace('results/', '') for dirs in glob.glob(filepath)]

print("Program Begin.")

start_time = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

words = get_dirs('results/*')

index = np.arange(len(words))

linear_data = []
nonlinear_data = []

linear_count = []
nonlinear_count = []
for i in words:
    print('Scanning word ' + i)
    nonlinear_content = []
    try:
        with open('results/' + i + '/nonlinear_' + i + '_comparison.dat') as f:
            nonlinear_content = f.readlines()
    except:
        print('Skipping ' + i + '...')
        continue

    nonlinear_results = []
    for j in nonlinear_content:
        subcontent = j.split('\t')
        nonlinear_results.append([subcontent[0], float(subcontent[1])])
        if (subcontent[0] == i):
            nonlinear_data.append(float(subcontent[1]))

    with open('results/' + i + '/linear_' + i + '_comparison.dat') as f:
        linear_content = f.readlines()

    linear_results = []
    for j in linear_content:
        subcontent = j.split('\t')
        subcontent[1] = subcontent[1].replace('[[', '').replace(']]', '')
        linear_results.append([subcontent[0], float(subcontent[1])])
        if (subcontent[0] == i):
            linear_data.append(float(subcontent[1]))

    linear_results.sort(key=lambda x: x[1])
    linear_results.reverse()

    nonlinear_results.sort(key=lambda x: x[1])
    nonlinear_results.reverse()

    for j in range(len(nonlinear_results)):
        if (nonlinear_results[j][0] == i):
            nonlinear_count.append(j)

    for j in range(len(linear_results)):
        if (linear_results[j][0] == i):
            linear_count.append(j)

final_comparison = []
nonlinear_better = 0
linear_better = 0
for i in range(len(nonlinear_data)):
    print(str(nonlinear_data[i]) + ' > ' + str(linear_data[i]) + ' and ' + str(nonlinear_count[i]) + ' < ' + str(linear_count[i]) + ' so for ' + str(words[i]) + ' better is ', end='')
    if (nonlinear_data[i] > linear_data[i] and nonlinear_count[i] < linear_count[i]):
        print('NONLINEAR')
        final_comparison.append(words[i] + ' better with NONLINEAR')
        nonlinear_better += 1
    else:
        print('LINEAR')
        final_comparison.append(words[i] + ' better with LINEAR')
        linear_better += 1

bar_width = 0.30
opacity = 0.8
shift = 0.15
 
fig = plt.figure(111)

print(len(nonlinear_data))
print(len(linear_data))
print(len(nonlinear_count))
print(len(linear_count))

rects1 = plt.bar(index + shift, nonlinear_data, bar_width,
                 alpha=opacity,
                 color='b',
                 label='nonlinear')
 
rects2 = plt.bar(index + bar_width + shift, linear_data, bar_width,
                 alpha=opacity,
                 color='g',
                 label='linear')
 
plt.xlabel('Definition Word')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity between Word and Phrase')
plt.xticks(index + bar_width, words)
plt.legend()

fig = plt.figure(222)
rects1 = plt.bar(index + shift, nonlinear_count, bar_width,
                 alpha=opacity,
                 color='b',
                 label='nonlinear')
 
rects2 = plt.bar(index + bar_width + shift, linear_count, bar_width,
                 alpha=opacity,
                 color='g',
                 label='linear')
 
plt.xlabel('Definition Word')
plt.ylabel('Rank')
plt.title('Rank of Correct Word')
plt.xticks(index + bar_width, words)
plt.legend()
plt.tight_layout()

print('-------------------------------------------')
print('Nonlinear Mean for Correct Word Rank: ', end='')
print(np.mean(nonlinear_count))
print('Nonlinear Median for Correct Word Rank: ', end='')
print(np.median(nonlinear_count))
print('Nonlinear Mean for Correct Word Cosine Similarity: ', end='')
print(np.mean(nonlinear_data))
print('Nonlinear Median for Correct Word Cosine Similarity: ', end='')
print(np.median(nonlinear_data))
print('Linear Mean for Correct Word Rank: ', end='')
print(np.mean(linear_count))
print('Linear Median for Correct Word Rank: ', end='')
print(np.median(linear_count))
print('Linear Mean for Correct Word Cosine Similarity: ', end='')
print(np.mean(linear_data))
print('Linear Median for Correct Word Cosine Similarity: ', end='')
print(np.median(linear_data))
print('-------------------------------------------')
print('Overall nonlinear superiority (better rank and cosine similarity): ' + str(nonlinear_better) + ' out of ' + str(len(nonlinear_count)) + '.')
print('Overall linear superiority (better rank and cosine similarity): ' + str(linear_better) + ' out of ' + str(len(nonlinear_count)) + '.')
print('-------------------------------------------')
print('Program End.')
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()