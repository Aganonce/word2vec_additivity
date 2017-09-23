import matplotlib
matplotlib.use('TkAgg')

import sys
import os
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
    with open('results/' + i + '/nonlinear_' + i + '_comparison.dat') as f:
        nonlinear_content = f.readlines()

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

newpath = 'plots/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

bar_width = 0.30
opacity = 0.8
shift = 0.15
 
fig = plt.figure(111)

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
 

plt.savefig(word + '_cos_similarity.png')
plt.clf

plt.xlabel('Definition Word')
plt.ylabel('Rank')
plt.title('Rank of Correct Word')
plt.xticks(index + bar_width, words)
plt.legend()

print('Nonlinear Mean: ', end='')
print(np.mean(nonlinear_count))
print('Nonlinear Median: ', end='')
print(np.median(nonlinear_count))
print('Linear Mean: ', end='')
print(np.mean(linear_count))
print('Linear Median: ', end='')
print(np.median(linear_count))

plt.savefig(word + '_ranking.png')

print('Program End.')
print("--- %s seconds ---" % (time.time() - start_time))