import sys
import os

import gensim, logging
import numpy as np
import time

import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize

def get_dirs(filepath):
    return [dirs.replace('results/', '') for dirs in glob.glob(filepath)]

def cos_angle(d1, d2):
    return np.divide(np.dot(d1,d2),np.dot(np.linalg.norm(d1),np.linalg.norm(d2)))

def output_comparisons(word, word_type):
    print('Scanning ' + word + '...')
    with open('results/' + word + '/nonlinear_' + word + '_comparison.dat') as f:
        nonlinear_content = f.readlines()

    with open('results/' + word + '/linear_' + word + '_comparison.dat') as f:
        linear_content = f.readlines()

    word_type = word_type[0]
    wordnet_definition = word + '.' + word_type + '.01'
    sentence = wn.synset(wordnet_definition).definition()

    if (';' in sentence):
        sent_list = sentence.split(';')
        sentence = sent_list[0]

    nonlinear_results = []
    for i in nonlinear_content:
        subcontent = i.split('\t')
        nonlinear_results.append([subcontent[0], float(subcontent[1])])

    linear_results = []
    for i in linear_content:
        subcontent = i.split('\t')
        linear_results.append([subcontent[0], float(subcontent[1])])

    nonlinear_results.sort(key=lambda x: x[1])

    nonlinear_results.reverse()

    linear_results.sort(key=lambda x: x[1])

    linear_results.reverse()

    d = open('results/' + word + '/complete_' + word + '_comparison_results.dat', 'w')

    d.write('Definition: "' + sentence + '" Target word: "' + word + '"\n\n')

    d.write('Linear Composition\n')
    for i in range(10):
        d.write(linear_results[i][0] + '\t' + str(linear_results[i][1]) + '\n')

    d.write('\n')

    d.write('Non-linear Composition\n')
    for i in range(10):
        d.write(nonlinear_results[i][0] + '\t' + str(nonlinear_results[i][1]) + '\n')

    d.close()

    return 0

def main():
    word = ''
    word_type = ''
    if (len(sys.argv) >= 3):
        word = str(sys.argv[1])
        word_type = str(sys.argv[2])
        output_comparisons(word, word_type)
        return 0
    else:
        print('ERROR: Program requires target word (i.e. house) and word part of speech (i.e. noun)')
        return 0


if __name__ == '__main__':
    print("Program Begin.")
    start_time = time.time()
    main()
    print("Program End.")
    print("--- %s seconds ---" % (time.time() - start_time))