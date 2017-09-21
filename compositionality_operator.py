#---------------------------------------#
#   Nonlinear Composition Program
#           James Flamino
#---------------------------------------#

from __future__ import print_function

import sys
import os

import gensim, logging
import numpy as np
import time

import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize

# Calculate cos angle
def cos_angle(d1, d2):
    return np.divide(np.dot(d1,d2),np.dot(np.linalg.norm(d1),np.linalg.norm(d2)))

# Calculate p(w|c)
def e_vec(c_i, w_j):
    u_c_i = model.wv[c_i]
    v_w_j = model.wv[w_j]
    return np.exp(np.dot(u_c_i.T, v_w_j))

# Calculate p(w|C)
def p_w_C_calc(w_j, p_w_j, C):
    p_w_C_sum = p_w_j
    for i in range(len(C)):
        p_w_C_sum *= e_vec(C[i], w_j)
    return p_w_C_sum

def generate_linear_data(C, word_freq, word):
    total_vector = model.wv[C[0]]
    for i in C[1:]:
        try:
            new_vector = model.wv[i]
            total_vector = np.add(total_vector, new_vector)
        except:
            continue

    newpath = 'results/' + word 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    d = open('results/' + word + '/linear_' + word + '_comparison.dat', 'w')

    for i in range(len(word_freq)):
        total_vector_arr = np.array([total_vector])
        total_vector_arr = total_vector_arr.T
        angle = cos_angle(np.array([model.wv[word_freq[i][1]]]), total_vector_arr)
        d.write(word_freq[i][1] + '\t' + str(angle[0][0]) + '\n')

    d.close()
    return 0

# Main program
def main():
    # Collect user arguments
    word = ''
    word_type = ''
    if (len(sys.argv) >= 3):
        word = str(sys.argv[1])
        word_type = str(sys.argv[2])
    else:
        print('ERROR: Program requires target word (i.e. house) and word part of speech (i.e. noun)')
        return 0

    # Collect word frequencies from dictionary and initialize all values of c
    print('Collecting word frequencies from dictionary...')
    with open('data/wiki-english_wordids.txt') as f:
        content = f.readlines()

    word_freq = []
    c = []
    for i in content[1:]:
        subcontent = i.split('\t')
        try:
            check_vector = model.wv[subcontent[1]]
            word_freq.append([int(subcontent[0]), subcontent[1], int(subcontent[2])])
            c.append([subcontent[1]])
        except:
            continue

    n = len(word_freq)

    print('Vocabulary length n: ' + str(n))

    # Assemble target phrase
    print('Assembling set C...')
    word_type = word_type[0]
    wordnet_definition = word + '.' + word_type + '.01'
    print(wordnet_definition)
    sentence = wn.synset(wordnet_definition).definition()

    if (';' in sentence):
        sent_list = sentence.split(';')
        sentence = sent_list[0]

    print('Initializing target phrase: ' + sentence)

    stop = stopwords.words('english') + list(string.punctuation)
    subsentence = [i for i in word_tokenize(sentence.lower()) if i not in stop]

    C = []
    for i in subsentence:
        try:
            new_vector = model.wv[i]
            C.append(i)
        except:
            continue

    print('Tokenized target phrase: ', end='')
    print(C)

    m = len(C)

    print('Target phrase length m: ' + str(m))
    
    # Generating linear comparisons for C
    generate_linear_data(C, word_freq, word)

    # Generating nonlinear comparisons for C
    # Calculate p for all w
    print('Calculating p...')
    p_w_row = np.zeros(len(word_freq))

    # Find p_w
    print('Finding p_w...')
    for i in range(len(word_freq)):
        p_w_row[i] = word_freq[i][2]

    p_w_row = np.array([p_w_row])

    p_w = p_w_row.T

    p_w = p_w ** (1 - m)

    print('p_w shape: ', end='')
    print(p_w.shape)

    # Find p
    p = np.zeros(n)
    Z_C = 0
    for i in range(len(word_freq)):
        p_val = p_w_C_calc(word_freq[i][1], p_w[i][0], C)
        p[i] = p_val
        Z_C += p_val

    p = np.array([p])

    p = p / Z_C

    print('p shape: ', end='')
    print(p.shape)

    diag_p = np.diag(p[0])

    print('daig_p shape: ', end='')
    print(diag_p.shape)

    # Calculate tau
    print('Calculating tau...')
    V = []
    for i in range(len(word_freq)):
        V.append(model.wv[word_freq[i][1]])

    V = np.array(V)

    print('V shape: ', end='')
    print(V.shape)

    ones = np.ones((1, n))

    print('ones shape: ', end='')
    print(ones.shape)

    tau = np.dot(diag_p, V)

    tau = np.dot(ones, tau)

    tau = tau.T

    print('tau shape: ', end='')
    print(tau.shape)

    # Calculate p_w
    # Sift through all c to find smallest cos angle
    d = open('results/' + word + '/nonlinear_' + word + '_comparison.dat', 'w')

    results = []
    print('Finding cos angle between p_c and tau for all c...')
    for i in c:
        print('Evaluating word ' + i[0] + '...')
        p_c = 0
        for j in range(len(word_freq)):
            p_c += e_vec(i[0], word_freq[j][1]) * model.wv[word_freq[j][1]]
        p_c = np.array([p_c])

        cos_angle_val = cos_angle(p_c, tau)
        results.append([i[0], cos_angle_val[0][0]])
        d.write(i[0] + '\t' + str(cos_angle_val[0][0]) + '\n')

    d.close()

    return 0

if __name__ == '__main__':
    print("Program Begin.")
    start_time = time.time()
    # Load pre-trained model
    print('Loading pre-trained model...')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.KeyedVectors.load_word2vec_format('model/word2vec_model.bin.gz', binary=True)
    main()
    print("Program End.")
    print("--- %s seconds ---" % (time.time() - start_time))