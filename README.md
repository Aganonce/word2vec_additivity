# Word2Vec Nonlinear Additivity

Requires Python 2.7 or Python 3.5. For Python 2.7 run `setup_py2.sh`. For Python 3.5 run `setup_py3.sh`.

## Nonlinear Compositionality Operator

Running `compositionality_operator.py word word_type` will find the cosine similarity between the word given and the NLTK WordNet definition of that word after evaluating them with the nonlinear composition operator. Argument word_type is the part of speech of the word (i.e. noun, verb).

Running `filtered_compositionality_operator.py word word_type` works like the unfiltered version, but will limit set c to the top 200 words as dictated by cosine similarity between the linear composition of the wordnet definition and all words in the vocabulary.

Both programs will output two files. The nonlinear comparison file will contain set c of words with their cosine similarities to the nonlinear composition of the WordNet phrase for the target word, and the linear comparison file will contain those values for the linear composition.

## Data Analysis

Running `data_comparison.py word word_type` will pull the linear and nonlinear output files for the argument word and output a file that compares the top ten cosine similarities.

Running `graph_results.py` will pull all files from results and save plots for the cosine similaritiy of the correct words and their actual rankings in the vocabulary, as well as output median and mean rankings for all results to the command line.

## C++ Version

To collect word embeddings generated from Google's original C Word2Vec (hierarchical softmax, no negative sampling) for the C++ version of the vanilla nonlinear composition code, navigate to the model folder and call `wget https://www.dropbox.com/s/j3iy88n206cpz6t/small_vectors.bin?dl=0`. The C++ code itself can be compiled as usual in the source folder.

## References

[1] Gittens, Alex, Dimitris Achlioptas, and Michael W. Mahoney. "Skip-Gram− Zipf+ Uniform= Vector Additivity." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017.
