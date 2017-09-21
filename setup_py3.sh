#!/bin/sh

echo "Creating virtualenv..."

virtualenv -p python3 wordvec_env

source wordvec_env/bin/activate

pip install --upgrade pip
pip install numpy
pip install scipy
pip install gensim
pip install nltk
pip install matplotlib

python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet

deactivate

echo "Downloading word2vec model..."
mkdir model
cd model
wget https://www.dropbox.com/s/q300yasg7kdxo6a/word2vec_model.bin.gz

echo "Done."
