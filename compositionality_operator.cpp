#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <ctime>

#include <Eigen/Dense>

using Eigen::MatrixXd;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries
const int m = 4;                   // target phrase length

int main(int argc, char **argv) {
  int start_s = clock();

  // --------------------- Google Word2Vec Model Loading (BEGIN) --------------------- //
  std::cout << "Reading in model..." << std::endl;
  
  FILE *f;
  char st1[max_size];
  char *bestw[N];
  char st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, i, j, cn, bi[100]; // size is size of the vector embedding (# of nodes in hidden layer of Skip-Gram)
  float *M;
  char *vocab;

  f = fopen("model/small_vectors.bin", "rb"); // pull word embeddings

  if (f == NULL) {
    std::cout << "Input file not found\n" << std::endl;
    return -1;
  }

  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);

  vocab = (char *)malloc((long long)words * max_w * sizeof(char)); // allocate words
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float)); // allocate embeddings

  if (M == NULL) { // catch for vector sets
    std::cout << "ERROR: Cannot allocate memory for vector values." << std::endl;
    return -1;
  }

  for (b = 0; b < words; b++) { // popuate arrays
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  // --------------------- Google Word2Vec Model Loading (END) --------------------- //

  // --------------------- Assemble word frequencies from dictionary (BEGIN) --------------------- //
  std::cout << "Reading in dictionary..." << std::endl;
  std::ifstream file("data/text_complete_dictionary.txt");
  std::string str;

  std::vector< std::vector<std::string> > word_freq;

  while (std::getline(file, str)) { // get all word frequencies and push them into 2D vector < word, word frequency >
    std::vector<std::string> tokens;
    std::string buf;
    std::stringstream ss(str);
    while (ss >> buf) tokens.push_back(buf);
    word_freq.push_back(tokens);
  }

  std::cout << "Assembling p_w..." << std::endl;
  MatrixXd p_w(words, 1); // initialize empty matrix representing word frequencies
  for (i = 0; i < word_freq.size(); i++) {
    p_w(i, 0) = pow(std::stof(word_freq[i][1]), (1 - m)); // assemble p_w matrix for word_frequencies
  }

  // --------------------- Assemble word frequencies from dictionary (END) --------------------- //

  // --------------------- Assemble linear vectorization of target phrase C (BEGIN) --------------------- //
  std::cout << "Vectorizing target phrase C..." << std::endl;

  MatrixXd A(1, size); // initialize empty base matrix for vector embeddings
  MatrixXd B(1, size); // initialize empty matrix to add into A
  MatrixXd D(1, size); // initialize tau2 matrix

  MatrixXd E(m, size); // initialize empty matrix representing phrase vector embeddings
  MatrixXd V(words, size); // initialize empty matrix representing all vector embeddings
  
  for (a = 0; a < size; a++) {
    A(0, a) = 0;
  }
  
  char C[m][15] = {"male", "sovereign", "ruler", "kingdom"}; // target phrase
  
  float sum_A = 0;
  for (i = 0; i < m; i++) { // iterate through words in phrase and find their respective word embeddings
    
    for (b = 0; b < words; b++) {
      if (!strcmp(&vocab[b * max_w], C[i])) break; // find word from phrase in loaded vocab
    }

    for (a = 0; a < size; a++) {
      sum_A += (M[a + b * size] * M[a + b * size]);
      B(0, a) = M[a + b * size]; // assemble values for vector embedding of word in Eigen matrix
      E(i, a) = M[a + b * size];
    }

    sum_A = sqrt(sum_A);

    A += B; // add all vector embeddings for total linear composition of phrase
  }

  float lin[words];
  for (b = 0; b < words; b++) { // iterate through words in vocabulary and find all cos similarities to linear composition
    float sum_B = 0;
    
    for (a = 0; a < size; a++) {
      sum_B += (M[a + b * size] * M[a + b * size]);
      B(0, a) = M[a + b * size]; // assemble values for vector embedding of word in Eigen matrix
      V(b, a) = M[a + b * size];
    }
    sum_B = sqrt(sum_B);

    float cos_sim = ((A * B.transpose()) / (sum_A * sum_B))(0);
    lin[b] = cos_sim;
  }
  
  // --------------------- Assemble linear vectorization of target phrase C (END) --------------------- //

  // --------------------- Calculate tau (BEGIN) --------------------- //
  std::cout << "Calculating p..." << std::endl;
  
  A.setZero(1, words); // repurpose A to hold p values
  B.setOnes(1, words); // repurpose B to hold 1s

  float Z_c = 0;
  for (i = 0; i < words; i++) {
    float p_w_C_sum = p_w(i, 0);
    for (j = 0; j < m; j++) {
      p_w_C_sum *= exp((E.row(j) * V.row(i).transpose())(0));
    }
    A(0, i) = p_w_C_sum;
    Z_c += p_w_C_sum;
  }
  
  A = A / Z_c;

  E.setZero(words, size);
  std::cout << "Calculating tau..." << std::endl;
  std::cout << "diag_p dot V equal to tau_step1..." << std::endl;
  for (i = 0; i < words; i++) {
    for (j = 0; j < size; j++) {
      E(i, j) = V(i, j) * A(0, i);
    }
  }

  std::cout << "one_transpose dot tau_step1 equal to tau_step2..." << std::endl;
  D = B * E;

  std::cout << "transpose of tau_step2 equal to tau..." << std::endl;
  A.setZero(size, 1); // repurpose A to tau
  A = D.transpose();

  // --------------------- Calculate tau (END) --------------------- //

  // --------------------- Calculate cos_sim for all c (BEGIN) --------------------- //
  std::cout << "Calculate cos_sim for all c..." << std::endl;

  // std::ofstream d;
  // d.open("nonlinear_composition_king.txt");
  
  B.setZero(1, size);
  for (a = 0; a < words; a++) {
    B *= 0;
    for (b = 0; b < words; b++) {
      B.row(0) += exp((V.row(a) * V.row(b).transpose())(0)) * V.row(b);
    }

    float next_sum_A = 0;
    float next_sum_B = 0;
    for (b = 0; b < size; b++) {
      next_sum_B += (B(0, b) * B(0, b));
      next_sum_A += (A(b, 0) * A(b, 0));
    }

    next_sum_B = sqrt(next_sum_B);
    next_sum_A = sqrt(next_sum_A);

    // d << word_freq[a][0] << "\t" << ((B * A) / (next_sum_B * next_sum_A))(0) << std::endl;

    std::cout << word_freq[a][0] << "\t" << lin[a] << std::endl; // linear composition
    std::cout << word_freq[a][0] << "\t" << ((B * A) / (next_sum_B * next_sum_A))(0) << std::endl; // nonlinear composition
    
  }

  // d.close();
  // --------------------- Calculate cos_sim for all c (END) --------------------- //

  int stop_s = clock();
  std::cout << "Time (sec): " << (stop_s-start_s) / double(CLOCKS_PER_SEC) << std::endl;
}