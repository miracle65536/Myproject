# Finding the association between words.
---
## Overall
---
This program implements a graph algorithm based on word vector models to compute the shortest path between two words. Users can input a starting word and a target word, and the program will calculate the shortest path between them and output the path.

The program consists of the following modules:
* Data Preprocessing(Including tokenization, part-of-speech tagging, lemmatization, and stop word removal.)
* To create and train a word vector model using the Word2Vec library
* To calculate the Euclidean distance between vectors and use it as the weight for graph edges
---
## Third-party library
* pandas
* numpy
* NLTK
* gensim
