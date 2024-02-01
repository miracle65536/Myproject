# Finding the association between words.

## Overview

This program implements a graph algorithm based on word vector models to compute the shortest path between two words. Users can input a starting word and a target word, and the program will calculate the shortest path between them and output the path.

The program consists of the following modules:
* import Third-party library
    ```
    from gensim.models import Word2Vec
    from gensim.models import KeyedVectors
    import pandas as pd
    import numpy as np
    from queue import PriorityQueue
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from queue import LifoQueue
    ```
* Data Preprocessing(Including tokenization, part-of-speech tagging, lemmatization, and stop word removal.)
    ```
    # tokenize
    tokens = [word_tokenize(comment) for comment in commentary]
    # tokens = [word for word in word_tokenize(" ".join(commentary))]
    tokens = [[word.replace("'", "") for word in comment] for comment in tokens]
    punc = [',', '-', '.', "'", '[', ']', '', ' ', '(', ')', '!']
    tokens = [[word for word in sentence if word not in punc and not word.isdigit()] for sentence in tokens]

    # POS tagging
    pos_tags = [pos_tag(word) for word in tokens]

    # Named Entity Recognition
    ner_tags = [ne_chunk(word) for word in pos_tags]

    # Lemmatization


    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


    wnl = WordNetLemmatizer()
    lowercase_pos_tags = [[(token.lower(), pos) if pos != 'NNP' else (token, pos) for token, pos in pos_tag] for pos_tag in pos_tags]
    union = []
    for pos_tag in lowercase_pos_tags:
        lemmas_words = []
        for token, pos in pos_tag:
            if pos != 'NNP':
                wn_pos = get_wordnet_pos(pos)
                if wn_pos is not None:
                    if token.lower() == 'pass':
                        lemmas_words.append(token)
                    else: 
                        lemma = wnl.lemmatize(token, pos=wn_pos)
                        lemmas_words.append(lemma)
                else:
                    lemmas_words.append(token)
            else:
                lemmas_words.append(token)
        union.append(lemmas_words)

    # remove stop words
    stop_word = stopwords.words('english')
    stop_word.remove('won')
    commentary_filtered = [[word for word in words if word not in stop_word] for words in union]


    ```
* To create and train a word vector model using the Word2Vec library
  ```
  model = Word2Vec(commentary_filtered, vector_size=100, window=5, min_count=1, workers=4)
  ```
* To calculate the Euclidean distance between vectors and use it as the weight for graph edges
  ```
  def get_dist(self, word1, word2):
        v1 = self.model.wv[word1]
        v2 = self.model.wv[word2]
        distance = np.sqrt(np.sum((v1 - v2) ** 2))
        distance = distance ** 2   
        return distance
  ```
* To calculate the shortest path between nodes using Dijkstra's algorithm(heap optimization) and record the path
  ```
  def dijkstra(self, s, t):
        self.dist[s] = 0
        q = PriorityQueue()
        q.put((0, s))
        while not q.empty():
            dis, n = q.get()
            if dis > self.dist[n]:
                continue
            for i in range(self.count):
                if self.graph[i][n] == -1:
                    continue
                new_dist = self.dist[n] + self.graph[i][n]
                if new_dist < self.dist[i]:
                    self.dist[i] = float(new_dist)
                    if n != i:
                        self.pre[i] = [n]
                    q.put((float(self.dist[i]), i))
                elif new_dist == self.dist[i]:
                    if n != i:
                        self.pre[i].append(n)
  ```
* To find the recorded path using dfs(non-recursive)
  ```
     def record_shortest_path(self, start, target):
        stack = LifoQueue()
        stack.put((start, [start]))

        while not stack.empty():
            node, cur_path = stack.get()

            if node == target:
                self.path.append(cur_path)
            for child in self.pre[node]:
                    stack.put((child, cur_path + [child]))
  ```
## Demo
Enter two words
```
pass
shot
```
Output
```
pass -> replaces -> play -> Dangerous -> excessive -> Lukas -> Nmecha -> Ampadu -> Ethan -> Gazzaniga -> Assisted -> missed -> shot
```
## Third-party library
* pandas
* numpy
* NLTK
* gensim
