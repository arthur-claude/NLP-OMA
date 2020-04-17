# Aspect-Based Sentiment Analysis 

This script is an implementation of a Sentiment Analysis system, written by Arthur Claude, 
Antoine Guiot and Armand Margerin.

The goal of this exercise was to implement a classifier to predict aspect-based polarities of opinions in
sentences, that assigns a polarity label to every triple <aspect categories, aspect_term, sentence>.
The polarity labels are positive, negative and neutral. Note that a sentence may have several
opinions.

The training set contains 1503 lines, i.e. 1503 opinions.
The development set contains 376 lines, i.e. 376 opinions.

Each line contains 5 tab-separated fields: the polarity of the opinion, the aspect category on which
the opinion is expressed (there are 12 different aspects categories), a specific target term, the character offsets of the term (start:end), and the
sentence in which that opinion is expressed.

## Prerequisites

In the script, the following libraries are used:
- nltk for pre-processing
- scikit-learn for pre-processing
- tensorflow (keras) for pre-processing and construction of the model

Pre-trained word vectors from **fastText** are used. If this data is not present in the user's *resources* file, the download is automatic in our code.

## Classifier Class

### Preprocessing: read_data function
This function allows to read the data .csv files and to convert them into the format required for the use of our model. 

First, the polarity labels are processed into integers, and the label vector is converted into a binary class matrix thanks to keras.

Then, some pre-processing operations are applied to the sentences of each row of the dataframe:
- The target term is removed from the sentence
- Stopwords are removed from the sentence, except some particular stopwords that can have an impact on sentiment analysis (not, no, nor).
- The python file **word2vec.py** is used to encode the sentences. Each word in the sentence is encoded with its corresponding vector of size 300 given by the pre-trained vectors of fastText. In order to have the same input size for each sentence, we set the sentence length to 100. In this way, for sentences with less than 100 words, the sequence of words is padded with zeros vectors until 100 words, and for sentences with more than 100 words, the sentence is truncated (meaning we could possibly loss information). If needed, this parameter can be ser as more than 100.

Finally, the aspect categories are encoded thanks to a one-hot encoder for each row.

The function **read_data** returns 3 elements: the set of preprocessed sentences, the set of encoded categories, and the set of labels.  

### Model: train and predict functions
The model used is a Deep Learning model. 
The sentences, size (100,300), go through a Dense layer of size 16, then in a LSTM of size 16. 
In parallel, the encoded categories, size (12,), go through a Dense layer of size 16.
The outputs are combined thanks to a Concatenate Layer, and the final layer is a Dense layer of size 3.

From the final vectors obtained, we can deduce the polarity of the opinion.

The function **train** is used to train the model from the train set, and the function **predict** is used to predict opinions for a new dataset.

## Results obtained

The accuracy obtained on the development set is 78.73% (mean for 5 runs, with a variance of 0.48%).

Please note that the majority class of the dev set is about 70% (positive labels).
