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
- xxx for ???
- xxx for ???

## Classifier Class





## Results obtained

The accuracy obtained on the development set is XX%.

Please note that the majority class of the dev set is about 70% (positive labels).
