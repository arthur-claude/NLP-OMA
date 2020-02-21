# Skip-gram with negative sampling from scratch 

This script is an implementation of Skip-Gram with Negative Sampling, written by Arthur Claude, 
Antoine Guiot and Armand Margerin.

This implementation is inspired by the work presented by Mikolov et al. at NIPS in 2013, 
and based on the optimisation of a log-likelihood function using a stochastic gradient descent. 
The whole method is presented in an article written by Adrien Guille (Université Lyon 2) available at:
http://mediamining.univ-lyon2.fr/people/guille/word_embedding/skip_gram_with_negative_sampling.html?fbclid=IwAR0cO53tj_8Pcs9yXb_QuPOOjvbQ1tk-cc0dJ6cGinMAQa9bwL2ENTkLRW4

## Prerequisites

In the script, the next libraries are used:
- pickle for saving and loading data
- spacy for preprocessing
- decimal for printing issues


## Data Importation

Two functions allow to import the data that will be used in the following.

### Function text2sentences
The function **text2sentences** takes into parameter a path to a text file where each line corresponds to a sentence.
The raw text file is then converted into tokenized sentences.
For each sentence, a list containing all the words of the sentence is created, and these lists are stored in the list **sentences**. 
Finally, stopwords and punctuation are removed thanks to **spacy**, and the function return **sentences**.

### Function loadPairs
The function **loadPairs** takes into parameter a path to a csv file containing at 
least three columns: word1, word2 and similarity. 
It returns an iterator of tuples, where each tuple contains two words and their similarity.

## The SkipGram Class

Here is where most of the work has been done. This class allows to create a Skip-Gram instance.

### Function __init__
The function **__init__** takes into parameters:
- **sentences** a list of lists as created by the function **text2sentences**.
- **nEmbed** the size of the embeddings.
- **negativeRate** the ratio of between the numbers of negatives and positives pairs (see the theory).
- **winSize** the size of the window in which the "context" words are observed for each word. 
- **minCount** the minimum number of times a word must appear to be considered. 

This function allows to initialize all the required elements for the class **SkipGram**.

The principals elements are:
- **vocab**, a dictionary containing the number of occurences of each word in the input **sentences** appearing more than **minCount** times.
- **w2id**, a dictionnary containing containing all the words of **vocab** and their ids.
- **U**, the word representation matrix. Each word is represented by one line of this matrix. That is what we are trying to build.
- **V**, the matrix of contextual representations of words. Each word as a contextual word is represented by a line of this matrix.
- **q**, a list of probabilities. The i-th item of this list is the probability that the i-th word is taken as a negative word. It will be used in the sample function.

### Function sample
The function **sample** allows to sample **negativeRate** negatives words and takes into parameter **omit**, an object containing the words to ommit during the random negative words selection.

Thus, this fonction selects randomly **negativeRate** negatives words into the vocabulary,  on the basis of the probabilities contained into **q**.

### Function train
The function **train** allows to train our model. It takes into parameter **nb_epochs**, the number of training epochs to be realized.

Firstly, the learning rate **eta** is initialized to 0.025. 

Then, for each epoch, 


### Function trainWord

### Function save

### Function similarity

### Function load (static method)



## Acknowledgments

