# Classification of News HeadLines

###### Background

We have conducted News Headline Classification using 10 machine learning algorithms, analyzed the outcome and compared our results using simple bar chart plots. News Headline Classification falls under Text classification which is the problem of assigning categories to text data according to its content. There are different techniques to extract information from raw text data and use it to train a classification model. In our project we explore a few such techniques. If we improve the performance of this, the resultant output will also change. 


Models implemented:

 * Linear SVM
 * Polynomial SVM
 * Radial Basis Function SVM
 * Multi-Layer Perceptron Classifier
 * AdaBoost
 * Decision Trees
 * Random Forest
 * Gradient Boosting
 * Extra Trees
 * Nearest Neighbors


Metrics used to evaluate the performance of models:

 * Accuracy
 * Precision
 * Recall
 * F1 Score
 * Confusion Matrix
 * ROC curve 
 
###### Objective



###### Feature Extraction Techniques:
The collection of text documents is converted to a matrix of token counts using count vectorize that produces a sparse representation of the counts.

TFIDF,term frequency–inverse document frequency, is the statistic that is intended to reflect how important a word is to a document in our corpus. This is used to extract the most meaningful words in the Corpus. 

###### Link to Dataset: [News Article Dataset](http://acube.di.unipi.it/tmn-dataset/) 
TagMyNews Datasets is a collection of datasets of short text fragments that we used for the evaluation of  our topic-based text classifier. This is a dataset of  ~32K english news extracted from RSS feeds of popular newspaper websites (nyt.com, usatoday.com, reuters.com). Categories are: Sport, Business, U.S., Health, Sci&Tech, World and Entertainment.



Packages required: 

 * Pandas
 * sklearn
 * Numpy
 
 
 
![Multinomial Naive Bayes](https://i.imgur.com/2gaK9iO.png)
![Softmax](https://i.imgur.com/R2XHiuB.png)
![SVM](https://i.imgur.com/dvfwxY8.png)
![Average of three](https://i.imgur.com/1WtrPRv.png)






