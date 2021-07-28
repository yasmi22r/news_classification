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

We evaluated each classifier’s ability to select the appropriate category for given the news headline and their brief description. The confusion matrix is created to explore the results and calculate the metrics. We also relied on True Positive, True Negative, False Positive and False Negative values from the Confusion Matrix to calculate True Positive Rate and False Positive rate that would help us plot our ROC curves. We also plotted curves for Accuracy, Precision, Recall and F1 score for all 10 classifiers.

###### Feature Extraction Techniques:
The collection of text documents is converted into a matrix of token counts using Count Vectorize function that produces a sparse representation of the counts.

After the sparse representation of Count Vectorize we have used TFIDF; also known as terms frequency-inverse document frequency, it is the metric that determines how important a word is to a document in our corpus which is a technique used to extract the most meaningful words in the corpus.

###### System Diagram

![Screen Shot 2021-07-28 at 9 32 00 PM](https://user-images.githubusercontent.com/7517102/127351307-6af01599-ca0f-43ee-96c9-58f85bd1ab46.png)


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






