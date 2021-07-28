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

###### Feature Extraction Techniques
The collection of text documents is converted into a matrix of token counts using Count Vectorize function that produces a sparse representation of the counts.

After the sparse representation of Count Vectorize we have used TFIDF; also known as terms frequency-inverse document frequency, it is the metric that determines how important a word is to a document in our corpus which is a technique used to extract the most meaningful words in the corpus.

###### System Diagram

![Screen Shot 2021-07-28 at 9 32 00 PM](https://user-images.githubusercontent.com/7517102/127351307-6af01599-ca0f-43ee-96c9-58f85bd1ab46.png)


###### Corpus Collection: Training & Testing Data 

Our corpus is a collection of datasets of short text fragments that we used for the evaluation of our topic based text classifier.

This is a dataset of ~32k English news extracted from RSS feeds of popular newspaper websites like nyt.com, usatoday.com, reuters.com. We have used the ‘train_test_split’ to split the data in 80:20 ratio i.e 80% of the data was used for training the model while 20% of the data was used for testing the model that is built out of it.

Categories for our news classification are:
Sport, Business, U.S, Health, Sci&Tech, World and Entertainment


Packages required: 

 * Pandas
 * Sklearn
 * Numpy
 * Matplotlib
 * Pickle
 
Pickle in python is mainly used to serialize and deserialize a Python object structure. It is used to develop our model in this case.

###### Experimental Results

Accuracy:

![Accuracy_graph](https://user-images.githubusercontent.com/7517102/127352176-f67f8421-3fae-4f94-95ce-afd2690a2576.png)

The above barchart demonstrates that Nearest Neighbors, Extra Trees and Gradient Boosting generated similar results which 98%, 87 and 90% accuracy respectively.

F1-Score:

![F1score](https://user-images.githubusercontent.com/7517102/127352323-ce081c4e-f3b5-402d-a59b-27a9d04eb668.png)

Recall:

![recall](https://user-images.githubusercontent.com/7517102/127352408-09da904f-17f4-48bb-8d6b-42fbd40c4fcd.png)

Precision:

![precision](https://user-images.githubusercontent.com/7517102/127352440-9cae68cb-f257-4206-ba86-851799b47827.png)

ROC Curve:

![adaconf](https://user-images.githubusercontent.com/7517102/127352604-ae87b0a0-c9c9-4cf1-b939-6aa021ebd87a.png)
 
ROC Curve for AdaBoost Classifier

[roc_NB.pdf](https://github.com/yasmi22r/news_classification/files/6894531/roc_NB.pdf)

Roc Curve for Naïve Bayes classifier

Roc Curves for the remaining 8 classifiers didn’t render a proper trend except these two.

Analyzing the Accuracy, F1-score, Recall and Precision graphs, we have determined and grouped our classifiers into three classes, i.e low performance, mid performance and high performance. We can determine that classifiers such as Linear_SVM and Poly_SVM are low performance classifiers and that holds true across all metrics i.e Accuracy, F1-score, Recall and Precision. Consequently, Neural Net, AdaBoost and Decision Tree belong to the medium performance tier. Lastly, we have Random Forest, RBF_SVM, Gradient Boosting Classifier, Extra Trees and Nearest Neighbors that have proven to be of high performance and they belong to the high-performance tier with outcome of 80% and above for all four metrics, Accuracy, F1-score, Recall and Precision.

###### Conclusion

In our project News headlines classification we have dealt with text classification which uses out-of-the-box approach than traditional classification with numerical data. We have implemented and achieved our desired outcome regardless of the fact that we look through resources online.  We have seen that classifier such as Nearest Neighbor and Radial Basis Function SVM proves to be one of the high performing algorithms in our news classification problemset with an almost ~99% accuracy.

Limitations:

We can improve the other classifiers by applying Word2Vec or BERT. 

TF-IDF has a few limitations and algorithms such as Word2Vec can be used to improve our model.

TF-IDF is a normalized data format and in comparison with Word2Vec, the latter produces one vector per word. Word2Vec is great for looking into documents and characterizing content and subcategories of content. 


