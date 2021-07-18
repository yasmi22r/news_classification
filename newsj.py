#1
# with open('news', 'r') as f:
#     text = f.read()
#     news = text.split("\n\n")
#     count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}
#     for news_item in news:
#         lines = news_item.split("\n")
#         print(lines[6])
#         file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+')
#         count[lines[6]] = count[lines[6]] + 1
#         file_to_write.write(news_item)  # python will convert \n to os.linesep
#         file_to_write.close()
        
        
        
        
# #2
import pandas
import glob

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
directory_list = ["data/sport/*.txt", "data/world/*.txt","data/us/*.txt","data/business/*.txt","data/health/*.txt","data/entertainment/*.txt","data/sci_tech/*.txt",]

text_files = list(map(lambda x: glob.glob(x), directory_list))
text_files = [item for sublist in text_files for item in sublist]

training_data = []


for t in text_files:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data' : t[0] + ' ' + t[1], 'flag' : category_list.index(t[6])})
    
training_data[0]        

# # #3
training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8') # create a train data train_data1.csv is an example
print(training_data.data.shape)

# # #4
import pickle
from sklearn.feature_extraction.text import CountVectorizer


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb")) # count_vector1.pkl is an example


# #5

from sklearn.feature_extraction.text import TfidfTransformer

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf1.pkl","wb")) # tfidf1.pkl is an example 


# From 2 to 5 is a cycle

# #6


# # Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

#SAVE MODEL
pickle.dump(clf, open("nb_model.pkl", "wb"))


# #7



import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = "Waka Flocka responds to Fan calling his Music irrelevant"  # this is an input for testing whether the model is working or not.
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))  # Firat algorithm loaded

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)


## Fix Reshaping problems
# from sklearn import metrics
print("Baysian Prediction:" + "  " + category_list[predicted[0]])
# print("Accuracy:",metrics.accuracy_score(y_test, predicted))
# print("Baysian Accuracy:",metrics.accuracy_score(y_test, predicted))
# #8


predicted = loaded_model.predict(X_test)
result_bayes = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_bayes.to_csv('res_bayes.csv', sep = ',')

# for predicted_item, result in zip(predicted, y_test):
#     print(category_list[predicted_item], ' - ', category_list[result])
    
    
# #9


# from sklearn.metrics import confusion_matrix  

# confusion_mat = confusion_matrix(y_test,predicted)
# print(confusion_mat)


# #10


##################################### Nural  Model End #################################
from sklearn.neural_network import MLPClassifier

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_neural.fit(X_train, y_train)




# #11


pickle.dump(clf_neural, open("softmax.pkl", "wb"))

# #12
from sklearn import metrics

predicted = clf_neural.predict(X_test)
result_softmax = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_softmax.to_csv('res_softmax.csv', sep = ',')

print("Neural Network Softmax Prediction:" + "  " + category_list[predicted[0]])
print("Neural Network Softmax Accuracy:", metrics.accuracy_score(y_test, predicted))
# for predicted_item, result in zip(predicted, y_test):
#     print(category_list[predicted_item], ' - ', category_list[result])

##################################### Nural Model End #################################    
    
    
    ##################################### SVM Model End #################################
# #13


from sklearn import svm
from sklearn import metrics
clf_svm = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_svm.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_svm, open("svm.pkl", "wb"))


#14


predicted = clf_svm.predict(X_test)
result_svm = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_svm.to_csv('res_svm.csv', sep = ',')
print( "SVM Prediction" + "  " + category_list[predicted[0]])
print("SVM Accuracy:",metrics.accuracy_score(y_test, predicted))
# for predicted_item, result in zip(predicted, y_test):
#     print(category_list[predicted_item], ' - ', category_list[result])
##################################### SVM Model End #################################    
  
  
#15  
##################################### Decision Model Start #################################
# We  applied The  Decision Tree Algorithm 
#Total Lines 09
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf_decision= DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_decision.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_decision, open("decisionmodel.pkl", "wb"))



#Train the model using the training sets y_pred=clf.predict(X_test)


predicted_decision = clf_decision.predict(X_test)
result_decision = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted_decision})
result_decision.to_csv('res_decision.csv', sep = ',')
print("Decision Tree Prediction" + "  "  + category_list[predicted_decision[0]])
print("Decision Tree Accuracy:",metrics.accuracy_score(y_test, predicted_decision))
##################################### Decision Model End #################################


# #15
# We will build a model using Random Forest Algorithm

##################################### Random Model Start #################################
#16 steps Total Line 09
#Implementing The  Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#Create a Gaussian Classifier n_estimators=100
clf_random=RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_random.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_random, open("random.pkl", "wb"))



#Train the model using the training sets y_pred=clf.predict(X_test)


random_pred=clf_random.predict(X_test)
result_random = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': random_pred})
result_random.to_csv('res_random.csv', sep = ',')
print("Random Forest Prediction" + "  " + category_list[random_pred[0]])
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, random_pred))
##################################### Random Model End #################################

#Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#clf.predict([[3, 5, 4, 2]])


#  Adaboost Algorithm #####################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.3, random_state=42)

# Train Adaboost classifier
model = abc.fit(X_train_tfidf, training_data.flag)

pickle.dump(abc, open("adaboostmodel.pkl", "wb"))


# train the model using the training sets y_pred=clf.predict(X_test)
predicted_decision = model.predict(X_test)
result_decision = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted_decision})
result_decision.to_csv('res_adaboost.csv', sep = ',')

# for predicted_item, result in zip(predicted, y_test):
    # print(category_list[predicted_item], ' - ', category_list[result])

print( "Ada Boost Prediction:" + "  "  + category_list[predicted_decision[0]])


# Model Accuracy, how often is the classifier correct?
print("Ada Boost Accuracy:",metrics.accuracy_score(y_test, predicted_decision))
