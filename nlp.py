import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv("Restaurant_Reviews.tsv" , delimiter = '\t' , quoting = 3)
#delimeter bcz pd accepts csv
#quoting to remove double quotes

#cleaning the text

import re #lib to clean
nltk.download('stopwords')#download package
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]' , ' ' ,dataset['Review'][i] )
    #what all we dont want to delete + seperated by space + the dataset
    review = review.lower()
    #change to lower case
    review = review.split()
    #split sentence in single seperate words
    ps = PorterStemmer()
    #object of class
    
    #stemming will convert words to original form. loved - love
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)#join all words seperated by space
    corpus.append(review)
    
    
#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

