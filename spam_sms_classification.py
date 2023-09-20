import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Loading data
df = pd.read_csv('//content//sample_data//spam.csv', encoding='ISO-8859-1')

X = df['v2'] #column
y = df['v1'] #target

#convert labels to binary values (ham: 0, spam:1)
y = y.map({'ham' : 0, 'spam' : 1})

from numpy.lib.function_base import vectorize
#Text Vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

#Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

#Predict and Evaluate

y_pred = svm_model.predict(X_test)

#accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f'Acc : {accuracy}')


#classification report
report = classification_report(y_test, y_pred)
print(f'rep = \n{report}')
