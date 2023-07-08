import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()  # load in dataset

X = cancer.data # set X to independent variables
y = cancer.target   # set Y to dependent variables

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,train_size=0.8) # split up independent and dependent variables into test and training sets

classes = ['malignant', 'benign']   # values of the dependent variable, whether the cancer is malignant or benign

clf = svm.SVC(kernel="linear", C=2) # create support vector classifier using the linear kernel type
clf.fit(x_train, y_train) # train the classifier using the training data

y_pred = clf.predict(x_test) # stores the model's classification of the dependent variable based on the test data

acc = metrics.accuracy_score(y_test, y_pred) # gives a measure of how accurately the classifier is able to classify the dependent data

print("The classifier correctly classified " + str(acc*100) + "% of data")  # outputs the accuracy to console