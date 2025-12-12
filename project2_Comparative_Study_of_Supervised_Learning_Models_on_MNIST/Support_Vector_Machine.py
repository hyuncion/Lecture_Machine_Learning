from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

# Preprocess the data
X = mnist['data']
y = mnist['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test
svm_score_list = []
svm_score_list2 = []

# C
model1 = SVC(C=0.01)
model1.fit(X_train, y_train)
score = model1.score(X_test, y_test)
svm_score_list.append(score)

model2 = SVC(C=0.1)
model2.fit(X_train, y_train)
score = model2.score(X_test, y_test)
svm_score_list.append(score)

model3 = SVC(C=1)
model3.fit(X_train, y_train)
score = model3.score(X_test, y_test)
svm_score_list.append(score)

model4 = SVC(C=10)
model4.fit(X_train, y_train)
score = model4.score(X_test, y_test)
svm_score_list.append(score)

# kernel
model5 = SVC(C=10, kernel='sigmoid')
model5.fit(X_train, y_train)
score = model5.score(X_test, y_test)
svm_score_list2.append(score)

model6 = SVC(C=10, kernel='rbf')
model6.fit(X_train, y_train)
score = model6.score(X_test, y_test)
svm_score_list2.append(score)

model7 = SVC(C=10, kernel='poly')
model7.fit(X_train, y_train)
score = model7.score(X_test, y_test)
svm_score_list2.append(score)

model8 = SVC(C=10, kernel='linear')
model8.fit(X_train, y_train)
score = model8.score(X_test, y_test)
svm_score_list2.append(score)

print(svm_score_list)
print(svm_score_list2)


def BestSVM():
    model7 = SVC(C=10, kernel='poly')
    model7.fit(X_train, y_train)
    score = model7.score(X_test, y_test)
    print(score)

