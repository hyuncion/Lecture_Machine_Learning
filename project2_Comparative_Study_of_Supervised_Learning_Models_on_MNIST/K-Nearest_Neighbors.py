from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
knn_score_list = []

# n_neighbors
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train, y_train)
score = model1.score(X_test, y_test)
knn_score_list.append(score)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
score = model2.score(X_test, y_test)
knn_score_list.append(score)

model3 = KNeighborsClassifier(n_neighbors=7)
model3.fit(X_train, y_train)
score = model3.score(X_test, y_test)
knn_score_list.append(score)

model4 = KNeighborsClassifier(n_neighbors=9)
model4.fit(X_train, y_train)
score = model4.score(X_test, y_test)
knn_score_list.append(score)

# distance
model5 = KNeighborsClassifier(n_neighbors=3, weights='distance')
model5.fit(X_train, y_train)
score = model5.score(X_test, y_test)
print(score)


print(knn_score_list)

def BestKNN():
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)