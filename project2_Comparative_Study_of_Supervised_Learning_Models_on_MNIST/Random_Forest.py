from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
rf_score_list = []

# est
model1 = RandomForestClassifier(n_estimators=50)
model1.fit(X_train, y_train)
score = model1.score(X_test, y_test)
rf_score_list.append(score)

model2 = RandomForestClassifier(n_estimators=100)
model2.fit(X_train, y_train)
score = model2.score(X_test, y_test)
rf_score_list.append(score)

model3 = RandomForestClassifier(n_estimators=150)
model3.fit(X_train, y_train)
score = model3.score(X_test, y_test)
rf_score_list.append(score)

model4 = RandomForestClassifier(n_estimators=200)
model4.fit(X_train, y_train)
score = model4.score(X_test, y_test)
rf_score_list.append(score)

print(rf_score_list)



def BestRF():
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)