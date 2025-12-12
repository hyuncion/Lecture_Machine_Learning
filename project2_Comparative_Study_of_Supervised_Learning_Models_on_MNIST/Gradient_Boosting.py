from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

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
gb_score_list = []
gb_score_list2 = []

# est
model1 = GradientBoostingClassifier(n_estimators=50)
model1.fit(X_train, y_train)
score = model1.score(X_test, y_test)
gb_score_list.append(score)

model2 = GradientBoostingClassifier(n_estimators=100)
model2.fit(X_train, y_train)
score = model2.score(X_test, y_test)
gb_score_list.append(score)

model3 = GradientBoostingClassifier(n_estimators=150)
model3.fit(X_train, y_train)
score = model3.score(X_test, y_test)
gb_score_list.append(score)

model4 = GradientBoostingClassifier(n_estimators=200)
model4.fit(X_train, y_train)
score = model4.score(X_test, y_test)
gb_score_list.append(score)

# learning rate

model5 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01)
model5.fit(X_train, y_train)
score = model5.score(X_test, y_test)
gb_score_list2.append(score)

model6 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
model6.fit(X_train, y_train)
score = model6.score(X_test, y_test)
gb_score_list2.append(score)

model7 = GradientBoostingClassifier(n_estimators=200, learning_rate=1)
model7.fit(X_train, y_train)
score = model7.score(X_test, y_test)
gb_score_list2.append(score)

model8 = GradientBoostingClassifier(n_estimators=200, learning_rate=10)
model8.fit(X_train, y_train)
score = model8.score(X_test, y_test)
gb_score_list2.append(score)

print(gb_score_list)
print(gb_score_list2)

def BestGB():
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)