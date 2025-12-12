from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
sol_score_list = []
mi_score_list = []
c_score_list = []

# Solver
# solver lbfgs
model1 = LogisticRegression(solver='lbfgs')
model1.fit(X_train, y_train)
score = model1.score(X_test, y_test)
sol_score_list.append(score)

#solver netwon-cg
model2 = LogisticRegression(solver='newton-cg')
model2.fit(X_train, y_train)
score = model2.score(X_test, y_test)
sol_score_list.append(score)

#solver sag
model3 = LogisticRegression(solver='sag')
model3.fit(X_train, y_train)
score = model3.score(X_test, y_test)
sol_score_list.append(score)

#solver sag
model4 = LogisticRegression(solver='saga')
model4.fit(X_train, y_train)
score = model4.score(X_test, y_test)
sol_score_list.append(score)

# Max iter

model5 = LogisticRegression(max_iter=100, solver='sag')
model5.fit(X_train, y_train)
score = model5.score(X_test, y_test)
mi_score_list.append(score)

model6 = LogisticRegression(max_iter=400, solver='sag')
model6.fit(X_train, y_train)
score = model6.score(X_test, y_test)
mi_score_list.append(score)

model7 = LogisticRegression(max_iter=700, solver='sag')
model7.fit(X_train, y_train)
score = model7.score(X_test, y_test)
mi_score_list.append(score)

model8 = LogisticRegression(max_iter=1000, solver='sag')
model8.fit(X_train, y_train)
score = model8.score(X_test, y_test)
mi_score_list.append(score)

# C Value

model9 = LogisticRegression(C=0.01, max_iter = 700, solver='sag')
model9.fit(X_train, y_train)
score = model9.score(X_test, y_test)
c_score_list.append(score)

model10 = LogisticRegression(C=0.1, max_iter=700, solver='sag')
model10.fit(X_train, y_train)
score = model10.score(X_test, y_test)
c_score_list.append(score)

model11 = LogisticRegression(C=1, max_iter=700, solver='sag')
model11.fit(X_train, y_train)
score = model11.score(X_test, y_test)
c_score_list.append(score)

model12 = LogisticRegression(C=10, max_iter=700, solver='sag')
model12.fit(X_train, y_train)
score = model12.score(X_test, y_test)
c_score_list.append(score)

# result
print(sol_score_list)
print(mi_score_list)
print(c_score_list)

def BestLogistic():
    model = LogisticRegression(max_iter=700, solver='sag')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)