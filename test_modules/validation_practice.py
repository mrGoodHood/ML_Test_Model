from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)


param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 7, None],
    "min_samples_leaf": [1, 2, 5, 10]
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)


# # print(scores)
# # print(scores.mean())
# # print(scores.std())
# print(grid.best_params_)
# print(grid.best_score_)

test_score = grid.score(X_test, y_test)

print(grid.best_score_)
print(test_score)
