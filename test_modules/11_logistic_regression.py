from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt


X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)

# for i in range(5):
#     print(
#         "Вероятности:", y_proba[i],
#         "| Предсказание:", y_pred[i],
#         "| Настоящий класс:", y_test[i]
#     )

# print("Коэффициенты:", model.coef_)
# print("Свободный член:", model.intercept_)

x1_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

w1, w2 = model.coef_[0]
b = model.intercept_[0]

x2_boundary = -(w1 * x1_values + b) / w2

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.plot(
    x1_values,
    x2_boundary
)

plt.xlabel("x1")
plt.ylabel("x1")
plt.title("Граница решений Logistic Regression")
plt.show()
