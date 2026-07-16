import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_absolute_error


housing = fetch_california_housing()


df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# print("Первые предсказания:")
# print(y_pred[:5])
#
# print()
# print("Настоящие значения:")
# print(y_test.head())

mae = mean_absolute_error(y_test, y_pred)

print()
print("MAE:", mae)
