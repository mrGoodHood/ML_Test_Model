import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    "area": [30, 40, 50, 60, 70, 80, 90, 100],
    "rooms": [1, 1, 2, 2, 3, 3, 4, 4],
    "price": [150, 180, 220, 260, 310, 350, 400, 450]
}

df = pd.DataFrame(data)

x = df[["area", "rooms"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)

# print("X_train:", x_train.shape)
# print("X_test:", x_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

model = LinearRegression()

model.fit(x_train, y_train)

# print("Коэффициенты:", model.coef_)
# print("Свободный член:", model.intercept_)

predictions = model.predict(x_test)

# print("Предсказания:", predictions)
# print("Правильные ответы:", y_test.values)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
