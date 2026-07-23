x = 3   # Входное значение
y = 6   # Правильный ответ
w = 1   # Начальный вес модели
learning_rate = 0.3    # Размер одного шага градиентного спуска

for step in range(10):
    y_pred = w * x
    loss = (y_pred - y) ** 2
    gradient = 2 * (y_pred - y) * x
    w = w - learning_rate * gradient

    print(step, y_pred, loss, w)