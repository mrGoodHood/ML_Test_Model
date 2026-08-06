from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


X = [
    [20, 30],
    [22, 80],
    [28, 40],
    [30, 90],
    [40, 35],
    [42, 85],
    [48, 45],
    [50, 95]
]

y = [0, 1, 0, 1, 0, 1, 0, 1]

model = DecisionTreeClassifier()
model_2 = DecisionTreeClassifier()
model_3 = DecisionTreeClassifier(random_state=42)

# model.fit(X, y)
# model_2.fit(X, y)
model_3.fit(X,y)


# plt.figure(figsize=(8, 5))
# plot_tree(
#     model,
#     feature_names=["Возраст"],
#     class_names=["Не купил", "Купил"],
#     filled=True
# )
# plt.show()


plt.figure(figsize=(9, 5))

plot_tree(
    model_3,
    feature_names=["Возраст", "Доход"],
    class_names=["Не купил", "Купил"],
    filled=True
)

plt.show()
