from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(
    n_samples=500,
    noise=0.3,
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)


deep_tree = DecisionTreeClassifier(
    random_state=42
)

deep_tree.fit(X_train, y_train)

train_accuracy = deep_tree.score(X_train, y_train)
test_accuracy = deep_tree.score(X_test, y_test)


# controlled_tree = DecisionTreeClassifier(
#     max_depth=6,
#     min_samples_split= 7,
#     max_leaf_nodes= 12,
#     random_state=42
# )
#
# controlled_tree.fit(X_train, y_train)


leaf_tree = DecisionTreeClassifier(
    min_samples_leaf=10,
    random_state=42
)

leaf_tree.fit(X_train, y_train)

print("Глубина:", leaf_tree.get_depth())
print("Листьев:", leaf_tree.get_n_leaves())
print("Train:", leaf_tree.score(X_train, y_train))
print("Test:", leaf_tree.score(X_test, y_test))
