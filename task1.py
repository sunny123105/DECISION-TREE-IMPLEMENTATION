import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
print(iris)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
X, y = iris.data, iris.target
# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot tree
plt.figure(figsize=(10,6))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
