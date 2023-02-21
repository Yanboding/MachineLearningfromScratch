import pandas as pd
import numpy as np
class DecisionNode:
    def __init__(self, left, right, condition, decision_string):
        self.left = left
        self.right = right
        self.condition = condition
        self.decision_string = decision_string

class TerminationNode:

    def __init__(self, value):
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, impurity_fn, min_sample_split=2, max_depth=float("inf")):
        self.impurity_fn = impurity_fn
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        Xy = pd.concat([X,y], axis=1)
        self.root = self._build_tree(Xy)

    def _information_gain(self, parent, children):
        return self.impurity_fn(parent) - sum(child.shape[0]/parent.shape[0] * self.impurity_fn(child) for child in children)

    def _build_tree(self, Xy, depth=0):
        X, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]
        if self.min_sample_split < X.shape[0] and self.max_depth > depth:
            best_split = self._best_split(Xy)
            if best_split["info_gain"] > 0:
                left = self._build_tree(best_split['left'], depth+1)
                right = self._build_tree(best_split['right'], depth+1)
                return DecisionNode(left, right, best_split['condition'], best_split['decision_string'])

        return TerminationNode(y.mode().values[0])

    def _best_split(self, Xy):
        max_info_gain = -1
        best_split = {}
        X, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]
        numerical_features = set(X._get_numeric_data())
        categorical_features = set(X.columns) - numerical_features
        # case 1: split data using categorical features
        for feature in categorical_features:
            for value in Xy[feature].unique():
                info_gain = self._information_gain(y, [y[Xy[feature] == value], y[Xy[feature] != value]])
                if max_info_gain < info_gain:
                    best_split["info_gain"] = info_gain
                    best_split["condition"] = lambda data: data[feature] == value
                    best_split["decision_string"] = feature + " == " + value
                    best_split["left"] = Xy[Xy[feature] == value]
                    best_split["right"] = Xy[Xy[feature] != value]

        # case 2: split data using numerical features
        for feature in numerical_features:
            sorted_values = Xy[feature].sort_values()
            averages = ((sorted_values + sorted_values.shift()).dropna() / 2).unique()
            for value in averages:
                info_gain = self._information_gain(y, [y[Xy[feature] < value], y[Xy[feature] >= value]])
                if max_info_gain < info_gain:
                    best_split["info_gain"] = info_gain
                    best_split["condition"] = lambda data: data[feature] < value
                    best_split["decision_string"] = feature + " < " + str(value)
                    best_split["left"] = Xy[Xy[feature] < value]
                    best_split["right"] = Xy[Xy[feature] >= value]
        return best_split

    def predict(self, x):
        current = self.root
        while isinstance(current,TerminationNode) == False:
            if current.condition(x) == True:
                current = current.left
            else:
                current = current.right
        return current.value


def entropy(X):
    res = 0
    n = len(X)
    for value in np.unique(X):
        p = len(X[X==value]) / n
        res += - p * np.log2(p)
    return res

def gini(X):
    n = len(X)
    return 1 - sum((len(X[X==value]) / n)**2 for value in np.unique(X))

def print_tree(node, depth=0):
    if node != None:
        if isinstance(node, TerminationNode):
            print(' ' * 4 * depth + '-> ' + str(node.value))
        else:

            print_tree(node.left, depth + 1)
            print(' ' * 4 * depth + '-> ' + str(node.decision_string))
            print_tree(node.right, depth + 1)



if __name__ == "__main__":
    titanic = pd.read_csv("../Dataset/train.csv").dropna()

    #titanic_lite = titanic.loc[:,["Embarked", "Age", "Fare","Survived"]]
    titanic_lite = titanic.loc[:,["Embarked", "Survived"]]
    decisionTreeClassifier = DecisionTreeClassifier(impurity_fn=gini)
    decisionTreeClassifier.fit(titanic_lite.iloc[:, :-1],titanic_lite.iloc[:, -1])
    print(print_tree(decisionTreeClassifier.root))


