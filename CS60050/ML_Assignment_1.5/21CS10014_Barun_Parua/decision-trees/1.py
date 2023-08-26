import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('../../dataset/decision-tree.csv')

# Splitting the dataset into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

class DecisionTree:
    def __init__(self):
        self.tree = {}

    def entropy(self, column):
        values = column.value_counts()
        total = len(column)
        entropy = 0
        for value in values:
            probability = value / total
            entropy -= probability * np.log2(probability)
        return entropy

    def information_gain(self, data, feature, target):
        total_entropy = self.entropy(data[target])
        values = data[feature].unique()
        weighted_entropy = 0
        for value in values:
            subset = data[data[feature] == value]
            weighted_entropy += (len(subset) / len(data)) * self.entropy(subset[target])
        return total_entropy - weighted_entropy

    def build_tree(self, data, features, target):
        if len(data) == 0:
            return data[target].mode()[0]
        if len(data[target].unique()) == 1:
            return data[target].iloc[0]
        if len(features) == 0:
            return data[target].mode()[0]

        best_feature = max(features, key=lambda feature: self.information_gain(data, feature, target))
        tree = {best_feature: {}}
        features.remove(best_feature)
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            subtree = self.build_tree(subset, features.copy(), target)
            tree[best_feature][value] = subtree
        return tree

    def train(self, train_data, target):
        features = train_data.columns.tolist()
        features.remove(target)
        self.tree = self.build_tree(train_data, features, target)

    def classify_example(self, example, tree):
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        subtree = tree[feature].get(example[feature])
        if subtree is None:
            return tree[feature].mode()[0]
        return self.classify_example(example, subtree)

    def predict(self, test_data):
        predictions = []
        for _, example in test_data.iterrows():
            prediction = self.classify_example(example, self.tree)
            predictions.append(prediction)
        return predictions

# Instantiate the DecisionTree class
tree = DecisionTree()

# Train the tree using training data
tree.train(train_data, target="Outcome")

# Predict on test data
predictions = tree.predict(test_data)

# Calculate accuracy, precision, and recall manually
correct = 0
tp, fp, fn = 0, 0, 0
for i in range(len(predictions)):
    if predictions[i] == test_data.iloc[i]["Outcome"]:
        correct += 1
    if predictions[i] == 1 and test_data.iloc[i]["Outcome"] == 1:
        tp += 1
    elif predictions[i] == 1 and test_data.iloc[i]["Outcome"] == 0:
        fp += 1
    elif predictions[i] == 0 and test_data.iloc[i]["Outcome"] == 1:
        fn += 1

accuracy = correct / len(predictions)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
