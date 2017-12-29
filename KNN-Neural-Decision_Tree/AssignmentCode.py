import math
import pandas as pd
import numpy as np

# Data with features and target values
# Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
# Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")
CANDIDATE_ID = "can_id"
CANDIDATE_NAME = "can_nam"
NET_OPERATING_EXPENDITURE = "net_ope_exp"
NET_CONTRIBUTION = "net_con"
TOTAL_LOAN = "tot_loa"
CANDIDATE_OFFICE = "can_off"
CANDIDATE_INCUMBENT = "can_inc_cha_ope_sea"
WINNER = "winner"
columns = [CANDIDATE_ID,
           CANDIDATE_NAME, NET_OPERATING_EXPENDITURE,
           NET_CONTRIBUTION, TOTAL_LOAN, CANDIDATE_OFFICE,
           CANDIDATE_INCUMBENT, WINNER]

DATA_RATIO = 0.8

# ========================================== Data Helper Functions ==========================================


# Normalize values between 0 and 1
# dataset: Pandas dataframe
# categories: list of columns to normalize, e.g. ["column A", "column C"]
# Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData


# Encode categorical values as mutliple columns (One Hot Encoding)
# dataset: Pandas dataframe
# categories: list of columns to encode, e.g. ["column A", "column C"]
# Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)


# Split data between training and testing data
# dataset: Pandas dataframe
# ratio: number [0, 1] that determines percentage of data used for training
# Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
    tr = int(len(dataset) * ratio)
    return dataset[:tr], dataset[tr:]


# Convenience function to extract Numpy data from dataset
# dataset: Pandas dataframe
# Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
    features = dataset.drop(["can_id", "can_nam", "winner"], axis=1).values
    labels = dataset["winner"].astype(int).values
    return features, labels


# Convenience function to extract data from dataset (if you prefer not to use Numpy)
# dataset: Pandas dataframe
# Return: features list and corresponding labels as a list
def getPythonList(dataset):
    f, l = getNumpy(dataset)
    return f.tolist(), l.tolist()


# Calculates accuracy of your models output.
# solutions: model predictions as a list or numpy array
# real: model labels as a list or numpy array
# Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
    predictions = np.array(solutions)
    labels = np.array(real)

    return (predictions == labels).sum() / float(labels.size)


# ===========================================================================================================

class KNN:
    ratio = 0.8
    k = 7

    def __init__(self):
        # KNN state here
        # Feel free to add methods
        # Data prepossessing
        self.train_data = np.array([])
        self.train_labels = np.array([])
        pass

    def euclideanDistance(self, point1, point2):
        return np.sum(np.power(np.array(point1 - point2), 2))

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        self.train_data = features
        self.train_labels = labels
        pass

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        results = []

        for feature in features:
            tuples = []
            for train_data, train_label in zip(self.train_data, self.train_labels):
                distance = self.euclideanDistance(train_data, feature)
                tuples.append((distance, train_label))
                tuples = sorted(tuples, key=lambda x: x[0])
                if len(tuples) > self.k:
                    tuples.pop()

            voting = 0
            result = 0
            for _, isWin in tuples:
                if result == isWin:
                    voting += 1
                else:
                    if voting == 0:
                        result = isWin
                    else:
                        voting -= 1
            results.append(result)

        return results


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def de_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def de_relu(x):
    return 1. * (x > 0)


# Mean Square Error
def error_func(output, standard_output):
    output = np.array(output)
    standard_output = np.array(standard_output)
    return output - standard_output


class NeuralLayer:
    def __init__(self, input_num, output_num):
        # bias for 1
        self.w = np.random.rand(input_num + 1, output_num) - 0.5
        self.input = np.array([])
        self.output = np.array([])
        self.error = np.array([])

    def forward(self, input_list):
        self.input = np.append(np.array(input_list), 1.0)
        self.output = self.activate(self.input.dot(self.w))
        return self.output

    def backPropagation(self, error_vec):
        self.error = error_vec
        next_error = error_vec.dot(self.w[:-1].transpose())
        return self.deactive(self.input[:-1], next_error)

    def descentGradient(self, learning_rate):
        self.w -= learning_rate * np.outer(self.input.transpose(), self.error)

    def activate(self, input_vec):
        return sigmoid(input_vec)

    def deactive(self, output, pre_error_vec):
        np_de_sigmoid = np.vectorize(de_sigmoid)
        return np.multiply(pre_error_vec, np_de_sigmoid(output))


class Perceptron:
    learning_rate = 0.01
    iter_time = 10
    input_num = 9
    output_num = 1

    def __init__(self):
        # Perceptron state here
        # Feel free to add methods
        self.neuralLayer = NeuralLayer(self.input_num, self.output_num)
        return

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        for i in range(self.iter_time):
            for feature, label in zip(features, labels):
                output = self.neuralLayer.forward(feature)
                self.neuralLayer.backPropagation(error_func(output, label))
                self.neuralLayer.descentGradient(self.learning_rate)

    def get_results(self, outputs):
        results = []
        for output in outputs:
            if output >= 0.5:
                results.append(1)
            else:
                results.append(0)

        return results

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        outputs = []
        for feature in features:
            outputs.append(self.neuralLayer.forward(feature))

        return self.get_results(outputs)


class MLP:
    iter_time = 10
    neural_structures = [9, 50, 1]
    learning_rate = 0.01

    def __init__(self):
        self.neuralLayers = []
        for i in range(len(self.neural_structures) - 1):
            self.neuralLayers.append(
                NeuralLayer(self.neural_structures[i], self.neural_structures[i + 1]))

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        for i in range(self.iter_time):
            for feature, label in zip(features, labels):
                output = feature
                for nueralLayer in self.neuralLayers:
                    output = nueralLayer.forward(output)

                error = error_func(output, label)
                for neuralLayer in reversed(self.neuralLayers):
                    error = neuralLayer.backPropagation(error)

                for neuralLayer in self.neuralLayers:
                    neuralLayer.descentGradient(self.learning_rate)

    def get_results(self, outputs):
        results = []
        for output in outputs:
            if output >= 0.5:
                results.append(1)
            else:
                results.append(0)

        return results

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        outputs = []
        for feature in features:
            output = feature
            for neuralLayer in self.neuralLayers:
                output = neuralLayer.forward(output)
            outputs.append(output)
        return self.get_results(outputs)


def info_func(l):
    total = float(sum(l))
    res = 0
    for num in l:
        if num != 0:
            res += -(num / total) * math.log(num / total, 2)
    return res


class TreeNode:
    def __init__(self):
        self.ranges = []
        self.next_nodes = []
        self.predict_val = None
        self.is_terminate = False
        self.index = None

    def setRanges(self, ranges):
        self.ranges = ranges
        self.next_nodes = [None for _ in range(len(ranges) - 1)]

    def getNext(self, value):
        for i in range(len(self.ranges) - 1):
            if i == len(self.ranges) - 2:
                return self.next_nodes[i]
            if self.ranges[i] <= value < self.ranges[i + 1]:
                return self.next_nodes[i]


class ID3:
    bucket_size = 5  # this is ID3 bucket_size

    def __init__(self):
        # Decision tree state here
        # Feel free to add methods
        self.ranges = np.linspace(0, 100, num=self.bucket_size + 1)
        self.root = None
        pass

    def count_feature(self, features, labels, feature_ranges):
        counts = []
        for col in range(features.shape[1]):
            feature_range = feature_ranges[col]
            count = [[0, [0, 0]] for _ in range(len(feature_range) - 1)]
            for i in range(features.shape[0]):
                for j in range(len(feature_range) - 1):
                    if j == len(feature_range) - 2:
                        count[j][0] += 1
                        if labels[i] == 0:
                            count[j][1][0] += 1
                        else:
                            count[j][1][1] += 1
                        break
                    else:
                        if feature_range[j] <= features[:, col][i] < feature_range[j + 1]:
                            count[j][0] += 1
                            if labels[i] == 0:
                                count[j][1][0] += 1
                            else:
                                count[j][1][1] += 1
                            break
            counts.append(count)
            # print col, " Feature count", count
        return counts

    def build_tree(self, features, labels, feature_ranges):
        # Since range of each features from 0 - 1
        # We can split from [0, 1]
        features = np.array(features)
        labels = np.array(labels)
        total_num = len(labels)
        tree_node = TreeNode()
        tree_node.is_terminate = False
        label_status = [(labels == 0).sum(), (labels == 1).sum()]

        # terminate when labels all 0 or 1
        # if also catch features and labels empty
        if label_status[0] == 0 or label_status[1] == 0:
            tree_node.is_terminate = True
            if label_status[1] == 0:
                tree_node.predict_val = 0
            else:
                tree_node.predict_val = 1
            tree_node.labels = label_status
            return tree_node

        feature_statuses = self.count_feature(features, labels, feature_ranges)
        label_info = info_func(label_status)
        attrs_info = []
        for feature_status in feature_statuses:
            attr_info = 0
            for single_range in feature_status:
                attr_info += sum(single_range[1]) / float(total_num) * info_func(single_range[1])
            if attr_info == 0.0:
                attr_info = 1.0
            attrs_info.append(attr_info)

        if np.unique(attrs_info).size == 1:
            tree_node.is_terminate = True
            if label_status[0] >= label_status[1]:
                tree_node.predict_val = 0
            else:
                tree_node.predict_val = 1
            tree_node.labels = label_status

            return tree_node

        split_feature_index = np.argmax(label_info - np.array(attrs_info))
        tree_node.index = split_feature_index
        features_group, labels_group = self.split_for_index(features, labels, feature_ranges, split_feature_index)

        # Set Tree Node ranges
        tree_node.setRanges(feature_ranges[split_feature_index])
        feature_ranges = np.delete(feature_ranges, split_feature_index)
        i = 0

        for next_feature, next_label in zip(features_group, labels_group):
            tree_node.next_nodes[i] = self.build_tree(next_feature, next_label, feature_ranges)
            if not next_feature:
                if label_status[0] >= label_status[1]:
                    tree_node.next_nodes[i].predict_val = 0
                else:
                    tree_node.next_nodes[i].predict_val = 1
            i += 1
        tree_node.labels = label_status

        return tree_node

    def split_for_index(self, features, labels, feature_ranges, index):
        vector = features[:, index]
        features = np.delete(features, index, 1)
        features_group = [[] for _ in range(len(feature_ranges[index]) - 1)]
        labels_group = [[] for _ in range(len(feature_ranges[index]) - 1)]
        for i in range(len(vector)):
            for j in range(len(features_group)):
                if j == len(features_group) - 1:
                    features_group[j].append(features[i])
                    labels_group[j].append(labels[i])
                    break
                else:
                    if feature_ranges[index][j] <= vector[i] < feature_ranges[index][j + 1]:
                        features_group[j].append(features[i])
                        labels_group[j].append(labels[i])
                        break

        # print "Feature Group Len: ", [len(x) for x in features_group]
        return features_group, labels_group

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        # Build tree

        feature_ranges = self.split_range(features)
        self.root = self.build_tree(features, labels, feature_ranges)

    def split_range(self, features):
        feature_ranges = []
        for i in range(features.shape[1]):
            ranges = []
            if len(np.unique(features[:, i])) < self.bucket_size:
                ranges = np.linspace(0.0, 1.0, num=len(np.unique(features[:, i])) + 1)
            else:
                for split_percentage in self.ranges:
                    if len(ranges) == 0 or ranges[-1] != np.percentile(features[:, i], split_percentage):
                        ranges.append(np.percentile(features[:, i], split_percentage))
            feature_ranges.append(ranges)
        # print "Split Range: ", feature_ranges
        return feature_ranges

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        results = []

        for feature in features:
            cur = self.root
            indexs = []
            values = []
            while not cur.is_terminate:
                prev_index = cur.index
                indexs.append((prev_index, cur.labels))
                cur = cur.getNext(feature[cur.index])
                values.append(feature[prev_index])
                feature = np.delete(feature, prev_index)
            results.append(cur.predict_val)
        return results


def prepoccessing(dataset):
    train_data, test_data = trainingTestData(
        normalizeData(
            encodeData(dataset, [CANDIDATE_OFFICE, CANDIDATE_INCUMBENT]),
            [NET_OPERATING_EXPENDITURE, NET_CONTRIBUTION, TOTAL_LOAN]),
        DATA_RATIO)

    train_data, train_labels = getNumpy(train_data)
    test_data, test_labels = getNumpy(test_data)
    return train_data, test_data, train_labels, test_labels


def id3prepoccessing(dataset):
    def label_encoding(dataset, labels):
        for label in labels:
            dataset[label] = dataset[label].astype('category')
            dataset[label] = dataset[label].cat.codes
        return dataset

    id3train_data, id3test_data = trainingTestData(
        normalizeData(
            label_encoding(dataset, [CANDIDATE_OFFICE, CANDIDATE_INCUMBENT]),
            [NET_OPERATING_EXPENDITURE, NET_CONTRIBUTION, TOTAL_LOAN, CANDIDATE_OFFICE, CANDIDATE_INCUMBENT]),
        DATA_RATIO)

    id3train_data, id3train_labels = getNumpy(id3train_data)
    id3test_data, id3test_labels = getNumpy(id3test_data)
    return id3train_data, id3test_data, id3train_labels, id3test_labels




# # KNN
if __name__ == "__main__":
    train_data, test_data, train_labels, test_labels = prepoccessing(dataset)
    id3train_data, id3test_data, id3train_labels, id3test_labels = id3prepoccessing(dataset)
    knn = KNN()
    print "KNN"
    print "KNN for k = ", knn.k
    knn.train(train_data, train_labels)
    print evaluate(knn.predict(test_data), test_labels)

    print "Perceptron"
    perceptron = Perceptron()
    perceptron.train(train_data, train_labels)
    print evaluate(perceptron.predict(test_data), test_labels)

    print "MLP"
    mlp = MLP()
    mlp.train(train_data, train_labels)
    print evaluate(mlp.predict(test_data), test_labels)

    id3 = ID3()
    print "ID3"
    print "ID3 - bucket_size = ", id3.bucket_size
    id3.train(id3train_data, train_labels)
    print evaluate(id3.predict(id3test_data), test_labels)
    # ID3 in here for categorical data change into three values [0, 0.5, 1]
    # So my bucket for categorical data here is [0, 0.3333333] [0.33333, 0.666666] [0.666666, 1]


