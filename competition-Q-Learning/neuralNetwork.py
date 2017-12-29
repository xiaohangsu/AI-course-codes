import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def de_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def de_relu(x):
    return 1. * (x > 0)


class NeuralLayer:
    def __init__(self, input_num, output_num, hidder_node_num):
        self.w1 = np.random.rand(input_num + 1, hidder_node_num) - 0.5
        self.w2 = np.random.rand(hidder_node_num + 1, output_num) - 0.5
        self.input1  = np.array([])
        self.input2  = np.array([])
        self.output1 = np.array([])
        self.output2 = np.array([])
        self.error1  = np.array([])
        self.error2  = np.array([])

    def forward(self, input_list):
        self.input1 = np.append(np.array(input_list), 1.0)
        self.output1 = self.activate(self.input1.dot(self.w1))
        self.input2 = np.append(self.output1, 1.0)
        self.output2 = self.activate(self.input2.dot(self.w2))
        return self.output2.tolist()

    def backPropagation(self, error_vec):
        self.error2 = error_vec
        self.error1 = error_vec.dot(self.w2[:-1].transpose())
        self.error1 = self.deactive(self.output1, self.error1)

    def descentGradient(self, learning_rate):
        self.w1 -= learning_rate * np.outer(self.input1.transpose(), self.error1)
        self.w2 -= learning_rate * np.outer(self.input2.transpose(), self.error2)

    def activate(self, input_vec):
        return sigmoid(input_vec)

    def deactive(self, output, pre_error_vec):
        np_de_sigmoid = np.vectorize(de_sigmoid)
        return np.multiply(pre_error_vec, np_de_sigmoid(output))

    def readWFromFile(self, filename):
        w_pair = np.load(filename)
        self.w1, self.w2 = w_pair[0], w_pair[1]

    def writeWToFile(self, filename):
        np.save(filename, [self.w1, self.w2])

