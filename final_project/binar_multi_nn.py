import numpy as np
import pandas as pd

data = pd.read_csv('age_final_numeric_corr_2.csv')
test_data = data[25000:]


data = np.array(data[:25000])
m, n = data.shape
np.random.shuffle(data)
test_data = np.array(test_data)

test_data = test_data.T
labels_test = test_data[0].astype(int)
test_samples = test_data[1:]
test_samples = test_samples / 255.

train_data = data[:m].T
labels_train = train_data[0].astype(int)
train_samples = train_data[1:n]
train_samples = train_samples / 255.
_,m_train = train_samples.shape


class MultilayerNN():
    def __init__(self, sizes, epochs=200, rate=0.1):
        self.input_size = sizes[0]
        self.layer_1_size = sizes[1]
        self.output_size = sizes[2]
        self.epochs = epochs
        self.rate = rate
        self.weights = self.set_weights()
    
    def set_weights(self):
        w1 = np.random.rand(self.layer_1_size, self.input_size) - 0.5
        b1 = np.random.rand(self.layer_1_size, 1) - 0.5
        w2 = np.random.rand(self.output_size, self.layer_1_size) - 0.5
        b2 = np.random.rand(self.output_size, 1) - 0.5
        return w1, b1, w2, b2


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def softmax_derivative(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def forward(self, w1, b1, w2, b2, X):
        input_weighted = w1.dot(X) + b1
        activated_1 = self.ReLU(input_weighted)
        Z2 = w2.dot(activated_1) + b2
        out_activated = self.sigmoid(Z2)
        return input_weighted, activated_1, Z2, out_activated

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward(self, Z1, A1, out_activated, w2, X, Y):
        one_hot_Y = self.one_hot(Y)
        dZ2 = out_activated - one_hot_Y
        # print('dZ2 shape', dZ2.shape)
        # print('w1 shape', w1.shape)
        # print('w2 shape', w2.shape)
        # print('hidden_weighted shape', Z1.shape)
        delta_w2 = 1 / m * dZ2.dot(A1.T)
        delta_bias2 = 1 / m * np.sum(dZ2)
        delta_input_weighted = w2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        delta_w1 = 1 / m * delta_input_weighted.dot(X.T)
        delta_bias1 = 1 / m * np.sum(delta_input_weighted)
        return delta_w1, delta_bias1, delta_w2, delta_bias2

    def update_weights(self, w1, b1, w2, b2, delta_w1, delta_bias1, delta_w2, delta_bias2, alpha):
        w1 -= alpha * delta_w1
        b1 -= alpha * delta_bias1    
        w2 -= alpha * delta_w2  
        b2 -= alpha * delta_bias2  
        return w1, b1, w2, b2

    def get_predictions(self,output):
        return np.argmax(output, 0)

    def get_accuracy(self,predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def make_predictions(self, X, w1, b1, w2, b2):
        _, _, _, out_activated = self.forward(w1, b1, w2, b2, X)
        predictions = self.get_predictions(out_activated)
        return predictions



    def train(self, X, Y):
        w1, b1, w2, b2 = self.weights
        for i in range(self.epochs):
            Z1, A1, Z2, out_activated = self.forward(w1, b1, w2, b2, X)
            delta_w1, delta_bias1, delta_w2, delta_bias2 = self.backward(Z1, A1, out_activated, w2, X, Y)
            w1, b1, w2, b2 = self.update_weights(w1, b1, w2, b2, delta_w1, delta_bias1, delta_w2, delta_bias2, self.rate)
            if i % 50 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(out_activated)
                print(self.get_accuracy(predictions, Y))
        return w1, b1, w2, b2



epochs_n = 3000
print(f'TEST {1}: sample size - {25000}, epochs - {epochs_n}')
network = MultilayerNN(sizes=[59, 28, 7], epochs=epochs_n, rate=1)
w1, b1, w2, b2 = network.train(X=train_samples, Y=labels_train)


test_predictions = network.make_predictions(test_samples, w1, b1, w2, b2)
print('Test sample accuracy:', network.get_accuracy(test_predictions, labels_test))

# softmax
# 19 features => 0.32 max precision
# 61 features => 0.3968
    
# sigmoid
# 59 features => 0.3967 (rate=1) (layer = 14)
# 59 features => 0.402 (rate=1) (layer = 28)
    