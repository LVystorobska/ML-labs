import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


df_main = pd.read_csv('./full_adult_clean.csv')
labels = pd.read_csv('./adult_labels.csv')

scaler = StandardScaler()
data = scaler.fit_transform(df_main)
X_train, X_valid, y_train, y_valid = train_test_split(data, labels, 
                                             test_size=0.3, random_state=2021)
y_train = y_train.to_numpy().astype(int)

class MultilayerNN():
    def __init__(self, sizes, data_shape, epochs=600, rate=0.1):
        self.input_size = sizes[0]
        self.layer_1_size = sizes[1]
        self.output_size = sizes[2]
        self.m, self.n = data_shape
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

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def tanh(self, x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return t

    def tanh_derivative(self, x):
        dt=1-self.tanh(x)**2
        return dt

    def forward(self, w1, b1, w2, b2, X):
        input_weighted = w1.dot(X.T) + b1
        activated_1 = self.ReLU(input_weighted)
        Z2 = w2.dot(activated_1) + b2
        out_activated = self.sigmoid(Z2)
        return input_weighted, activated_1, Z2, out_activated


    def backward(self, Z1, A1, out_activated, w2, X, Y):
        dZ2 = out_activated - Y
        delta_w2 = 1 / self.m * dZ2.dot(A1.T)
        delta_bias2 = 1 / self.m * np.sum(dZ2.T)
        delta_input_weighted = w2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        delta_w1 = 1 / self.m * delta_input_weighted.dot(X)
        delta_bias1 = (1 / self.m) * np.sum(delta_input_weighted)
        return delta_w1, delta_bias1, delta_w2, delta_bias2

    def update_weights(self, w1, b1, w2, b2, delta_w1, delta_bias1, delta_w2, delta_bias2, alpha):
        w1 -= alpha * delta_w1
        b1 -= alpha * delta_bias1  
        w2 -= alpha * delta_w2  
        b2 -= alpha * delta_bias2  
        return w1, b1, w2, b2

    def make_predictions(self, X, w1, b1, w2, b2):
        _, _, _, out_activated = self.forward(w1, b1, w2, b2, X)
        predictions = np.where(out_activated > 0.5, 1, 0)
        print('Out activated:', predictions)
        return predictions



    def train(self, X, Y):
        w1, b1, w2, b2 = self.weights
        for i in range(self.epochs):
            Z1, A1, Z2, out_activated = self.forward(w1, b1, w2, b2, X)
            delta_w1, delta_bias1, delta_w2, delta_bias2 = self.backward(Z1, A1, out_activated, w2, X, Y)
            w1, b1, w2, b2 = self.update_weights(w1, b1, w2, b2, delta_w1, delta_bias1, delta_w2, delta_bias2, self.rate)
            if i % 50 == 0:
                print("Iteration: ", i)
                predictions = np.where(out_activated > 0.5, 1, 0)
                print('Train sample accuracy:', f1_score(Y.T, predictions[0]))
        return w1, b1, w2, b2




network = MultilayerNN(sizes=[29, 100, 1], data_shape = df_main.shape, epochs=600, rate=0.1)
w1, b1, w2, b2 = network.train(X=X_train, Y=y_train.T)


test_predictions = network.make_predictions(X_valid, w1, b1, w2, b2)
print('Test sample accuracy:', f1_score(y_valid, test_predictions[0]))

# 0.5 => 65.8 % - 400 epochs
# 0.5 => 67 % - 600 epochs