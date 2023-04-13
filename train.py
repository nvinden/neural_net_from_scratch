import numpy as np
import pandas as pd
import pickle
import datetime
import os

EPOCHS = 30
LEARNING_RATE = 0.00003
LAYERS = [784, 30, 10]

LOAD_WEIGHTS = False

SAVE_NAME = '3_layers'

# A function that loads mnist data into a pandas dataframe
def load_mnist_data():
    # Load the data
    train = pd.read_csv('data/mnist_train.csv')
    test = pd.read_csv('data/mnist_test.csv')

    # Shuffle the data
    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    # Split the data into X and y for train, test and validation
    X_train = train.drop('label', axis=1).to_numpy()
    y_train = train['label'].to_numpy()

    X_test = test.drop('label', axis=1)
    y_test = test['label']

    X_val = X_test[:5000].to_numpy()
    y_val = y_test[:5000].to_numpy()

    X_test = X_test[5000:].to_numpy()
    y_test = y_test[5000:].to_numpy()

    # Return the data
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

        self.weights, self.biases = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        biases = []
        
        # Randomly initialize the weights
        # We use a normal distribution with mean 0 and standard deviation 1
        for i in range(len(self.layers) - 1):
            input_size = self.layers[i]
            output_size = self.layers[i + 1]

            # Adding weighs and biases for the current layer
            biases.append(np.random.normal(loc = 0, scale = 1.0, size = (1, output_size)))
            weights.append(np.random.normal(loc = 0, scale = 1.0, size = (input_size, output_size)))

        return weights, biases

    # Takes the input X and returns the output of the network
    # Forward pass uses ReLU on all layers except the last one, which uses softmax
    # Input: (batch_size, 784)
    def forward(self, X : np.ndarray, return_layer_activations : bool = False) -> np.ndarray:
        # Reshaping if there is no batch dimension
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 784:
            raise ValueError('The input must be of shape (batch_size, 784)')
        if len(X.shape) != 2:
            raise ValueError('The input must be of shape (batch_size, 784) or (784)')
        
        # Returning layer activations if requested. Used in backpropagation typically
        if return_layer_activations: 
            layer_activations = []
        
        # Forward pass
        for i in range(len(self.layers) - 1):
            if return_layer_activations: layer_activations.append(X)

            X = np.dot(X, self.weights[i]) + self.biases[i]

            # TODO: Implement dropout during training

            # ReLU for all layers except the last one
            if i != len(self.layers) - 2:
                # Normalize the data
                mean = np.array(np.mean(X, axis = 1)).reshape(-1, 1)
                std = np.array(np.std(X, axis = 1)).reshape(-1, 1)
                normalized_X = np.divide(np.subtract(X, mean), std)

                # ReLU
                X = np.maximum(normalized_X, 0)
            else: # Softmax for the last layer
                X = self.__softmax(X)

        # Return the layer activations if requested
        if return_layer_activations:
            return X, layer_activations
        
        return X

    # Takes the input X and the target y, and returns
    # the loss and the gradient of the loss with respect to the weights
    def backward(self, X : np.ndarray, y : np.ndarray) -> tuple[np.ndarray, float]:
        pred, layer_activations = self.forward(X, return_layer_activations = True)
        
        # Calculate the loss
        loss = self.__cross_entropy_loss(pred, y)

        weight_gradients = []
        bias_gradients = []

        for layer_num in range(len(self.weights) - 1, -1, -1):
            # dL/dW = dL/dA * dA/dZ * dZ/dW

            prev_activation = layer_activations[layer_num]

            # Final Layer
            if layer_num == len(self.weights) - 1:
                # Change in loss as a function of the weights and biases
                ground_truth = self.__labels_to_one_hot(y)
                activation_j = np.expand_dims(np.transpose(np.squeeze(prev_activation, axis = 0)), axis = 1)
                delta_k = pred - ground_truth

                dL_dW = np.matmul(activation_j, delta_k)
                dL_db = delta_k
            else: # Hidden Layers
                # Change in loss as a function of the weights and biases
                a_i = prev_activation
                relu_prime = self.__calculate_relu_derivative(layer_activations[layer_num + 1])
                # dL/dy = sum(dL(j)/dz(j) * W(j+1)) 
                W_j_plus_1 = self.weights[layer_num + 1]
                dL_j_dz_j = np.matmul(weight_gradients[-1].transpose(), relu_prime.transpose())
                dL_dy = np.matmul(W_j_plus_1, dL_j_dz_j)

                dL_dW = np.matmul(np.transpose(a_i), relu_prime) * np.sum(dL_dy, axis = 1)
                dL_db = relu_prime * np.sum(dL_dy, axis = 1)

            weight_gradients.append(dL_dW)
            bias_gradients.append(dL_db)
        
        # Reverse the gradients to match the order of the weights and biases
        weight_gradients = weight_gradients[::-1]
        bias_gradients = bias_gradients[::-1]

        # Preform gradient clipping to prevent exploding gradients
        weight_gradients = [np.clip(weight_gradient, -1, 1) for weight_gradient in weight_gradients]
        bias_gradients = [np.clip(bias_gradient, -1, 1) for bias_gradient in bias_gradients]

        return loss, (weight_gradients, bias_gradients)

    ###########
    # Helpers #
    ###########

    def __softmax(self, X : np.ndarray) -> np.ndarray:
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
    
    # Cross entropy loss
    def __cross_entropy_loss(self, pred : np.ndarray, y : np.ndarray) -> float:
        if len(y.shape) == 0:
            true_class_prob = np.zeros(shape = (1, 10))
            true_class_prob[0, y] = 1
        else:
            true_class_prob = np.zeros(shape = (y.shape[0], 10))
            true_class_prob[np.arange(y.size), y] = 1

        return -np.sum(np.log(pred) * true_class_prob) / true_class_prob.shape[0]
    
    # Derivative of the softmax function
    def __calculate_softmax_derivatve(self, X : np.ndarray) -> np.ndarray:
        softmax_out = self.__softmax(X)
        return softmax_out / (1.0 - softmax_out)

    # Derivative of the ReLU function
    def __calculate_relu_derivative(self, X : np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, 0)

    # Derivative of the cross entropy function
    def __calculate_cross_entropy_derivative(self, pred : np.ndarray, y : np.ndarray) -> np.ndarray:
        true_class_prob = self.__labels_to_one_hot(y)

        return -true_class_prob / pred
    
    def __labels_to_one_hot(self, y : np.ndarray) -> np.ndarray:
        if len(y.shape) == 0:
            one_hot = np.zeros(shape = (1, 10))
            one_hot[0, y] = 1
        else:
            one_hot = np.zeros(shape = (y.shape[0], 10))
            one_hot[np.arange(y.size), y] = 1

        return one_hot

def train(net : NeuralNetwork, train_data : np.ndarray, val_data : np.ndarray, epochs=10):
    X_train, y_train = train_data
    X_val, y_val = val_data

    avg_train_losses = []
    avg_val_losses = []

    avg_train_accuracies = []
    avg_val_accuracies = []

    # For each epoch
    for epoch in range(1, epochs + 1):

        avg_val_loss = get_validation_loss(net, X_val, y_val)
        avg_val_losses.append(avg_val_loss)

        val_accuracy = eval(net, val_data)
        avg_val_accuracies.append(val_accuracy)

        losses = []

        # For each example in the training set
        for i in range(X_train.shape[0]):
            X_batch = np.array(X_train[i])
            y_batch = np.array(y_train[i])

            loss, (weight_grad, bias_grad) = net.backward(X_batch, y_batch)

            losses.append(loss)

            # Updating the weights and biases
            for i in range(len(net.weights)):
                net.weights[i] -= LEARNING_RATE * weight_grad[i]
                net.biases[i] -= LEARNING_RATE * bias_grad[i]

        avg_loss = np.mean(losses)
        avg_train_losses.append(avg_loss)

        train_accuracy = eval(net, train_data)
        avg_train_accuracies.append(train_accuracy)

        print(epoch, avg_loss, avg_val_loss, train_accuracy, val_accuracy)

        if epoch % 5 == 0 and epoch != 0:
            save_data(net, avg_train_losses, avg_val_losses)

    return net, avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies

def get_validation_loss(net : NeuralNetwork, X_val : np.ndarray, y_val : np.ndarray) -> float:
    losses = []

    for i in range(X_val.shape[0]):
        X_batch = np.array(X_val[i])
        y_batch = np.array(y_val[i])

        loss, _ = net.backward(X_batch, y_batch)
        losses.append(loss)

    return np.mean(losses)


def eval(net : NeuralNetwork, test_data : np.ndarray):
    X_test, y_test = test_data

    correct_guesses = 0

    for i in range(X_test.shape[0]):
        X_batch = np.array(X_test[i])
        y_batch = np.array(y_test[i])

        pred_vector = net.forward(X_batch)
        prediction = pred_vector.argmax()

        #print('Prediction:', prediction, 'Actual: ', y_batch)

        if prediction == y_batch:
            correct_guesses += 1

    return float(correct_guesses) / float(X_test.shape[0])

def save_data(net, avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies):
    # Creating a method for saving everything
    # Create a directory for the current run
    run_dir = os.path.join("saves", SAVE_NAME + '_' + start_date_time)
    os.makedirs(run_dir)

    # Saving the weights and biases
    with open(os.path.join(run_dir, 'weights.pkl'), 'wb') as f:
        pickle.dump(net.weights, f)
        pickle.dump(net.biases, f)

    # Saving the losses in a pandas csv
    df = pd.DataFrame({'train_loss': avg_train_losses, 'val_loss': avg_val_losses, 'train_accuracy': avg_train_accuracies, 'val_accuracy': avg_val_accuracies})
    df.to_csv(os.path.join(run_dir, 'losses.csv'), index=False)

def main():
    global start_date_time
    start_date_time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    train_data, test_data, val_data = load_mnist_data()

    net = NeuralNetwork(layers = LAYERS)

    # Loading saved weights and biases
    if LOAD_WEIGHTS:
        with open('weights.pkl', 'rb') as f:
            weights = pickle.load(f)
            biases = pickle.load(f)
            net.weights = weights
            net.biases = biases

    net, avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies = train(net, train_data, val_data, epochs = EPOCHS)

    save_data(net, avg_train_losses, avg_val_losses)

    accuracy_on_test = eval(net, test_data)
    print('Accuracy: ', accuracy_on_test)

'''
Epoch Train Loss and Val loss of 20 epochs
1 1.8910248125755844 0.9066598151773069
2 0.7853883801136732 0.705261561068248
3 0.6955340516656342 0.6693685850868711
4 0.6822431915452479 0.6796129016275548
5 0.7173728657741884 0.747428342082974
6 0.8274607238245256 0.907169095318536
7 1.0052239265308296 1.0880439811614062
8 1.1613310888744866 1.2230937263566783
9 1.284066508283541 1.338250384763169
10 1.3958711280865803 1.4437385398300528
11 1.494520786077842 1.534511693267978
12 1.5771848213769906 1.6089244242025091
13 1.6441051550342975 1.6690671529427084
14 1.6992237664247059 1.718997491143662
15 1.7443720205997044 1.7595907995988858
16 1.7818428704069464 1.7945413506939825
17 1.8148602666607832 1.8261337171909462
18 1.8453701248460639 1.855368090920725
19 1.873594312775242 1.8826773416419804
20 1.9004549753505733 1.9088665722050886
'''

if __name__ == '__main__':
    main()