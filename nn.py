import numpy as np

class NN:
    def __init__(self, activation_function, loss_function, hidden_layers=[1024], input_d=784, output_d=10):
        self.weights = []
        self.biases = []
        self.activation_function = activation_function
        self.loss_function = loss_function

        # Initialization of weights and biases
        d1 = input_d
        hidden_layers.append(output_d)
        for d2 in hidden_layers:
            self.weights.append(np.random.randn(d2, d1)*np.sqrt(2.0/d1))
            self.biases.append(np.zeros((d2,1)))
            d1 = d2

    def print_model(self):
        """
        This function prints the shapes of weights and biases for each layer.
        """
        print("activation:{}".format(self.activation_function.__class__.__name__))
        print("loss function:{}".format(self.loss_function.__class__.__name__))
        for idx,(w,b) in enumerate(zip(self.weights, self.biases),1):
            print("Layer {}\tw:{}\tb:{}".format(idx, w.shape, b.shape))

    def predict(self, X):
        D = X
        ws = self.weights
        bs = self.biases
        for w,b in zip(ws[:-1], bs[:-1]):
            D = self.activation_function.activate(np.matmul(w,D)+b) 
            # Be careful of the broadcasting here: (d,N) + (d,1) -> (d,N) while (d,) + (d,1) -> (d,d), which is wrong
        Yhat = np.matmul(ws[-1], D)+bs[-1]
        return np.argmax(Yhat, axis=0)

    def compute_gradients(self, X, Y):
        ws = self.weights
        bs = self.biases
        D_stack = []

        #TODO 4: Implement forward pass to get Yhat and intermediate results for BackProp (which is similar to self.predit).         #Store intermediate results in D_stack for backpropagation

        Yhat = np.matmul(ws[-1], D) + bs[-1]

        training_loss = self.loss_function.loss(Y, Yhat)
        '''
        '''
        grad_b = []
        grad_W = []
        # TODO 5: Calculate grad_b and grad_W, which are lists of gradients for b's and w's of each layer. 
        # Take a look at the update step if you are not sure about the format.

        return training_loss, grad_W, grad_b

    def update(self, grad_W, grad_b, learning_rate):
        # Update the weights and biases
        num_layers = len(grad_W)
        ws = self.weights
        bs = self.biases
        for idx in range(num_layers):
            ws[idx] -= (grad_W[idx] * learning_rate)
            bs[idx] -= (grad_b[idx] * learning_rate)
        self.weights = ws
        self.biases = bs
        return 

class activationFunction:
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        raise NotImplementedError("Abstract class.")

    def backprop_grad(self, grad):
        """
        The output of backprop_grad should have the same shape as X
        """
        raise NotImplementedError("Abstract class.")

class Relu(activationFunction):
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        return X*(X>0)

    def backprop_grad(self, X):
        """
        The output of backprop_grad should have the same shape as X
        """
        return (X>0).astype(np.float64)

class Linear(activationFunction):
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        return X
    def backprop_grad(self,X):
        """
        The output of backprop_grad should have the same shape as X
        """
        return np.ones(X.shape, dtype=np.float64)

class LossFunction:
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        raise NotImplementedError("Abstract class.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat
        """
        raise NotImplementedError("Abstract class.")

class SquaredLoss(LossFunction):
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        #TODO 0: loss function for squared loss.
        raise NotImplementedError("Implement SquaredLoss.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat
        """
        #TODO 1: gradient for squared loss.
        raise NotImplementedError("Implement SquaredLoss.")


class CELoss(LossFunction):
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        #TODO 2: loss function for cross-entropy loss.
        raise NotImplementedError("Implement CELoss.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat, which
        has the same shape of Yhat and Y.
        """
        #TODO 3: gradient for cross-entropy loss.
        raise NotImplementedError("Implement CELoss")
