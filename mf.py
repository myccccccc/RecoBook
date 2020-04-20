import numpy as np
import time
from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings("error")


class MF():
    
    def __init__(self, R, K, alpha, beta, iterations, decay_rate=1, val_size=0):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        - decay_rate (float): should between 0.0 and 1.0for every new iteration self.alpha *= self.decay_rate
        - val_size (float): should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation set.
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.decay_rate = decay_rate
        self.val_size = val_size

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.random((self.num_users, self.K)) * 2 - 1
        self.Q = np.random.random((self.num_items, self.K)) * 2 - 1
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create samples
        samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        self.train_samples, self.val_samples = train_test_split(samples, test_size=self.val_size)
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):

            time_start=time.time()

            np.random.shuffle(self.train_samples)
            self.sgd()
            train_rmse = self.rmse("train")
            val_rmse = self.rmse("val")
            training_process.append((i, train_rmse, val_rmse))
            self.alpha *= self.decay_rate
            
            time_end=time.time()

            print("Iteration: %d ; train_rmse = %.4f ; val_rmse = %.4f" % (i, train_rmse, val_rmse), end="\t")
            print('time cost',time_end-time_start,'s')
        
        return training_process

    def rmse(self, t):
        """
        A function to compute the total mean square error
        - t (string) : if t == "train" compute train_rmse, if t == "val" compute test_rmse
        """
        if t == "train":
            samples = self.train_samples
        else:
            samples = self.val_samples
            if not samples:
                return 0
        predicted = self.full_matrix()
        error = 0
        for x, y, rating in samples:
            error += pow(rating - predicted[x, y], 2)
        return np.sqrt(error / len(samples)) # in case there is no val_samples

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.train_samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            change = self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.P[i, :] += np.array(list(map(lambda x: 0 if abs(x) > 0.1 else x, change))) # avoid gradient exploding problem
            change = self.alpha * (e * P_i - self.beta * self.Q[j,:])
            self.Q[j, :] += np.array(list(map(lambda x: 0 if abs(x) > 0.1 else x, change))) # avoid gradient exploding problem
            
            # try:
            #     change = self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            #     # avoid gradient exploding problem
            #     for t in range(len(change)):
            #         if abs(change[t]) > 0.1:
            #             continue
            #         else:
            #             self.P[i, t] += change[t]
            # except RuntimeWarning:
            #     print("i:{} j:{}".format(i, j))
            #     print("self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])")
            #     print("e = {}, self.Q[j, :] = {}, self.P[i,:] = {}".format(e, self.Q[j, :], self.P[i,:]))
            #     self.P[i, :] = np.random.normal(scale=1./self.K, size=(self.K))
                
            # try:
            #     change = self.alpha * (e * P_i - self.beta * self.Q[j,:])
            #     # avoid gradient exploding problem
            #     for t in range(len(change)):
            #         if abs(change[t]) > 0.1:
            #             continue
            #         else:
            #             self.Q[j, t] += change[t]
            # except RuntimeWarning:
            #     print("i:{} j:{}".format(i, j))
            #     print("self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])")
            #     print("e = {}, P_i = {}, self.Q[j,:] = {}".format(e, P_i, self.P[i,:]))
            #     self.Q[j, :] = np.random.normal(scale=1./self.K, size=(self.K))
            

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
