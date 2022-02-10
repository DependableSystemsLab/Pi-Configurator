import numpy as np
import pandas

errorList = []


def dimension_changer(filename, featurename):
    df = pandas.read_csv(filename)

    df_pivot = pandas.pivot_table(df, index="Upstream_Knobs",
                                  columns="Downstream_Knobs")

    df_pivot.sort_index(axis=0)

    df_accuracy = df_pivot[featurename]

    MatrixFileName = filename[:-4]+'_'+featurename+".csv"
    df_accuracy.to_csv(MatrixFileName)
    return MatrixFileName


class MF():

    def __init__(self, RO, R, xmax, xmin, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - RO (ndarray)  : original complete matrix
        - R (ndarray)   : to be completed matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.RO = RO
        self.R = R
        self.xmax = xmax
        self.xmin = xmin
        self.num_upstream, self.num_downstream = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize upstream and downstream latent feature matrice
        self.P = np.random.normal(
            scale=1./self.K, size=(self.num_upstream, self.K))
        self.Q = np.random.normal(
            scale=1./self.K, size=(self.num_downstream, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_upstream)
        self.b_i = np.zeros(self.num_downstream)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_upstream)
            for j in range(self.num_downstream)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse, errorPercent = self.mse()
            errorList.append(mse)
        print(
            #"The last mse: ", round(mse, 2),
            "Error Percentage: ", round(errorPercent, 2))

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = np.transpose(np.argwhere(R == 0))
        predicted = self.full_matrix()
        squareError = 0
        diffError = 0
        self.ind = zip(xs, ys)
        count = 0
        for x, y in zip(xs, ys):
            count = count + 1
            predicted_pos_val = (
                predicted[x, y] * (self.xmax - self.xmin))+self.xmin
            squareError += pow(predicted_pos_val - self.RO[x, y], 2)
            diffError += abs(predicted_pos_val - self.RO[x, y])
        count = count+1
        mse = np.sqrt(squareError/count)
        errorPercentage = (diffError/sumQoS)*100
        return mse, errorPercentage

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Compute prediction and error
            prediction = self.get_QoS(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update downstream and downstream knowb latent feature matrices
            self.P[i, :] += self.alpha * \
                (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * \
                (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_QoS(self, i, j):
        """
        Get the predicted QoS of upstream i and downsteam j
        """
        prediction = self.b + self.b_u[i] + \
            self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Compute the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


filename = ""  # Input dataset
featurename = ""  # Input feature name e.g., CPUCycle, WallClock, Memory, Accuracy
QoS_File = dimension_changer(filename, featurename)

a = pandas.read_csv(QoS_File, sep=",")
a = a.drop(a.columns[0], axis=1)

R = a.values
RO = R.copy()
row, col = R.shape

degree = 1  # degree of sparcity
sumQoS = np.ndarray.sum(R[degree:-degree, degree:-degree])
R[degree:-degree, degree:-degree] = 0  # Generating the training data
print("Data Used %: ", round(
    ((RO.size-R[degree:-degree, degree:-degree].size)/RO.size)*100))  # Calculate training data size

np.savetxt('matrixBlanked.csv', R, delimiter=',', fmt='%.2f')


xmax_new, xmin_new = R.max(), R.min()
R_new = (R - xmin_new)/(xmax_new - xmin_new)
mf = MF(RO, R_new, xmax_new, xmin_new, K=row+col,
        alpha=0.1, beta=0.02, iterations=1000)
mf.train()
x = mf.full_matrix()
y = (x*(xmax_new - xmin_new))+xmin_new

for a, b in mf.ind:
    R[a, b] = y[a, b]

np.savetxt('matrixFilled.csv', R, delimiter=',', fmt='%.2f')

print()
print("Global bias:")
print(mf.b)
print()
print("Upstream knob bias:")
print(mf.b_u)
print()
print("Downstream knob bias:")
print(mf.b_i)
