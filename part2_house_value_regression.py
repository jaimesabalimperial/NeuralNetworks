import copy

import matplotlib.pyplot as plt
import sklearn.impute
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import torch
import torch.nn as nn

pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
from numpy.random import default_rng


class CrossValidation:

    def __init__(self):
        seed = 60012  # set random seed to obtain reproducible results
        rg = default_rng(seed)
        self.folds = None
        self.random_generator = rg

    def k_fold_split(self, x, y):
        """ Split n_instances into n mutually exclusive splits at random.

        Args:
            n_splits (int): Number of splits
            n_instances (int): Number of instances to split
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list (length n_splits). Each element in the list should contain a
                numpy array giving the indices of the instances in that split.
        """

        # generate a random permutation of indices from 0 to n_instances
        shuffled_indices = self.random_generator.permutation(len(x))

        # split shuffled indices into almost equal sized splits
        split_indices = np.array_split(shuffled_indices, self.folds)

        return split_indices

    def train_test_k_fold(self, x, y):
        """ Generate train and test indices at each fold.

        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list of length n_folds. Each element in the list is a list (or tuple)
                with two elements: a numpy array containing the train indices, and another
                numpy array containing the test indices.
        """

        # split the dataset into k splits
        split_indices = self.k_fold_split(x, y)

        folds = []
        for k in range(self.folds):
            # pick k as test
            test_indices = split_indices[k]

            # combine remaining splits as train
            # this solution is fancy and worked for me
            # feel free to use a more verbose solution that's more readable
            train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

            folds.append([train_indices, test_indices])

        return folds


class Regressor(torch.nn.Module):

    def __init__(self, x, nb_epoch=100, nodes_h_layers=[400,150,50], activation="relu", lr=0.01, dropout=0.1):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        super(Regressor, self).__init__()
        X, _ = self._preprocessor(x, training=True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.nodes_h_layer = nodes_h_layers
        activations = {"tanh": torch.nn.Tanh(), "relu": torch.nn.ReLU(True)}
        self.activation = activations[activation]
        self.layers = []
        self.lr = lr
        self.dropout=dropout

       
        if len(self.nodes_h_layer) == 0:
            self.layer_1 = torch.nn.Linear(in_features=self.input_size, out_features=self.output_size)
            self.layers += [self.layer_1]
        else:
            i = 0
            structure = [self.input_size] + self.nodes_h_layer + [self.output_size]
            # set attributes layer_i according to inputs
            while i < (len(structure)-1):
                name = "layer_" + str(i+1)
                setattr(self, name, torch.nn.Linear(in_features=structure[i], out_features=structure[i + 1]))
                self.layers += [getattr(self, name)]
                # add activation functions and dropouts
                if i < (len(structure) - 2) and activation is not None:
                    self.layers += [self.activation]
                    self.layers += [torch.nn.Dropout(p=self.dropout)]
                i += 1

        self.model = torch.nn.Sequential(*self.layers)
        print(self.model)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)  
        self.scaler = None
        self.labelEncoder = None
        self.data_mean = None
        self.model = None
        self.losses = None

    def forward(self, x):
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        for layer in self.layers:
            output = layer(x)
            x = output
            
        return output 

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if training:
            # presenting statistics of the data
            #print("The number of rows and colums are {} and also called shape of the matrix".format(x.shape))
            #print("Columns names are \n {}".format(x.columns))

            # Impute the missing values in the data set
            self.data_mean = x.mean(numeric_only=True)
            x.fillna(x.mean(numeric_only=True), inplace=True)
                
            # Label encode for label features
            #print(x.dtypes)
            labelEncoder = LabelEncoder()
            #print(x["ocean_proximity"].value_counts())
            x["ocean_proximity"] = labelEncoder.fit_transform(x["ocean_proximity"])
            self.labelEncoder = labelEncoder
            x["ocean_proximity"].value_counts()
            x.describe()

            # Standardize training data
            # Standardize x
            independent_scaler = StandardScaler()
            x = independent_scaler.fit_transform(x)
            self.independent_scaler = independent_scaler

        # if test\validation, use the stored parameters
        else:
            #Impute the missing values in the data set
            x.fillna(self.data_mean, inplace=True)

            # encode non-numerate features if it wasn't encoded before
            x["ocean_proximity"] = x["ocean_proximity"].map(lambda s: -1 if s not in self.labelEncoder.classes_ else s)
            self.labelEncoder.classes_ = np.append(self.labelEncoder.classes_, -1)
            x["ocean_proximity"] = self.labelEncoder.transform(x["ocean_proximity"])

            # Standardize test data
            x = self.independent_scaler.transform(x)

        # Return preprocessed x and y, return None for y if it was None
        return x, y

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        # convert data frame to tensor for the NN
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y.values, dtype=torch.float)
        new_shape = (len(Y), 1)
        Y = Y.view(new_shape)

        loss_func = nn.MSELoss()

        #train the model with nb_epoch epochs and present the loss for each epoch
        losses = []
        for t in range(self.nb_epoch):
            #training loss
            prediction = self.forward(X)  # input x and predict based on x
            loss_train = loss_func(prediction, Y)  # must be (1. nn output, 2. target)
            losses.append(np.sqrt(loss_train.item())) #root mean squared error
            if torch.isnan(loss_train):
                break
            self.optimiser.zero_grad()  # clear gradients for next train
            loss_train.backward()  # backpropagation, compute gradients
            self.optimiser.step()  # apply gradients
            print(f'epoch {t+1} finished with training loss: {loss_train}.')

        self.losses = losses

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # I am not sure if needed here
        # convert data frame to tensor for the NN
        X = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            prediction = self.forward(X)
        return np.array(prediction)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        Y_pred = self.predict(x)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y, Y_pred))
        return rmse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    def plot_losses(self):
        plt.figure()
        plt.grid()
        plt.plot(np.linspace(0, len(self.losses), len(self.losses)), self.losses)
        plt.plot(np.linspace(0, len(self.losses_val), len(self.losses_val)), self.losses_val)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend(['train loss', 'validation loss'])
        plt.show()


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch():
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # options for train test split
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=output_label),
        data[output_label],
        test_size=0.20, random_state=42)

    # fit regressor
    regressor = Regressor(x_train, nb_epoch=100, nodes_h_layers=[300,350,50], activation="relu", lr=0.01, dropout=0.2)
    regressor.fit(x_train, y_train)
    #y_pred = regressor.predict(x_test)
    #sqrt_error = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
    #print(sqrt_error)
    test_err = regressor.score(x_test, y_test)
    print("\nTest regressor error: {}\n".format(test_err))

    #regressor.plot_losses()
    save_regressor(regressor)

    new_regressor = load_regressor()



if __name__ == "__main__":
    example_main()

