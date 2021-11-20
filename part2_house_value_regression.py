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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import torch
import torch.nn as nn

pd.options.mode.chained_assignment = None  # default='warn'

class Net(nn.Module):
    def __init__(self, D_in, D_out, H1=400, H2=150, H3=50):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y_pred = self.relu(self.linear1(x).clamp(min=0))
        y_pred = self.relu(self.linear2(y_pred).clamp(min=0))
        y_pred = self.dropout(y_pred)
        y_pred = self.relu(self.linear3(y_pred).clamp(min=0))
        y_pred = self.dropout(y_pred)
        y_pred = self.linear4(y_pred)
        return y_pred


class Regressor():

    def __init__(self, x, nb_epoch=1000):
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

        # Replace this code with your own
        # X, _ = self._preprocessor(x, training=True)
        self.input_size = x.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.independent_scaler = None
        self.labelEncoder = None
        self.data_mean = None
        self.model = None
        self.losses = None
        self.losses_val = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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
            print("The number of rows and colums are {} and also called shape of the matrix".format(x.shape))
            print("Columns names are \n {}".format(x.columns))

            # Impute the missing values in the data set
            self.data_mean = x.mean(numeric_only=True)
            x.fillna(x.mean(numeric_only=True), inplace=True)

            # Label encode for categorical feature (ocean_proximity)
            print(x.dtypes)
            labelEncoder = LabelEncoder()
            print(x["ocean_proximity"].value_counts())
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
            # Impute the missing values in the data set
            x.fillna(self.data_mean, inplace=True)

            # encode non-numerate features if it wasn't encoded before
            if x["ocean_proximity"].dtype != 'int64':
                x["ocean_proximity"] = self.labelEncoder.transform(x["ocean_proximity"])
                x["ocean_proximity"].value_counts()
                x.describe()

            # Standardize test data
            x = self.independent_scaler.transform(x)

        # Return preprocessed x and y, return None for y if it was None
        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y, x_val, y_val):
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
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y.values, dtype=torch.float)
        new_shape = (len(Y), 1)
        Y = Y.view(new_shape)
        X_val, Y_val = self._preprocessor(x_val, y=y_val, training=False)  # Do not forget
        X_val = torch.tensor(X_val, dtype=torch.float)
        Y_val = torch.tensor(Y_val.values, dtype=torch.float)
        new_shape = (len(Y_val), 1)
        Y_val = Y_val.view(new_shape)
        regressor = Net(self.input_size, self.output_size)
        loss_func = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-1)

        # train the model with nb_epoch epochs and present the loss for each epoch
        losses = []
        losses_val = []
        for t in range(self.nb_epoch):
            # training loss
            prediction = regressor(X)  # input x and predict based on x
            loss_train = loss_func(prediction, Y)  # must be (1. nn output, 2. target)
            losses.append(loss_train.item())
            print(f'epoch {t} finished with training loss: {loss_train}.')

            # validation loss
            val_prediction = regressor(X_val)
            loss_val = loss_func(val_prediction, Y_val)
            losses_val.append(loss_val.item())

            if torch.isnan(loss_train):
                break
            optimizer.zero_grad()  # clear gradients for next train
            loss_train.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        self.losses = losses
        self.losses_val = losses_val
        self.model = regressor
        return regressor

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
        X = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            saved_model = load_regressor()
            prediction = saved_model(X)
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
        sqrt_error = np.sqrt(sklearn.metrics.mean_squared_error(y, Y_pred))
        return sqrt_error

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
    #bins = 5
    #sale_price_bins = pd.qcut(
    #    data[output_label], q=bins, labels=list(range(bins)))
    #x_train, x_test, y_train, y_test = train_test_split(
    #    data.drop(columns=output_label),
    #    data[output_label],
    #    random_state=12,
    #    stratify=sale_price_bins)

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=output_label),
        data[output_label],
        test_size=0.2, random_state=42)

    # Spliting input and output
    #x_train = data.loc[:, data.columns != output_label]
    #y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=100)
    regressor.fit(x_train, y_train, x_test, y_test)
    Y_pred = regressor.predict(x_test)
    sqrt_error = np.sqrt(sklearn.metrics.mean_squared_error(y_test, Y_pred))
    regressor.plot_losses()
    save_regressor(regressor.model)

    # Error on test set
    error_train = regressor.score(x_train, y_train)
    print("\nTrain regressor error: {}\n".format(error_train))
    error_test = regressor.score(x_test, y_test)
    print("\nTest regressor error: {}\n".format(error_test))



if __name__ == "__main__":
    example_main()

