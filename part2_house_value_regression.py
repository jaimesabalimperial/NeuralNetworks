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

class Net(nn.Module):
    def __init__(self, D_in, D_out, H1=500, H2=1000, H3=200):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
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
        self.imputer = None
        return

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
            print(x.isnull().sum())
            imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="median")
            x.iloc[:, 4:5] = imputer.fit_transform(x.iloc[:, 4:5])
            self.imputer = imputer
            x.isnull().sum()

            # Label encode for categorical feature (ocean_proximity)
            print(x.dtypes)
            labelEncoder = LabelEncoder()
            print(x["ocean_proximity"].value_counts())
            x["ocean_proximity"] = labelEncoder.fit_transform(x["ocean_proximity"])
            self.labelEncoder = labelEncoder
            x["ocean_proximity"].value_counts()
            x.describe()

            # Standardize training  data
            independent_scaler = StandardScaler()
            x = independent_scaler.fit_transform(x)
            self.independent_scaler = independent_scaler

            # convert data frame to tensor for the NN
            x = torch.tensor(x, dtype=torch.float)
            if y is not None:
                y = torch.tensor(y.values, dtype=torch.float)

        # if test\validation, use the stored parameters
        else:
            # Impute the missing values in the data set
            x.iloc[:, 4:5] = self.imputer.fit_transform(x.iloc[:, 4:5])

            # encode non-numerate features
            x["ocean_proximity"] = self.labelEncoder.transform(x["ocean_proximity"])
            x["ocean_proximity"].value_counts()
            x.describe()

            # Standardize test data
            x = self.independent_scaler.transform(x)

            # convert data frame to tensor for the NN
            x = torch.tensor(x, dtype=torch.float)
            if y is not None:
                y = torch.tensor(y.values, dtype=torch.float)

        # Return preprocessed x and y, return None for y if it was None
        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
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
        regressor = Net(self.input_size, self.output_size)
        loss_func = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-4)

        losses = []
        for t in range(self.nb_epoch):
            prediction = regressor(X)  # input x and predict based on x
            loss = loss_func(prediction, Y)  # must be (1. nn output, 2. target)
            losses.append(loss.item())
            print(f'epoch {t} finished with training loss: {loss}.')
            if torch.isnan(loss):
                break
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        plt.figure()
        plt.plot(range(len(losses)), losses)
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

        X, _ = self._preprocessor(x, training=False) # Do not forget
        with torch.no_grad():
            saved_model = load_regressor()
            prediction = saved_model(x)
        return prediction

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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=100)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

