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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    def __init__(self, k=3):
        seed = 60012  # set random seed to obtain reproducible results
        rg = default_rng(seed)
        self.folds = k
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
            train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

            folds.append([train_indices, test_indices])

        return folds



class Regressor(torch.nn.Module):

    def __init__(self, x, nb_epoch=2000, nodes_h_layers=[60, 80], activation="relu", lr=0.455, dropout=0.0):
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
        super(Regressor, self).__init__()
        self.labelEncoder = dict.fromkeys([col for col in x.columns if x[col].dtype in ["bool", "object"]])
        self.scaler = None
        self.data_mean = None
        X, _ = self._preprocessor(x, training=True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.nodes_h_layer = nodes_h_layers
        activations = {"tanh": torch.nn.Tanh(), "relu": torch.nn.ReLU(True)}
        self.activation = activations[activation]
        self.layers = []
        self.lr = lr
        self.dropout = dropout 

        if len(self.nodes_h_layer) == 0:
            self.layer_1 = torch.nn.Linear(in_features=self.input_size, out_features=self.output_size)
            self.layers += [self.layer_1]
        else:
            i = 0
            structure = [self.input_size] + self.nodes_h_layer + [self.output_size]
            # set attributes layer_i according to inputs
            while i < (len(structure) - 1):
                name = "layer_" + str(i + 1)
                setattr(self, name, torch.nn.Linear(in_features=structure[i], out_features=structure[i + 1]))
                self.layers += [getattr(self, name)]
                # add activation functions and dropouts
                if i < (len(structure) - 2) and activation is not None:
                    self.layers += [self.activation]
                    self.layers += [torch.nn.Dropout(p=self.dropout)]
                i += 1

        self.model = torch.nn.Sequential(*self.layers)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.model = None
        self.losses = []
        self.val_losses = []

    def forward(self, x):
        """Performs a forward pass through the regressor.
        
        Args:
            x {np.ndarray} -- Processed input array of size (batch_size, input_size).
            
        Returns:
            output {np.ndarray} -- Predictions from current state of the model"""
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
            - x {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of size (batch_size, input_size).
            - y {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of size (batch_size, 1).

        """
        if training:
            # Impute the missing values in the data set
            self.data_mean = x.mean(numeric_only=True)
            x.fillna(x.mean(numeric_only=True), inplace=True)

            # Label encode for label features (specific to housing data)
            for col in list(x.columns):
                if x[col].dtype in ["bool", "object"]:
                    labelEncoder = LabelEncoder()
                    x[col] = labelEncoder.fit_transform(x[col])
                    self.labelEncoder[col] = labelEncoder

            #Normalise training data
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            self.scaler = scaler

        # if test\validation, use the stored parameters
        else:
            # Impute the missing values in the data set
            x.fillna(self.data_mean, inplace=True)

            # encode non-numerate features if it wasn't encoded before
            for col in list(x.columns):
                if x[col].dtype in ["bool", "object"]:
                    x[col] = x[col].map(lambda s: -1 if s not in self.labelEncoder[col].classes_ else s)
                    self.labelEncoder[col].classes_ = np.append(self.labelEncoder[col].classes_, -1)
                    x[col] = self.labelEncoder[col].transform(x[col])

            # Standardize test data
            x = self.scaler.transform(x)

        # Return preprocessed x and y, return None for y if it was None
        return x, y

    def fit(self, x_train, y_train, x_val = None, y_val = None):
        """
        Regressor training function. 

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        """
        X_train, Y_train = self._preprocessor(x_train, y=y_train, training=True) #process train data with standard scaler and label encoder

        #convert data frame to tensor for the NN
        X_train = torch.tensor(X_train, dtype=torch.float)
        Y_train = torch.tensor(Y_train.values, dtype=torch.float)

        #reshape target tensor 
        new_shape_train = (len(Y_train), 1)
        Y_train = Y_train.view(new_shape_train)

        if x_val is not None and y_val is not None:
            X_val, Y_val = self._preprocessor(x_val, y=y_val) #process val data with standard scaler and label encoder

            #convert data frame to tensor for the NN
            X_val = torch.tensor(X_val, dtype=torch.float)
            Y_val = torch.tensor(Y_val.values, dtype=torch.float)

            #reshape target tensor 
            new_shape_val = (len(Y_val), 1)
            Y_val = Y_val.view(new_shape_val)

        loss_func = nn.MSELoss() #define loss function

        # train the model with nb_epoch epochs and present the loss for each epoch
        for t in range(self.nb_epoch):
            #training loss
            train_prediction = self.forward(X_train)  # redict based on x_train
            loss_train = loss_func(train_prediction, Y_train)  # must be (1. nn output, 2. target)
            self.losses.append(np.sqrt(loss_train.item()))  # root mean squared error

            #same operations as above for val data
            if x_val is not None and y_val is not None:
                val_prediction = self.forward(X_val) 
                loss_val = loss_func(val_prediction, Y_val) 
                self.val_losses.append(np.sqrt(loss_val.item())) 

            #condition used in testing code (can be removed)
            if torch.isnan(loss_train):
                break

            #perform backpropagation using the training loss obtained
            self.optimiser.zero_grad()  # clear gradients for next train
            loss_train.backward()  # backpropagation, compute gradients
            self.optimiser.step()  # apply gradients

            #print different message if using validation data or not
            if x_val is not None and y_val is not None:
                print(f'epoch {t + 1} finished with ---> train_loss = {np.sqrt(loss_train.item()):.4f} ;  val_loss = {np.sqrt(loss_val.item()):.4f}')
            else:
                print(f'epoch {t + 1} finished with ---> train_loss = {np.sqrt(loss_train.item()):.4f}')


    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            prediction {np.darray} -- Predicted value for the given input (batch_size, 1).

        """
        X, _ = self._preprocessor(x, training=False)  

        # convert data frame to tensor for the NN
        X = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            prediction = self.forward(X)

        prediction = np.array(prediction)
        return prediction

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a test dataset when evaluating on unseen data.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            rmse {float} -- Quantification of the efficiency of the model.

        """
        self.eval()
        Y_pred = self.predict(x)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y, Y_pred))
        return rmse
    
    def score_training(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset whilst training.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            rmse {float} -- Quantification of the efficiency of the model.

        """

        self.train()
        Y_pred = self.predict(x)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y, Y_pred))
        return rmse

    def plot_losses(self, val=False):
        """"""
        plt.figure()
        plt.grid()
        plt.plot(np.linspace(0, len(self.losses), len(self.losses)), self.losses)

        if val:
            plt.plot(np.linspace(0, len(self.val_losses), len(self.val_losses)), self.val_losses)

        plt.xlabel("Epoch")
        plt.ylabel("RMSE Loss")
        plt.legend(['train loss', 'validation loss'])
        plt.show()


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.

    Args:
        trained_model (Regressor()): Trained Regressor() object.

    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.

    Returns:
        trained_model (Regressor()): Trained Regressor() object.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model part2_model.pickle\n")
    return trained_model

def RegressorHyperParameterSearch(x_trainval, y_trainval,  x_test, y_test, lr_list, dropouts, num_layers, 
                                  minNodes, maxNodes, step, activations_list=["tanh", "relu"], nb_epochs = 2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        x_trainval (pd.DataFrame): Dataframe containing the input training data to the regressor. 
        y_trainval (pd.DataFrame): Dataframe containing the target training data to the regressor. 
        x_test (pd.DataFrame): Dataframe containing the input test data to evaluate the regressor. 
        y_test (pd.DataFrame):Dataframe containing the target test data to evaluate the regressor. 
        lr_list (np.ndarray): 
        dropouts (np.ndarray): 
        num_layers (int): 
        minNodes (int): 
        maxNodes (int): 
        step (int): 
        activations_list {list}: 
        nb_epochs {int}: number of epochs to train the models with 


    Returns:
        best_params (dict): Dictionary containing optimised hyper-parameters.

    """
    best_params = {"lr": None, "dropout": None, "activation": None, "n_per_layer": None}
    possible_n_per_layer = [np.arange(minNodes, maxNodes, step) for _ in range(num_layers)]
    node_combinations = [list(x) for x in np.array(np.meshgrid(*possible_n_per_layer)).T.reshape(-1,len(possible_n_per_layer))]
    best_val_err = float("inf")
    worst_models = {}
    best_models = {}
    cv = CrossValidation()

    total_hp_combs = len(node_combinations)*len(activations_list)*len(dropouts)*len(lr_list)

    i = 0
    for nodes_h_layers in node_combinations:
        for activation in activations_list:
            for dropout in dropouts:
                for lr in lr_list:
                    i += 1
                    print(f"\nModel {i}/{total_hp_combs}")
                    splits = cv.train_test_k_fold(x_trainval, y_trainval)
                    cv_val_errors = []
                    for j, (train_indices, val_indices) in enumerate(splits):
                        print(f"inner fold {j+1}/{cv.folds}")
                        #retrieve training and validation sets from random indices (splits)
                        x_train = x_trainval.iloc[train_indices, :]
                        y_train = y_trainval.iloc[train_indices]
                        x_val = x_trainval.iloc[val_indices, :]
                        y_val = y_trainval.iloc[val_indices]

                        regressor = Regressor(x_train, nb_epoch=nb_epochs, nodes_h_layers=nodes_h_layers, 
                                              activation=activation, lr=lr, dropout=dropout)
                        regressor.fit(x_train, y_train, x_val=x_val, y_val=y_val)
                        val_err = regressor.score_training(x_val, y_val)
                        cv_val_errors.append(val_err)

                    mean_val_error = np.mean(cv_val_errors)

                    #rank top 5 worst models
                    if len(worst_models) < 5:
                        worst_models[(tuple(nodes_h_layers),activation,dropout,lr)] = mean_val_error
                    
                    else:
                        if val_err > np.min(list(worst_models.values())):
                            del worst_models[min(worst_models, key=worst_models.get)]
                            worst_models[(tuple(nodes_h_layers),activation,dropout,lr)] = mean_val_error
                    
                    #rank top 5 best models
                    if len(best_models) < 5:
                        best_models[(tuple(nodes_h_layers),activation,dropout,lr)] = mean_val_error
                    
                    else:
                        if val_err < np.max(list(best_models.values())):
                            del best_models[max(best_models, key=best_models.get)]
                            best_models[(tuple(nodes_h_layers),activation,dropout,lr)] = mean_val_error

                    if mean_val_error < best_val_err:
                        best_val_err = mean_val_error

                        print(f"\nUpdated best val error: {best_val_err}")
                        print(f"Hyperparemeters:")
                        print(f"Activation = {activation}")
                        print(f"Learning rate = {lr}")
                        print(f"Nodes per layer = {nodes_h_layers}")
                        print(f"Dropout = {dropout}")

                        best_params["lr"] = lr
                        best_params["dropout"] = dropout
                        best_params["activation"] = activation
                        best_params["n_per_layer"] = nodes_h_layers


    #save best regressor 
    regressor = Regressor(x_train, nb_epoch=nb_epochs, nodes_h_layers=best_params["n_per_layer"], 
                          activation=best_params["activation"], lr=best_params["lr"], dropout=best_params["dropout"])
    regressor.fit(x_train, y_train)
    print("Best regressor has validation error: ", best_val_err)
    #regressor.plot_losses()
    test_err = regressor.score(x_test, y_test)
    print("Best regressor has test error: ", test_err)

    #save best regressor
    save_regressor(regressor)

    return best_params


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")
    x = data.drop(columns=output_label)
    y = data[output_label]

    # options for train test split
    (x_train_val, x_test, 
    y_train_val, y_test) = train_test_split(x,y,test_size=0.20, random_state=42)

    #80/20/20 (train/test/val)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)

    # fit regressor
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train, x_val, y_val)
    #regressor.plot_losses(val=True) #plot losses
    print("Final validation loss: ", regressor.val_losses[-1])

    #save regressor
    save_regressor(regressor)

    #evaluate model on test data
    test_err = regressor.score(x_test, y_test)
    print("\nTest regressor error: {}\n".format(test_err))

    #test regressor loading 
    loaded_regressor = load_regressor()
    loaded_test_err = loaded_regressor.score(x_test, y_test)
    print("\nLoaded model test regressor error: {}\n".format(loaded_test_err))

    test_predictions = loaded_regressor.predict(x_test)

    #plt.figure()
    #plt.grid()
    #plt.plot(test_predictions, y_test, "r.", markersize=4)
    #plt.plot(y_test, y_test, "black", label="Expected result")
    #plt.xlabel("Prediction")
    #plt.ylabel("True Values")
    #plt.legend()
    #plt.show()

def example_tuning(num_layers):
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")
    x = data.drop(columns=output_label)
    y = data[output_label]

    #options for train test split
    (x_trainval, x_test, 
    y_trainval, y_test) = train_test_split(x,y,test_size=0.20, random_state=42)

    #define hyperparameter ranges
    minNodes = 40 
    maxNodes = 120 
    step = 20 
    lr_list = np.linspace(0.01, 0.9, 5)
    dropouts_list = np.linspace(0.0, 0.5, 5)

    best_params = RegressorHyperParameterSearch(x_trainval, y_trainval, x_test, y_test, lr_list, 
                                                                             dropouts_list, num_layers, minNodes, maxNodes, step,
                                                                             activations_list=["tanh", "relu"])

    print(f"Best hyperparameters for {num_layers} hidden layers: ", best_params)

    #print("\nTop 5 worst models: \n", worst_models)

    #print("\nTop 5 best models: \n", best_models)

    #best_regressor = load_regressor()
    #best_regressor.plot_losses()

if __name__ == "__main__":
    #num_layers = int(sys.argv[1])
    #example_tuning(num_layers)
    example_main()
