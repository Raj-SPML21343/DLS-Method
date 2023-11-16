import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
from scipy.optimize import minimize
import warnings


os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    def Z_function(self, u, w):
        """Calculate the Z value(Predicted Score) using the number of overs to go(u) and wickets in hand(w)."""
        return self.Z0[w-1] * (1 - np.exp(-self.L * u / self.Z0[w-1]))
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        if Z_0 is not None:
            self.Z0[w-1] = Z_0
        if L is not None:
            self.L = L

        predictions = self.Z_function(X, w)
        return predictions

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        self.L, self.Z0[w] = Params
        predictions = self.Z_function(X, w)
        loss = np.mean((predictions - Y) ** 2)
        return loss
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    data = pd.read_csv(data_path)   # convert to pd dataframe
    return data


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    # Convert data to DataFrame if it's a numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    # Keep only First Innings Data
    data = data[data['Innings']==1]

    # Keep only the desired columns
    columns_to_keep = [0, 1, 2, 3, 6, 7, 11]
    data = data.iloc[:, columns_to_keep]
    # Convert column 2 to proper date format (assuming it's in column index 2)
    for index, date_str in enumerate(data.iloc[:, 1]):
        try:
            data.iloc[index, 1] = pd.to_datetime(date_str, format='%d/%m/%Y').strftime('%d-%m-%Y')
        except ValueError:
            # Handle special case 'May 29-30, 1999'
            day_range = date_str.split('-')[1].split(',')[0]
            end_date = day_range.split('-')[-1]
            data.iloc[index, 2] = pd.to_datetime(end_date + '-' + date_str[:3] + '-' + date_str[-4:], format='%d-%b-%Y').strftime('%d-%m-%Y')

    # Remove rows with missing values
    data = data.dropna()

    # add extra rows with overs = 0

    # Sort the data based on Match and Over columns
    data.sort_values(by=['Match', 'Over'], inplace=True)
    # Group by Match and add new row at the beggining with overs = 0
    new_rows = []
    for match, group in data.groupby('Match'):
        new_row = group.iloc[0].copy()  # Copy the first row
        new_row['Over'] = 0
        new_row['Runs'] = 0
        new_row['Total.Runs'] = 0
        new_row['Runs.Remaining'] = new_row['Innings.Total.Runs']
        new_rows.append(new_row)

        for _, row in group.iterrows():
            new_rows.append(row)

    # Create a new DataFrame with the updated rows
    updated_data = pd.DataFrame(new_rows, columns=data.columns)

    return updated_data


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """

    def loss_function(params, data):
        Z0,L = params[:10], params[10]
        loss = 0
        for _, row in data.iterrows():
            w = int(row['Wickets.in.Hand'])
            u = float(50 - row['Over'])
            Y = float(row['Runs.Remaining'])
            # Filter out specific runtime warnings
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                # This line is causing the warning
                prediction = Z0[w-1] * (1 - np.exp(-L * u / Z0[w-1]))
            except RuntimeWarning:
                # Handle the warning gracefully
                prediction = Z0[w-1]  # or any other appropriate value
            loss += (prediction - Y) ** 2
        return loss

    initial_params = [10.0] * 10 + [10.0]  # Initial guesses for parameters (10 Z0 values and 1 L value)
    bounds = [(0, None)] * 10 + [(0, None)]  # Bounds for parameters (Z0 values must be positive, L can be positive)

    optimized_params = minimize(loss_function, initial_params, args=(data,), bounds=bounds).x
    model.L = optimized_params[-1]  # Update L value
    model.Z0 = optimized_params[:-1]  # Update Z0 values

    return model
def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    wickets_values = range(1, 11)  # wickets values from 1 to 10
    overs_remaining = np.linspace(0, 50, 100)  # u values from 0 to 50

    plt.figure(figsize=(10, 6))
    for w in wickets_values:
        plt_predictions = []  # List to store predictions for the current wickets value
        for u in overs_remaining:
            try:
                prediction = model.get_predictions(u, Z_0=model.Z0[w - 1], w=w, L=model.L)  # Get model prediction for the current wickets and overs values
                plt_predictions.append(prediction)
            except RuntimeWarning:  # Handle warnings and set prediction to NaN
                plt_predictions.append(np.nan)
        plt.plot(overs_remaining, plt_predictions, label=f'w = {w}')
        
        # Add wickets value just above the curve
        max_index = np.argmax(plt_predictions)  # Index of the maximum prediction
        if not np.isnan(plt_predictions[max_index]):
            plt.text(overs_remaining[max_index], plt_predictions[max_index], str(w), ha='center', va='bottom')

    plt.xlabel('Overs Remaining')
    plt.ylabel('Average Runs Obtainable')
    plt.title('Model Predictions for Different Wickets-in-Hand')
    
    # Add legend to the plot
    plt.legend(title='Wickets-in-Hand', loc='upper left')

    plt.grid()
    plt.savefig(plot_path)
    plt.show()


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    z_parameters = model.Z0
    l_parameter = model.L
    
    print("Z_0 Parameters:")
    for i, z_param in enumerate(z_parameters):
        print(f"Z_0({i + 1}): {z_param:.4f}")
    
    print(f"L: {l_parameter:.4f}")
    
    return z_parameters + [l_parameter]


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    total_squared_error = 0.0
    total_data_points = 0
    
    for _, row in data.iterrows():
        try:
            actual_runs_remaining = row['Runs.Remaining']
            u = 50 - row['Over']
            w = row['Wickets.in.Hand']
            predicted_runs_remaining = model.get_predictions(u, Z_0=model.Z0[w - 1], w=w, L=model.L)
            squared_error = (predicted_runs_remaining - actual_runs_remaining) ** 2
            total_squared_error += squared_error
            total_data_points += 1
        except (RuntimeWarning, IndexError):
            pass  # Skip calculation if warning or index error is encountered
    
    normalized_squared_error = total_squared_error / total_data_points
    print(f"Normalized Mean Square Error = {normalized_squared_error}")
    return normalized_squared_error


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)



if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
