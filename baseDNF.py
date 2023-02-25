import numpy as np

# distance euclidienne dans un espace normalisé [0,1]x[0,1]
def euclidean_dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def euclidean_dist_vect(x:tuple(), y_array:np.ndarray):
    """Compute euclidean distance between a point x and other points stored in y_array

    Args:
        x (tuple): _description_
        y_array (np.ndarray): shape [N, 2]
    """
    return np.sqrt((x[0] - y_array[:,0])**2 + (x[0] - y_array[:,0])**2)

def gaussian(distance, sigma):
    return np.exp(-(distance / sigma) ** 2 / 2)

def logistic_sigmoid(x):
    return 1/(1+np.exp(-(2*x-1)))