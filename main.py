import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm as tqdm
from baseDNF import gaussian, euclidean_dist, logistic_sigmoid

_C_EXC = 0.2
_C_INH = 0.16
_WIDTH = 45
_HEIGHT = 45
_SIGMA_EXC = 0.1 * 50*50 / (_WIDTH*_HEIGHT)
_SIGMA_INH = 2 * 50*50 / (_WIDTH*_HEIGHT)
_SIGMA = 0.1
_GAMMA = 4.5
_H = 0.05
_DT = 1e-0
_TAU = 0.1*_DT

_P1 = (0.4, 0.4)
_P2 = (0.6, 0.6)

def omega(
    d:float,
    c_exc:float,
    c_inh:float,
    sigma_exc:float,
    sigma_inh:float,):
    """Valeur du noyau d'intéraction (en différence de gaussiennes).

    Args:
        d (float): _description_
        c_exc (float): _description_
        c_inh (float): _description_
        sigma_exc (float): _description_
        sigma_exc (float): _description_
    """
    return c_exc*gaussian(d, sigma_exc) - c_inh*gaussian(d, sigma_inh)

def normalize_table(a:np.ndarray):
    """Normalize an array between 0 and 1

    Args:
        a (np.ndarray): array to normalize
    """
    a -= np.min(a)
    a /= np.max(a)
    return a

def gaussian_activity(
    p:tuple(),
    q:tuple(),
    sizes:int,
    sigma:float):
    """Génère une image contenant deux bulles d'activité gaussiennes.

    Args:
        p (tuple): _description_
        q (tuple): _description_
        sizes (int): _description_
        sigma (float): _description_
    """

    table = np.zeros((sizes, sizes))
    for i, x in enumerate(np.linspace(start=0, stop=1, num=sizes)):
        for j, y in enumerate(np.linspace(start=0, stop=1, num=sizes)):
            point = (x, y)
            table[i,j] = omega(
                euclidean_dist(point, p),
                _C_EXC,
                _C_INH,
                sigma,
                sigma) + \
                    omega(
                euclidean_dist(point, q),
                _C_EXC,
                _C_INH,
                sigma,
                sigma)
    
    table = normalize_table(table)

    return table

def plot_gaussian_activity(table:np.ndarray, title:str=""):
    """Plot gaussian activity over the grid.

    Args:
        table (np.ndarray): table to plot.
    """
    plt.imshow(table, cmap="binary", interpolation="nearest")
    plt.title(title)
    plt.show()

class DNF:
    def __init__(self, input_map, c_exc, c_inh, sigma_exc, sigma_inh, gamma, h, width=45, height=45, tau=1e-3, dt=1e-5): # à vous de déterminer les paramètres nécessaires
        self.dt = dt
        self.tau = tau

        self.c_exc = c_exc
        self.c_inh = c_inh
        self.sigma_exc = sigma_exc
        self.sigma_inh = sigma_inh
        self.gamma = gamma
        self.h = h
        
        self.width = width
        self.height = height
        self.N = self.height * self.width
        self.potentials = np.zeros([width, height], dtype=float)
        self.activities = input_map
        self.lateral = np.zeros([width, height], dtype=float)
        # noyau étendu pour permettre les convolutions du style :
        # signal.fftconvolve(self.activities, self.kernel, mode='same')
        self.kernel = np.zeros([width*2-1, height*2-1], dtype=float)
        self.init_kernel()


        # N.B.: le neurone d'indice [i,j] correspond à un point de coordonnées [i/(width-1),j/(height-1)] dans l'espace normalisé

    def init_kernel(self):
        for i, x in enumerate(np.linspace(-1, 1, self.height*2-1)):
            for j, y in enumerate(np.linspace(-1, 1, self.width*2-1)):
                p = (x,y)
                self.kernel[i, j] = omega(
                    euclidean_dist(p, (0,0)),
                    self.c_exc,
                    self.c_inh,
                    self.sigma_exc,
                    self.sigma_inh,
                )
        self.kernel /= self.kernel.max() # Maximum = 1
        
    def update_neuron(self, position):
        i, j = position
        mean = (logistic_sigmoid(self.potentials)*self.kernel[self.height-i-1:2*self.height-i-1, self.width-j-1:2*self.width-j-1]).mean()
        self.lateral[i, j] = self.potentials[i, j] + self.dt/self.tau * \
                                (-self.potentials[i, j] + \
                                mean + \
                                self.gamma * self.activities[i, j] + \
                                self.h)

    def update_map(self):
        for i in range(self.height):
            for j in range(self.width):
                self.update_neuron((i,j))
        self.potentials = self.lateral
        self.potentials /= self.potentials.max()

    def update_map_N(self, n, anim=True):
        if anim:
            fig, ax = plt.subplots()
            im = ax.imshow(self.potentials, vmax=1, cmap="binary", interpolation="nearest", animated=True)

            def init():
                im.set_array(self.potentials)
                return im
            
            def update_fig(*args):
                self.update_map()
                im.set_array(self.potentials)
                return im,

            ani = animation.FuncAnimation(fig, update_fig, init_func=init, interval=5, blit=True, )
            plt.show()
        
        else:
            for _ in tqdm.tqdm(range(n)):
                self.update_map()



if __name__ == "__main__":
    activity = gaussian_activity(_P1, _P2, _HEIGHT, _SIGMA)
    plot_gaussian_activity(activity, "Activity")

    dnf = DNF(
        input_map=activity,
        c_exc=_C_EXC,
        c_inh=_C_INH,
        sigma_exc=_SIGMA_EXC,
        sigma_inh=_SIGMA_INH,
        gamma=_GAMMA,
        h=_H,
        width=_WIDTH,
        height=_HEIGHT,
        tau=_TAU,
        dt=_DT,
    )

    plot_gaussian_activity(dnf.kernel, "Kernel")

    dnf.update_map_N(1000, anim=False)

    plot_gaussian_activity(dnf.potentials)
