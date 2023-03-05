import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm as tqdm
from baseDNF import gaussian, euclidean_dist, logistic_sigmoid
from neuron import CUBA_LIF

# DNF
_C_EXC = 0.2
_C_INH = 0.16
_WIDTH = 25
_HEIGHT = 25
_SIGMA_EXC = 0.1 * 50*50 / (_WIDTH*_HEIGHT)
_SIGMA_INH = 2 * 50*50 / (_WIDTH*_HEIGHT)
_SIGMA = 0.1
_SIGMA_SPIKE = 0.06
_GAMMA = 4.5
_GAMMA_MAX = 200
_H = 0.05

# ACTIVITIES
_P1 = (0.3, 0.3)
_P2 = (0.7, 0.7)

# SIMU
_DT = 1e-4
_TAU = 0.1*_DT
_N = 100

# NEURON CUBA LIF
_C = 3e-6     # F
_R = 600      # Ohm
_U0 = 0.1     # V
_I0 = 2.0e-3  # A
_SEUIL = 2e-1 # mV
_TAU = 1.5e-3   # s

_CONNEXION_WEIGHT_EX = 1e-3


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
    centers:list,
    sizes:int,
    sigma:float):
    """Génère une image contenant deux bulles d'activité gaussiennes.

    Args:
        p (list): Liste contenant la position des centres d'activité
        sizes (int): _description_
        sigma (float): _description_
    """

    table = np.zeros((sizes, sizes))
    for i, x in enumerate(np.linspace(start=0, stop=1, num=sizes)):
        for j, y in enumerate(np.linspace(start=0, stop=1, num=sizes)):
            point = (x, y)
            for p in centers:
                table[i,j] += omega(
                    euclidean_dist(point, p),
                    _C_EXC,
                    _C_INH,
                    sigma,
                    sigma)
    table = normalize_table(table)

    return table

def plot_gaussian_activity(table:np.ndarray, title:str="", file_name=None):
    """Plot gaussian activity over the grid.

    Args:
        table (np.ndarray): table to plot.
    """
    plt.imshow(table, cmap="binary", interpolation="nearest")
    plt.title(title)
    plt.show()
    if file_name != None:
        plt.savefig(file_name)

def moving_activities_in_circle(t, T, r, sizes, sigma, N=1):
    """Two gaussian activities moving in circle (diametrally opposit).

    Agrs :
        t (float): current time
        T (float): period of the circular motion
        dt (float): iteration time
        r (float): radius of the circular motion
        N (int): number of activity bubles
    """
    t = t%T
    pos = lambda r, theta : (r*np.cos(theta)/2+0.5, r*np.sin(theta)/2+0.5)

    centers = [pos(r, 2*np.pi*t/T + i*np.pi/N) for i in range(N)] 
    activity = gaussian_activity(centers, sizes, sigma)
    
    return activity

def update_activities(T, r, height, sigma):
    return lambda t : moving_activities_in_circle(t, T, r, height, sigma)

class DNF:
    def __init__(self, input_map, c_exc, c_inh, sigma_exc, sigma_inh, gamma, h, width=45, height=45, tau=1e-3, dt=1e-5): # à vous de déterminer les paramètres nécessaires
        self.dt = dt
        self.tau = tau
        self.t = 0.

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

        self.gif_title = "animation_potential.gif"


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
    
    def normalize_table(self, a:np.ndarray):
        """Normalize an array between 0 and 1

        Args:
            a (np.ndarray): array to normalize
        """
        a -= np.min(a)
        a /= np.max(a)
        return a
        
    def update_neuron(self, position):
        i, j = position
        mean = (logistic_sigmoid(self.potentials)*self.kernel[self.height-i-1:2*self.height-i-1, self.width-j-1:2*self.width-j-1]).mean()
        self.lateral[i, j] = self.potentials[i, j] + self.dt/self.tau * \
                                (-self.potentials[i, j] + \
                                mean + \
                                self.gamma * self.activities[i, j] + \
                                self.h)
        #self.gamma = min(self.gamma*1.2, _GAMMA_MAX)

    def update_map(self, update_activities=None):
        for i in range(self.height):
            for j in range(self.width):
                self.update_neuron((i,j))
        self.potentials = self.normalize_table(self.lateral)

        if update_activities != None:
            self.activities = update_activities(self.t)

        self.t += self.dt

    def image(self):
        return plt.imshow(self.potentials, cmap="binary", animated=True)

    def update_map_N(self, n, update_activities=None, anim=True):
        fig = plt.figure()
        ims = []
        SKIP = 5

        for i in tqdm.tqdm(range(n)):
            
            if anim:
                if i % SKIP == 0:
                    im = self.image()
                    ims.append([im])

            self.update_map(update_activities)

        if anim:
            ani = animation.ArtistAnimation(fig, ims, interval=_DT*1000*SKIP, blit=True)
            ani.save(f"figures/{self.gif_title}")

class SpikingDNF(DNF):
    def __init__(self,
                 C_LIF,
                 R_LIF,
                 U0_LIF,
                 I0_LIF,
                 SEUIL_LIF,
                 TAU_LIF,
                 input_map,
                 c_exc,
                 c_inh,
                 sigma_exc,
                 sigma_inh,
                 gamma,
                 h,
                 width=45,
                 height=45,
                 tau=0.001,
                 dt=0.00001):
        super().__init__(input_map, c_exc, c_inh, sigma_exc, sigma_inh, gamma, h, width, height, tau, dt)
        
        self.K = width*height

        self.neurons = [CUBA_LIF(
            i,
            C_LIF,
            R_LIF,
            0.,
            U0_LIF,
            SEUIL_LIF,
            TAU_LIF,
            random_init=False,
        ) for i in range(self.K)]
        self.I0 = I0_LIF

        self.synaptic_weights = np.zeros((self.K, self.K))
        self.init_weights()
        self.synaptic_bias = np.zeros(self.K)

        self.spikes = np.zeros((height, width))
        self.M = 30
        self.spikes_mov_avg = np.zeros((self.M, height, width))

        self.gif_title = "spikes.gif"

    def init_weights(self):
        for id_neuron_i in range(self.K):
            p_i = ((id_neuron_i//self.width)/self.height, (id_neuron_i%self.width)//self.width)
            for i, x in enumerate(np.linspace(0, 1, self.height)):
                for j, y in enumerate(np.linspace(0, 1, self.width)):
                    id_neuron_j = i*self.width + j
                    p_j = (x,y)
                    self.synaptic_weights[id_neuron_i, id_neuron_j] = omega(
                        euclidean_dist(p_i, p_j),
                        self.c_exc,
                        self.c_inh,
                        self.sigma_exc,
                        self.sigma_inh,
                    ) * _CONNEXION_WEIGHT_EX

    def update_mov_avg(self, mov_avg, a):
        mov_avg[1:] = mov_avg[:-1]
        mov_avg[0] = a
        return mov_avg.mean(axis = 0)
        
    def update_neuron(self, position):
        i, j = position
        id = i*self.width + j
        neuron = self.neurons[id]
        neuron.I_ext = lambda t : self.activities[i, j]*self.I0
        neuron.step(self.dt, self.neurons, self.synaptic_weights, self.synaptic_bias, mode="rk4")
        self.lateral[i, j] = neuron.u
        self.spikes[i, j] = int(neuron.spiked())
        neuron.update_time(self.dt)

    def normalize_table(self, a: np.ndarray):
        return a

    def image(self):
        avg_spikes = self.update_mov_avg(self.spikes_mov_avg, self.spikes)
        return plt.imshow(avg_spikes, cmap="binary", animated=True)


if __name__ == "__main__":
    QUESTION = 2

    if QUESTION == 1:
        activity = gaussian_activity([_P1, _P2], _HEIGHT, _SIGMA)
        plot_gaussian_activity(activity, "Activity")

        dnf = DNF(
            input_map=activity,
            c_exc=_C_EXC_DNF,
            c_inh=_C_INH_DNF,
            sigma_exc=_SIGMA_EXC,
            sigma_inh=_SIGMA_INH,
            gamma=_GAMMA,
            h=_H,
            width=_WIDTH,
            height=_HEIGHT,
            tau=_TAU,
            dt=_DT,
        )

        plot_gaussian_activity(dnf.kernel, "Kernel", f"figures/kernel.png")

        update_act = update_activities(2, 0.7, _HEIGHT, _SIGMA)
        dnf.update_map_N(_N, update_activities=update_act)

        plot_gaussian_activity(dnf.potentials, f"Potentials at time {_N*_DT}s", "figures/potential.png")

    if QUESTION == 2:

        activity = gaussian_activity([_P1, _P2], _HEIGHT, _SIGMA)

        spiking_dnf = SpikingDNF(
            C_LIF=_C,
            R_LIF=_R,
            U0_LIF=_U0,
            I0_LIF=_I0,
            SEUIL_LIF=_SEUIL,
            TAU_LIF=_TAU,
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

        update_act = update_activities(1e-1, 0.7, _HEIGHT, _SIGMA_SPIKE)
        spiking_dnf.update_map_N(_N, update_activities=update_act)
        plot_gaussian_activity(spiking_dnf.potentials, f"Potentials at time {_N*_DT}s", "figures/spiking_dnf_potential.png")