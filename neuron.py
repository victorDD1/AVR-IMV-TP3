import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

class LIF():
    def __init__(
        self,
        id:int,
        C:float,
        R:float,
        I_ext,
        u0:float,
        seuil:float,
        random_init:bool=True,
        ):
        """
        Neuron LIF.

        Args:
            - id : index
            - C : capacity (F)
            - R : Resistance (Ohm)
            - I_ext : External current, float or function
            - u0 : Default potential
            - seuil : spike threshold
        
        """
        self.id = id
        self.C = C
        self.R = R
        self.seuil = seuil
        self.u0 = u0

        if isinstance(I_ext, float):
            self.I_ext = lambda t : I_ext # For constant external current
        elif type(I_ext) == type(lambda t: t):
            self.I_ext = I_ext
        else:
            print("Warning. I_ext is not a float nor a function.")

        ### Initialisation (t = 0)
        # Time
        self.time = 0
        # Instant time of the spikes
        self.spike_time = [-1.]
        # Random initial activation
        if random_init:
            self.u = np.random.uniform(-u0, seuil/2)
        else:
            self.u = -u0      

    def du_t(self, u, u_ext, t):
        """
        Derivative of the potential wrt time.
        """
        return (-1/self.R*u + u_ext + self.I_ext(t))/self.C

    def is_spike(self):
        """
        True if there is a spike at the current iteration.
        """
        return self.u > self.seuil

    def spiked(self):
        """
        True if last time stemp is spike.
        """
        return self.spike_time[-1] == self.time

    def u_ext_connexion(self, neuron_list, synaptic_weights, synaptic_bias):
        """
        Compute potential of external connexions.
        """
        u_ext = 0

        for l, neuron in enumerate(neuron_list): # we assume that synaptic_weights[id, id] = 0
            if synaptic_weights[self.id, l] != 0:
                dirac = int(neuron.spiked())
                u_ext += dirac * synaptic_weights[self.id, l]
        u_ext += synaptic_bias[self.id]
        return u_ext

    def update_u(self, h, u_ext, mode="euler"):
        """
        Update potential of the neuron according to the model equations.
        Integration mode euler or rk4.
        """
        if mode == "euler":
            self.u += + h*self.du_t(self.u, u_ext, self.time + h)

        if mode == "rk4":
            k1 = self.du_t(self.u, u_ext, self.time)
            k2 = self.du_t(self.u + k1*h/2, u_ext, self.time + h/2)
            k3 = self.du_t(self.u + k2*h/2, u_ext, self.time + h/2)
            k4 = self.du_t(self.u + k3*h, u_ext, self.time + h)

            self.u += h*1/6*(k1 + 2*k2 + 2*k3 + k4)

    def step(self, h, neuron_list, synaptic_weights, synaptic_bias, mode="euler"):
        """
        Udpate the state of the neuron for one time stemp.
        Args:
            - h : step duration
            - neuron_list : list of the connected neurons
            - synaptic_weights[i, j]: weight of the connection between neuron j (in) and neuron i (out)
            - synaptic_bias[i]: weight of the connection of neuron i (out)
            - mode : integration mode euler of rk4
        """
        u_ext = self.u_ext_connexion(neuron_list, synaptic_weights, synaptic_bias)
        self.update_u(h, u_ext, mode)
        
        if self.is_spike():
            self.spike_time.append(self.time)
            self.u = -self.u0

    def update_time(self, h):
        """
        Update time of the current neuron.
        """
        self.time += h

class CUBA_LIF(LIF):
    def __init__(
        self,
        id: int,
        C: float,
        R: float,
        I_ext,
        u0: float,
        seuil: float,
        tau: float,
        random_init: bool = True):
        super().__init__(id, C, R, I_ext, u0, seuil, random_init)

        self.tau = tau

    def u_ext_connexion(self, neuron_list, synaptic_weights, synaptic_bias):
        u_ext = 0

        for l, neuron in enumerate(neuron_list): # we assume that synaptic_weights[id, id] = 0
            if synaptic_weights[self.id, l] != 0:
                t_j = abs(neuron.spike_time[-1]) # self.spike_time is initialized with [-1] => H(t - t_j) = 0 when abs(t_j)=1
                x_j = np.exp(-self.time/self.tau) * np.exp(t_j/self.tau) * int(self.time - t_j > 0)

                u_ext += x_j * synaptic_weights[self.id, l]
        u_ext += synaptic_bias[self.id]
        return u_ext


class LIF_enrichi(LIF):
    def __init__(self,
                 id: int,
                 C: float,
                 R: float,
                 I_ext,
                 u0: float,
                 seuil: float,
                 t_ltp: float = 0.,
                 alpha_p: float=1.,
                 alpha_n:float=1.,
                 beta_p:float=0.,
                 beta_n:float=0.,
                 t_refract=0.,
                 t_inhib=0.,
                 neuron_to_inhib=None,
                 random_init: bool = True):
        super().__init__(id, C, R, I_ext, u0, seuil, random_init)

        self.t_ltp = t_ltp

        self.alpha_p = alpha_p
        self.alpha_n = alpha_n

        self.beta_p = beta_p
        self.beta_n = beta_n

        self.t_refract = t_refract
        self.t_inhib = t_inhib
        self.last_t_inhib = 0.

        self.inhibited = False
        self.neuron_to_inhib = neuron_to_inhib # id of the neurons to inhibit, if empty, all neurons connected

    def set_I_ext(self, I_ext):
        """Set I_ext.

        Args:
            I_ext (function): External current.
        """
        if isinstance(I_ext, float):
            self.I_ext = lambda t : I_ext # For constant external current
        elif type(I_ext) == type(lambda t: t):
            self.I_ext = I_ext
        else:
            print("Warning. I_ext is not a float nor a function.")
    
    def spiked(self, delay, eps=5e-6):
        spiked = self.time - delay - eps/2 < self.spike_time[-1] < self.time - delay + eps/2
        return spiked

    def inhibit(self):
        if self.t_inhib > 0 and not(self.inhibited):
            self.inhibited = True
            self.last_t_inhib = self.time

    def try_uninhibit(self):
        if self.inhibited and self.last_t_inhib + self.t_inhib < self.time:
            self.inhibited = False

    def u_ext_connexion(self, neuron_list, synaptic_weights, synaptic_bias, delays):
        """
        Compute potential of external connexions.
        """
        u_ext = 0

        for l, neuron in enumerate(neuron_list):
            if l != self.id:
                # Spike et non inhibité
                dirac = int(neuron.spiked(delays[self.id, l]))
                dirac *= int(not(neuron.inhibited)) # 0 si inhibé sinon 1
                u_ext += dirac * synaptic_weights[self.id, l]
        u_ext += synaptic_bias[self.id]

        return u_ext

    def update_weights(self, neuron_list, synaptic_weights, learnable_weights, delays):
        """Update weights when neuron spike according to STDP update rule.
        """
        w_i_ = synaptic_weights[self.id, :]
        w_max, w_min = w_i_.max(), w_i_.min()

        for j, (neuron, w_ij, delay) in enumerate(zip(neuron_list, synaptic_weights[self.id, :], delays[self.id, :])):
            if learnable_weights[self.id, j] and len(neuron.spike_time) > 1:
                dt = neuron.spike_time[-1] - self.time + delay
                if -self.t_ltp < dt < 0:
                    w_ij += self.alpha_p*np.exp(-self.beta_p * (w_ij-w_min)/(w_max-w_min))
                else:
                    w_ij += self.alpha_n*np.exp(-self.beta_n * (w_max-w_ij)/(w_max-w_min))
                w_i_[j] = w_ij
        
        delta = synaptic_weights[self.id, :] - w_i_
        synaptic_weights[self.id, :] = w_i_

        return synaptic_weights

    def step(self, h, neuron_list, synaptic_weights, synaptic_bias, learnable_weights, delays, mode="euler"):
        """
        Udpate the state of the neuron for one time stemp.
        Args:
            - h : step duration
            - neuron_list : list of the connected neurons
            - synaptic_weights[i, j]: weight of the connection between neuron j (in) and neuron i (out)
            - synaptic_bias[i]: weight of the connection of neuron i (out)
            - delays[i, j]: delay of connexion between i and j
            - mode : integration mode euler of rk4
        """
        u_ext = self.u_ext_connexion(neuron_list, synaptic_weights, synaptic_bias, delays)
        self.update_u(h, u_ext, mode)
        
        if self.is_spike():
            self.spike_time.append(self.time)
            self.u = -self.u0

            # Learn
            synaptic_weights = self.update_weights(neuron_list, synaptic_weights, learnable_weights, delays)
            
            # Inhibit
            for id in self.neuron_to_inhib:
                if id != self.id:
                    neuron_list[id].inhibit()

        # Période réfractaire
        if self.t_refract > 0:
            if self.time - self.spike_time[-1] < self.t_refract:
                self.u = -self.u0

        self.try_uninhibit()

        return synaptic_weights


class Simulation():
    def __init__(self, neuron_list, synaptic_weights, synaptic_bias, learnable_weights, delays, h, N, mode='euler'):
        
        if type(neuron_list) != list:
            self.neuron_list = [neuron_list]
        else:
            self.neuron_list = neuron_list
        
        self.k = len(self.neuron_list) # number of neurons

        self.synaptic_weights = synaptic_weights # [k,k] matrix --- w_ij weight from j to i
        self.synaptic_bias = synaptic_bias # [k] vector --- bias of neuron i
        self.learnable_weights = learnable_weights # [k,k] matrix --- 1 if weight is learnable else 0
        self.delays = delays # [k,k] matrix --- delay d_ij from j to i

        self.h = h
        self.N = N
        self.g = 10
        self.n = self.N//self.g
        
        self.i = 0 # current step
        self.u = np.zeros((self.n, self.k))
        self.I_ext = np.zeros((self.n, self.k))
        self.spikes = np.zeros((self.n, self.k))
        self.inhibited = np.zeros((self.n, self.k))
        self.time = np.arange(0, self.n*h, h)

        self.mode = mode

    def reset(self, N):
        self.i = 0
        self.N = N
        self.n = self.N//self.g
        self.u = np.zeros((self.n, self.k))
        self.I_ext = np.zeros((self.n, self.k))
        self.spikes = np.zeros((self.n, self.k))
        self.inhibited = np.zeros((self.n, self.k))
        self.time = np.arange(0, self.n*self.h, self.h)


    def step(self):
        # Update neurons
        # Permutation so that the connexions are not deterministic (not always i that inhibit j before j does)
        for j in np.random.permutation(range(len(self.neuron_list))):
            neuron = self.neuron_list[j]
            self.synaptic_weights = neuron.step(self.h, self.neuron_list, self.synaptic_weights, self.synaptic_bias, self.learnable_weights, self.delays, self.mode)
            
            if self.i % 10 == 0:
                self.u[self.i//self.g, j] = neuron.u
                self.spikes[self.i//self.g, j] = int(neuron.spiked(0.))
                self.inhibited[self.i//self.g, j] = int(neuron.inhibited)
                self.I_ext[self.i//self.g, j] = neuron.I_ext(neuron.time)

        # Synchrone time update
        for neuron in self.neuron_list:
            neuron.update_time(self.h)
        
        self.i+=1

    def run(self):
        for _ in tqdm.tqdm(range(self.N)):
            self.step()
        
    def plot(self):
        plt.plot(self.time, self.u)
        plt.xlabel("Time (s)")
        plt.ylabel("U (V)")
        plt.show()

    def plot_spike(self, filename=""):
        fig = plt.figure(figsize=(8, 16))
        axes = []
        for i in range(len(self.neuron_list)):
            N_spikes = int(self.spikes[:, i].sum())

            ax = plt.subplot(self.k, 1, i+1)
            ax.plot(self.time*1e4, self.spikes[:, i])
            axes.append(ax)

            ax.set_ylabel(N_spikes)

            ax.set_yticklabels([])
            ax.set_yticks([])
            if i+1 != len(self.neuron_list):
                ax.set_xticks([])

        axes[0].get_shared_x_axes().join(axes[0], *axes[1:])


        plt.xlabel("Time (ms)")
        plt.suptitle("Spikes")

        if filename != "":
            plt.savefig(f"figures/{filename}")

        plt.show()

    def plot_I_ext(self, filename=""):
        fig = plt.figure(figsize=(8, 16))
        axes = []
        for i in range(len(self.neuron_list)):

            ax = plt.subplot(self.k, 1, i+1)
            ax.plot(self.time*1e4, self.I_ext[:, i])
            axes.append(ax)

            ax.set_yticklabels([])
            ax.set_yticks([])
            if i+1 != len(self.neuron_list):
                ax.set_xticks([])

        axes[0].get_shared_x_axes().join(axes[0], *axes[1:])

        plt.xlabel("Time (ms)")
        plt.suptitle("I_ext(t)")

        if filename != "":
            plt.savefig(f"figures/{filename}")

        plt.show()

    def plot_inhibited(self, filename=""):
        fig = plt.figure(figsize=(8, 16))
        axes = []
        for i in range(len(self.neuron_list)):

            ax = plt.subplot(self.k, 1, i+1)
            ax.plot(self.time*1e4, self.spikes[:, i], '-r')
            ax.plot(self.time*1e4, self.inhibited[:, i])
            axes.append(ax)


            ax.set_yticklabels([])
            ax.set_yticks([])
            if i+1 != len(self.neuron_list):
                ax.set_xticks([])

        axes[0].get_shared_x_axes().join(axes[0], *axes[1:])


        plt.xlabel("Time (ms)")
        plt.suptitle("Inhibited")

        if filename != "":
            plt.savefig(f"figures/{filename}")

        plt.show()

    def plot3d(self):
        
        fig = plt.figure()
        ax = plt.axes(projection = '3d')

        for k in range(self.k):
            ax.plot(xs=self.time, ys=self.u[:, k], zs=k)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('U (V)')
        ax.set_zlabel('Neurons')

        # On the y axis let's only label the discrete values that we have data for.
        ax.set_zticks(list(range(self.k)))
        ax.set_ylim(1.1 * np.min(self.u), 1.1 * np.max(self.u))

        ax.view_init(45, 0, 90)

        plt.show()