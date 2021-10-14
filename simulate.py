import jax
from jax import ops
from jax.ops import index_update
from jax import numpy as np
from matplotlib import pyplot as plt
from jax import jit, vmap, grad, pmap
from jax.experimental.ode import odeint
from jax import random
import numpy as onp
import matplotlib as mpl
import multiprocessing
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from celluloid import Camera
s_ = onp.s_
tqdm = lambda _: _

def make_transparent_color(ntimes, fraction):
  rgba = onp.ones((ntimes, 4))
  alpha = onp.linspace(0, 1, ntimes)[:, np.newaxis]
  color = np.array(mpl.colors.to_rgba(mpl.cm.gist_ncar(fraction)))[np.newaxis, :]
  rgba[:, :] = 1*(1-alpha) + color*alpha
  rgba[:, 3] = alpha[:, 0]
  return rgba

def get_potential(sim, sim_obj):

    dim = sim_obj._dim

    
    def potential(x1, x2):
      """The potential between nodes x1 and x2"""
      dist = np.sqrt(np.sum(np.square(x1[:dim] - x2[:dim])))
     
      min_dist = 1e-2
    #   bounded_dist = dist*(dist > min_dist) + min_dist*(dist <= min_dist)
      bounded_dist = dist + min_dist

      if sim == 'charge':
          charge1 = x1[-2]
          charge2 = x2[-2]

          potential = charge1*charge2/bounded_dist
        
          return potential
      else:
          raise NotImplementedError('No such simulation ' + str(sim))

    return potential

class SimulationDataset(object):

    """Docstring for SimulationDataset. """

    def __init__(self, sim = 'charge', n=5, dim=2,
            dt=0.01, nt=100, extra_potential=None,
            **kwargs):
        """TODO: to be defined.

        :sim: Simulation to run
        :n: number of bodies
        :nt: number of timesteps returned
        :dt: time step (can also set self.times later)
        :dim: dimension of simulation
        :pairwise: custom pairwise potential taking two nodes as arguments
        :extra_potential: function taking a single node, giving a potential
        :kwargs: other kwargs for sim

        """
        self._sim = sim
        self._n = n
        self._dim = dim
        self._kwargs = kwargs
        self.dt = dt
        self.nt = nt
        self.data = None
        self.times = np.linspace(0, self.dt*self.nt, num=self.nt)
        self.G = 1
        self.extra_potential = extra_potential
        self.pairwise = get_potential(sim=sim, sim_obj=self)

    def simulate(self, ns, key=0):
        rng = random.PRNGKey(key)
        vp = jit(vmap(self.pairwise, (None, 0), 0))
        n = self._n
        dim = self._dim 

        sim = self._sim
        params = 2
        total_dim = dim*2+params
        times = self.times
        G = self.G
        if self.extra_potential is not None:
          vex = vmap(self.extra_potential, 0, 0)

        @jit
        def total_potential(xt):
          sum_potential = np.zeros(())
          for i in range(n - 1):
            if sim == 'charge':
                #Only with adjacent nodes
                sum_potential = sum_potential + G*vp(xt[i], xt[[i+1]]).sum()
            else:
                sum_potential = sum_potential + G*vp(xt[i], xt[i+1:]).sum()
          if self.extra_potential is not None:
            sum_potential = sum_potential + vex(xt).sum()
          return sum_potential

        @jit
        def force(xt):
          return -grad(total_potential)(xt)[:, :dim]

        @jit
        def acceleration(xt):
          return force(xt)/xt[:, -1, np.newaxis]

        unpacked_shape = (n, total_dim)
        packed_shape = n*total_dim


        @jit
        def odefunc(y, t):
          dim = self._dim
          y = y.reshape(unpacked_shape)
          a = acceleration(y)
          return np.concatenate(
              [y[:, dim:2*dim],
              a, 0.0*y[:, :params]], axis=1).reshape(packed_shape)

        @partial(jit, backend='cpu')

       
    def make_sim(key):
            x0 = random.normal(key, (n, total_dim))
            if sim in 'charge':
                x0 = index_update(x0, s_[..., -2], np.sign(x0[..., -2])); #charge is 1 or -1

            x_times = odeint(
                odefunc,
                x0.reshape(packed_shape),
                times, mxstep=2000).reshape(-1, *unpacked_shape)

            return x_times

    keys = random.split(rng, ns)
    vmake_sim = jit(vmap(make_sim, 0, 0), backend='cpu')
    # self.data = jax.device_get(vmake_sim(keys))
    # self.data = np.concatenate([jax.device_get(make_sim(key)) for key in keys])
    data = []
    for key in tqdm(keys):
        data.append(make_sim(key))
    self.data = np.array(data)


       

    def plot(self, i, animate=False, plot_size=True, s_size=1):
        #Plots i
        n = self._n
        times = onp.array(self.times)
        x_times = onp.array(self.data[i])
        sim = self._sim
        masses = x_times[:, :, -1]
        if not animate:
            if sim in 'charge':
                rgba = make_transparent_color(len(times), 0)
                for i in range(0, len(times), len(times)//10):
                    ctimes = x_times[i]
                    plt.plot(ctimes[:, 0], ctimes[:, 1], color=rgba[i])
                plt.xlim(-5, 20)
                plt.ylim(-20, 5)
            else:
                for j in range(n):
                  rgba = make_transparent_color(len(times), j/n)
                  if plot_size:
                    plt.scatter(x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=3*masses[:, j]*s_size)
                  else:
                    plt.scatter(x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=s_size)
        else:
            if sim in 'charge': raise NotImplementedError
            fig = plt.figure()
            camera = Camera(fig)
            d_idx = 20
            for t_idx in range(d_idx, len(times), d_idx):
                start = max([0, t_idx-300])
                ctimes = times[start:t_idx]
                cx_times = x_times[start:t_idx]
                for j in range(n):
                  rgba = make_transparent_color(len(ctimes), j/n)
                  if plot_size:
                    plt.scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=3*masses[:, j])
                  else:
                    plt.scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=s_size)
#                 plt.xlim(-10, 10)
#                 plt.ylim(-10, 10)
                camera.snap()
            from IPython.display import HTML
            return HTML(camera.animate().to_jshtml())
