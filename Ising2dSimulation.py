from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

class Ising2d(object):
    
    def __init__(self, K_x, K_y, h, L_x, L_y):
        self.K_x = K_x
        self.K_y = K_y
        self.h = h
        self.L_x = L_x
        self.L_y = L_y
        self.size = L_x*L_y
        
        self.state = np.random.randint(2, size=(L_y, L_x), dtype='bool')
        spins = 2*self.state-1
        self.mag = np.sum(spins)
        
        staggered = np.ones((L_y, L_x))
        staggered[::2, ::2] = -1
        staggered[1::2, 1::2] = -1
        self.stag_mag = np.sum(staggered * spins) #antiferromagnetic order
        
        
        self.N_iter = 0
    
    def calc_energy(self):
        state = self.state
        K_x = self.K_x
        K_y = self.K_y
        spins = 2*state-1
        E_bulk = -np.sum(K_x * spins[:, :-1] * spins[:, 1:])\
                 - np.sum(K_y * spins[:-1, :] * spins[1:, :]) - h * np.sum(spins)
        E_BC = - sum(K_x * spins[:, -1] * spins[:, 0]) \
               - sum(K_y * spins[-1, :] * spins[0, :])
        E = E_bulk + E_BC
        return E
    
    def calc_mag(self):
        self.mag = np.sum(2*self.state - 1)
    
    def calc_stag_mag(self):
        spins = 2*self.state-1
        
        staggered = np.ones((L_y, L_x))
        staggered[::2, ::2] = -1
        staggered[1::2, 1::2] = -1
        self.stag_mag = np.sum(staggered * spins) #antiferromagnetic order
    
    def reset(self):
        self.state = np.random.randint(2, size=(L_y, L_x), dtype='bool')
        spins = 2*self.state-1
        self.mag = np.sum(spins)
        
        staggered = np.ones((L_y, L_x))
        staggered[::2, ::2] = -1
        staggered[1::2, 1::2] = -1
        self.stag_mag = np.sum(staggered * spins) #antiferromagnetic order
        
        self.N_iter = 0

    
    def step(self):
        L_x = self.L_x
        L_y = self.L_y
        K_x = self.K_x
        K_y = self.K_y
        col = np.random.randint(L_x)
        row = np.random.randint(L_y)
        state = self.state
        spin_state = 2*state - 1
        delta_s = -2*spin_state[row, col]
        dE_h = -h*delta_s
        if col == 0:
            dE_x = -K_x * (spin_state[row, -1] + spin_state[row, 1]) * delta_s
        elif col == L_x-1:
            dE_x = -K_x * (spin_state[row, col-1] + spin_state[row, 0]) * delta_s
        else:
            dE_x = -K_x * (spin_state[row, col-1] + spin_state[row, col+1])\
                   * delta_s
        
        if row == 0:
            dE_y = -K_y * (spin_state[-1, col] + spin_state[1, col]) * delta_s
        elif row == L_y-1:
            dE_y = -K_y * (spin_state[row-1, col] + spin_state[0, col]) * delta_s
        else:
            dE_y = -K_y * (spin_state[row-1, col] + spin_state[row + 1, col])\
                   * delta_s
            
        dE = dE_x + dE_y + dE_h
        
        if dE < 0:
            self.state[row, col] = not state[row, col]
        else:
            if np.random.rand() <= np.exp(-dE):
                self.state[row, col] = not state[row, col]
        
        self.N_iter += 1
        self.calc_mag()
        self.calc_stag_mag()

if __name__=="__main__":
    import matplotlib.animation as animation
    #------------------------------------------------------------
    # set up initial state and global variables
    K_x = np.arcsinh(1) / 2
    K_y = np.arcsinh(1) / 2
    h = 0
    L_x = 100
    L_y = 100
    eq_steps = 100000
    sim = Ising2d(K_x, K_y, h, L_x, L_y)

    # this will speed up equilibration by not animating first
    # equilibrating steps
    print("Please wait running for {} steps to equilibrate".format(eq_steps))
    for x in range(eq_steps):
        sim.step()

    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-0.5, L_x - 0.5), 
                         ylim=(-0.5, L_y - 0.5 + 0.1 * L_y))
    ax.grid()

    line, = ax.plot([], [], '.')
    time_text = ax.text(0.02, 0.95, '', 
                        transform=ax.transAxes, color='k', fontsize=10)

    def init():
        """initialize animation"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i, skip=100):
        """perform animation step"""
        global sim
        for x in range(skip):
            sim.step()
        
        size = sim.size
        mag = sim.mag / size
        stag_mag = sim.stag_mag / size
        energy = sim.calc_energy() / size
        
        row_loc, col_loc = np.where(sim.state)
        line.set_data(col_loc, row_loc)
        time_text.set_text((r'steps = {:}, $E$ = {:.2f},' + 
                            r'$M$ = {:.3f}, $M_s$ = {:.3f}')\
                            .format(sim.N_iter, energy, mag, stag_mag))
        return line, time_text

    # choose the interval based on dt and the time to animate one step
    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    # interval = 1000*dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=300,
                                  interval=0, blit=True, init_func=init)

    plt.show()
