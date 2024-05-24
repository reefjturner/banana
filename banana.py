'''
Banana
=====
Banana is a python package aimed to be a user friendly interface to simulate 
axion direct detection experiments with generalised velocity distributions.

Classes
-----------
rand  -  for generating random variables from arbitrary distributions

dark_matter  -  for all things dark matter.

axion  -  for all things axion.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.fft import fftfreq
from scipy.fft import fft
import scipy as sp

class rand:
    def gen_rand_phase(N = 1):
        '''
        Generates an array of random phases from [0, 2pi)
        
        Returns
        -------
        rand_phase : ndarray
        An array of 'N' random phases from [0, 2pi)
        '''
        return 2 * np.pi * np.random.rand(N)
    
    
class dark_matter:
    def get_rand_bulkSHM(v0, v_obs, N):
        mu = np.array([0,v_obs,0])
        s2 = v0**2 / 2
        cov = np.array([[s2,0,0],[0,s2,0],[0,0,s2]])
        v = np.random.multivariate_normal(mu, cov, N)
        return v 
        # return v + np.array([0,v_obs,0])
    

    def get_rand_bulkSHM2D(v0, v_obs, N):
        mu = np.array([0,v_obs])
        s2 = v0**2 / 2
        cov = np.array([[s2,0],[0,s2]])
        v = np.random.multivariate_normal(mu, cov, N)
        return v
    

    def get_rand_streamSHM2D(v0_stream, v_obs_stream, N):
        # v_obs needs to be 2D arrray
        s2 = v0_stream**2 / 2
        cov = np.array([[s2,0],[0,s2]])
        v = np.random.multivariate_normal(v_obs_stream, cov, N)
        return v
    

    def get_rand_streamSHM(v0_stream, v_obs_stream, N):
        # v_obs needs to be 3D arrray
        s2 = v0_stream**2 / 2
        cov = np.array([[s2,0,0],[0,s2,0], [0,0,s2]])
        v = np.random.multivariate_normal(v_obs_stream, cov, N)
        return v


    def get_completely_rand_streamSHM(v0_stream, v_obs_stream, N):
        # v_obs needs to be 3D arrray
        s2 = v0_stream**2 / 2
        cov = np.array([[s2,0,0],[0,s2,0], [0,0,s2]])
        v = np.random.multivariate_normal(v_obs_stream, cov, N)
        return v


class axion:   
    def gen_axion_field_2D(X, vel, axion_mass = 2 * np.pi, axion_density = 1, amplitude = 1):
        A = amplitude
        ma = axion_mass
        p_dm = axion_density
        t = X[0]
        x = X[1]
        y = X[2]
        phases = rand.gen_rand_phase(len(vel))
        axions = np.empty([len(vel), t.shape[0], t.shape[1], t.shape[2]])
        for i in range(0, vel.shape[0]):
            axions[i] = A*np.cos(ma * (1 + (vel[i][0]**2 + vel[i][1]**2)/2)*t + vel[i][0]*x + vel[i][1]*y + phases[i])
        axion_field = np.sum(axions, axis=0)
        return axion_field
    

    def gen_axion_field(X, vel, axion_mass = 2 * np.pi, axion_density = 1):
        ma = axion_mass
        N = vel.shape[0]
        p_dm = axion_density
        A = 1
        # A = np.sqrt(2*p_dm / N) / ma
        t = X[0]
        x = X[1]
        y = X[2]
        z = X[3]
        phases = rand.gen_rand_phase(len(vel))
        axions = np.empty([len(vel), t.shape[0], t.shape[1], t.shape[2], t.shape[3]])
        for i in range(0, vel.shape[0]):
            axions[i] = A*np.cos(ma*(1 + (vel[i][0]**2 + vel[i][1]**2 + vel[i][2]**2)/2)*t + vel[i][0]*x + vel[i][1]*y + vel[i][2]*z + phases[i])
        axion_field = np.sum(axions, axis=0)
        return axion_field
    

    def plot_axion_field_2D(X, axion_field, t):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X[1][t], X[2][t], axion_field[t], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')      
        plt.show() 


    def brute_force_fix_axion_field(t, x1, x2, vel, no_axions, ma= 2*np.pi, A = 1):
        axions = np.empty([no_axions, t.shape[-1]])
        for i in range(0,no_axions):
            phase = rand.gen_rand_phase(1)
            axion_x1 = np.cos(ma * (1 + (vel[i][0]**2 + vel[i][1]**2 + vel[i][2]**2)/2)*t + vel[i][0]*x1[0] + vel[i][1]*x1[1] + vel[i][2]*x1[2] + phase) 
            axion_x2 = np.cos(ma * (1 + (vel[i][0]**2 + vel[i][1]**2 + vel[i][2]**2)/2)*t + vel[i][0]*x2[0] + vel[i][1]*x2[1] + vel[i][2]*x2[2] + phase) 
            axions[i] = axion_x1 * axion_x2
        return np.sum(axions, axis=0)
    

    def animate_axion_field2D(X, axion_field, sample_rate, period, coherence_length, amplitude = 50, filename='simulation.mp4'):
        inter = 1000/sample_rate
        fig = plt.figure()
        ax = plt.axes(projection='3d', aspect='equal')
        ax.set_axis_off()
        print('starting animation')
        def animate(i):
            ax.clear()
            ax.plot_surface(X[1][i], X[2][i], axion_field[i], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
            ax.set_xlim([-10*coherence_length, 10*coherence_length])
            ax.set_ylim([-10*coherence_length, 10*coherence_length])
            ax.set_zlim([-amplitude,amplitude])
            print(f'Frame {i}')
        ani = FuncAnimation(fig, animate, frames=int(sample_rate*period),
                    interval=inter, repeat=True)
        ffmpeg_writer = animation.FFMpegWriter(fps=sample_rate)
        ani.save('simulation2.mp4', writer=ffmpeg_writer)
  


    def get_velocity_field2D(x, y, axion_field, t=0):
        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        return np.gradient(axion_field[t], dx)[0], np.gradient(axion_field[t], dy)[1]


    def plot_velocity_field2D(x,y, axion_field, broken_streamlines=False, c_length=2 * np.pi):
        plt.axes(aspect='equal')
        plt.xlabel('$x$ coordinate ($\lambda$)')
        plt.ylabel('$y$ coordinate ($\lambda$)')
        grad_x, grad_y = axion.get_velocity_field2D(x, y, axion_field)
        N = np.sqrt(grad_x**2 + grad_y**2)
        plt.streamplot(x/c_length, y/c_length, grad_x, grad_y,  color=N, cmap='viridis', broken_streamlines=broken_streamlines, linewidth=2/3, density = 2, arrowstyle='-')
        plt.show()


    def get_theta_field2D(x, y, axion_field):
        grad_x, grad_y  = axion.get_velocity_field2D(x, y, axion_field)
        return np.arctan2(grad_y, grad_x)
    

    def plot_theta_field2D(x, y, axion_field, c_invert=False):
        plt.axes(aspect='equal')
        theta = axion.get_theta_field2D(x, y, axion_field)
        if c_invert:
            plt.pcolormesh(x, y, theta, cmap='twilight_shifted')
            plt.show()
        else:
            plt.pcolormesh(x, y, theta, cmap='twilight')
            plt.show() 

    def gen_PSD(t, axion_field):
        '''
        Converts time series axion field signal into PSD.

        Parameters
        ----------
        t : ndarray
            An array of time values, from [0, T), with T being the period of the signal. 
        axion_field : ndarray
            An array of scalar amplitudes of the axion field as a function of time.
        
        Returns
        -------
        psd_axion_field : ndarray
            An array of the PSD of the axion signal in terms of angular frequency.
        '''
        return np.abs(fft(axion_field)[0:t.shape[-1]//2])**2
    

    def get_fourier_freq(t):
        '''
        Returns an array of corresponding angular frequencies from time interval.

        Parameters
        ----------
        t : ndarray
            An array of time values, from [0, T), with T being the period of the signal. 

        Returns
        -------
        fft_freq : ndarray
            An array of corresponding fourier angular frequencies.
        '''
        return fftfreq(t.shape[-1],np.diff(t)[0])[0:int(t.shape[-1])//2] * (2 * np.pi)
    

    def montee_carlo(t, X, v0, v_obs,n_axions, N_simulations):
        '''
        Creates 'N' montee carlo simulations, and averages the result.

        Parameters
        ----------
        t : ndarray
            An array of time values, from [0, T), with T being the period of the signal. 
        v : ndarray
            An array of velocities which span the possible set of velocities for the distribution.
        vel_dist : ndarray
            An array generated by putting the velocity array 'v' into the velocity distribution.

        Returns
        -------
        montee_axion : ndarray
            Array of average signal over 'N' montee carlo simulations in terms of angular frequency.
        '''
        omega = axion.get_fourier_freq(t)
        multi_axion = np.empty([len(t)])
        final_psd_axion = np.empty([N_simulations, omega.shape[-1]])
        for i in range(0, N_simulations):
            vel = dark_matter.get_rand_bulkSHM(v0, v_obs, n_axions)
            axion_field = axion.gen_axion_field(X, vel, axion_mass = 2 * np.pi, axion_density = 1)
            for j in range(0, len(t)):
                multi_axion[j] = axion_field[j][0][0][0]
            final_psd_axion[i] = axion.gen_PSD(t, multi_axion)
        return np.sum(final_psd_axion, axis = 0)
    

    def multi_montee_carlo(t, X, v0, v_obs,n_axions, N_simulations):
        '''
        Creates 'N' montee carlo simulations, and averages the result.

        Parameters
        ----------
        t : ndarray
            An array of time values, from [0, T), with T being the period of the signal. 
        v : ndarray
            An array of velocities which span the possible set of velocities for the distribution.
        vel_dist : ndarray
            An array generated by putting the velocity array 'v' into the velocity distribution.

        Returns
        -------
        montee_axion : ndarray
            Array of average signal over 'N' montee carlo simulations in terms of angular frequency.
        '''
        omega = axion.get_fourier_freq(t)
        multi_axion = np.empty([len(t)])
        final_psd_axion = np.empty([N_simulations, omega.shape[-1]])
        for i in range(0, N_simulations):
            vel = dark_matter.get_rand_bulkSHM(v0, v_obs, n_axions)
            axion_field = axion.gen_axion_field(X, vel, axion_mass = 2 * np.pi, axion_density = 1)
            # for j in range(0, len(t)):
            #     multi_axion[j] = axion_field[j][0][0][0] * axion_field[j][1][1][1]
            multi_axion = axion_field[:,0,0,0] * axion_field[:,1,1,1] 
            final_psd_axion[i] = axion.gen_PSD(t, multi_axion)
            print('\r' + f'Simulation {i+1} of {N_simulations} complete', end='')
        return np.sum(final_psd_axion, axis = 0)


    def multi_montee_carlo_stream(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations):
        '''
        Creates 'N' montee carlo simulations, and averages the result.

        Parameters
        ----------
        t : ndarray
            An array of time values, from [0, T), with T being the period of the signal. 
        v : ndarray
            An array of velocities which span the possible set of velocities for the distribution.
        vel_dist : ndarray
            An array generated by putting the velocity array 'v' into the velocity distribution.

        Returns
        -------
        montee_axion : ndarray
            Array of average signal over 'N' montee carlo simulations in terms of angular frequency.
        '''
        omega = axion.get_fourier_freq(t)
        multi_axion = np.empty([len(t)])
        final_psd_axion = np.empty([N_simulations, omega.shape[-1]])
        for i in range(0, N_simulations):
            vel_bulk = dark_matter.get_rand_bulkSHM(v0, v_obs, N_bulk_axions)
            vel_stream = dark_matter.get_rand_streamSHM(v0_stream, v_obs_stream, N_stream_axions)
            axion_field_bulk = axion.gen_axion_field(X, vel_bulk, axion_mass = 2 * np.pi, axion_density = 1)
            axion_field_stream = axion.gen_axion_field(X, vel_stream, axion_mass = 2 * np.pi, axion_density = 1)
            axion_field = axion_field_bulk + axion_field_stream
            # for j in range(0, len(t)):
            #     multi_axion[j] = axion_field[j][0][0][0] * axion_field[j][1][1][1]
            multi_axion = axion_field[:,0,0,0] * axion_field[:,1,1,1] 
            multi_axion = multi_axion - np.mean(multi_axion)
            final_psd_axion[i] = axion.gen_PSD(t, multi_axion)
            print('\r' + f'Simulation {i+1} of {N_simulations} complete', end='')
        return np.sum(final_psd_axion, axis = 0)
    

    def axion_field_generatororor(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations):
        '''
        Creates 'N' montee carlo simulations, and averages the result.

        Parameters
        ----------
        t : ndarray
            An array of time values, from [0, T), with T being the period of the signal. 
        v : ndarray
            An array of velocities which span the possible set of velocities for the distribution.
        vel_dist : ndarray
            An array generated by putting the velocity array 'v' into the velocity distribution.

        Returns
        -------
        montee_axion : ndarray
            Array of average signal over 'N' montee carlo simulations in terms of angular frequency.
        '''
        omega = axion.get_fourier_freq(t)
        multi_axion_1 = np.empty([len(t)])
        multi_axion_2 = np.empty([len(t)])
        final_psd_axion = np.empty([N_simulations, omega.shape[-1]])
        for i in range(0, N_simulations):
            vel_bulk = dark_matter.get_rand_bulkSHM(v0, v_obs, N_bulk_axions)
            vel_stream = dark_matter.get_rand_streamSHM(v0_stream, v_obs_stream, N_stream_axions)
            axion_field_bulk = axion.gen_axion_field(X, vel_bulk, axion_mass = 2 * np.pi, axion_density = 1)
            axion_field_stream = axion.gen_axion_field(X, vel_stream, axion_mass = 2 * np.pi, axion_density = 1)
            axion_field = axion_field_bulk + axion_field_stream
            for j in range(0, len(t)):
                multi_axion_1[j] = axion_field[j][0][0][0]
                multi_axion_2[j] = axion_field[j][1][1][1]
                multi_signal = multi_axion_1 - multi_axion_2
        # plt.plot(t, multi_axion_1)
        # plt.plot(t, multi_axion_2)
        plt.plot(t, multi_signal)
        # plt.plot(t, multi_axion_1**2)
        plt.show()
        return None
