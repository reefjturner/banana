import banana as bn
import numpy as np
import matplotlib.pyplot as plt
c = 299792
v0 = 220e3/c
v_obs = 223e3/c
ma = 2*np.pi
period = 100 # period data collection is taken over
sample_rate = 100 # sampling rate of experiment
N = 50
kai = 1
v0_stream = 50e3/c
v_obs_stream = [150e3/c, 150e3/c, 150e3/c] # random velocity different from bulk
N_bulk_axions = 500
N_stream_axions = 20
N_simulations = 2500
c_length = 2 * np.pi / (ma * v0)

ax = plt.figure().add_subplot(projection='3d')

t = np.linspace(0,period,int(period*sample_rate))
x = np.array([0, 0])
y = np.array([0, 0])
z = np.array([0, 0])
X = np.meshgrid(t, x, y, z, indexing='ij')
# monte_1 = bn.axion.multi_montee_carlo_stream(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations)
monte_1 = bn.axion.multi_montee_carlo(t, X, v0, v_obs, N_bulk_axions, N_simulations)

omega = bn.axion.get_fourier_freq(t)

# this is just to clip the 3D plot inside the axis limits
u_bound = 0
for freq in omega:
    if freq/(2*ma) < 6:
        u_bound += 1

# ax.plot(omega[1:u_bound]/(2*ma), monte_1[1:u_bound]/monte_1[1:u_bound], zs=3, zdir='y', label='0', clip_on=True)
ax.plot(omega[1:u_bound]/(2*ma), monte_1[1:u_bound], zs=3, zdir='y', label='0', clip_on=True)

t = np.linspace(0,period,int(period*sample_rate))
x = np.array([0, c_length/2])
y = np.array([0, 0])
z = np.array([0, 0])
X = np.meshgrid(t, x, y, z, indexing='ij')
# monte = bn.axion.multi_montee_carlo_stream(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations)
monte = bn.axion.multi_montee_carlo(t, X, v0, v_obs, N_bulk_axions, N_simulations)
# ax.plot(omega[1:u_bound]/(2*ma), monte[1:u_bound]/monte_1[1:u_bound], zs=2, zdir='y', label='1/2')
ax.plot(omega[1:u_bound]/(2*ma), monte[1:u_bound], zs=2, zdir='y', label='1/2', clip_on=True)

t = np.linspace(0,period,int(period*sample_rate))
x = np.array([0, c_length])
y = np.array([0, 0])
z = np.array([0, 0])
X = np.meshgrid(t, x, y, z, indexing='ij')
# monte = bn.axion.multi_montee_carlo_stream(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations)
monte = bn.axion.multi_montee_carlo(t, X, v0, v_obs, N_bulk_axions, N_simulations)
# ax.plot(omega[1:u_bound]/(2*ma), monte[1:u_bound]/monte_1[1:u_bound], zs=1, zdir='y', label='1')
ax.plot(omega[1:u_bound]/(2*ma), monte[1:u_bound], zs=1, zdir='y', label='1', clip_on=True)


t = np.linspace(0,period,int(period*sample_rate))
x = np.array([0, 1.5*c_length])
y = np.array([0, 0])
z = np.array([0, 0])
X = np.meshgrid(t, x, y, z, indexing='ij')
# monte = bn.axion.multi_montee_carlo_stream(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations)
monte = bn.axion.multi_montee_carlo(t, X, v0, v_obs, N_bulk_axions, N_simulations)
# ax.plot(omega[1:u_bound]/(2*ma), monte[1:u_bound]/monte_1[1:u_bound], zs=0, zdir='y', label='3/2')
ax.plot(omega[1:u_bound]/(2*ma), monte[1:u_bound], zs=0, zdir='y', label='3/2', clip_on=True)

# # monte = bn.axion.multi_montee_carlo(t, X, v0, v_obs, v0_stream, v_obs_stream, N_bulk_axions, N_stream_axions, N_simulations)
seps = [3,2,1,0]
squad = [0,'$\lambda_c/2$','$\lambda_c$','$3\lambda_c/2$']

ax.set_yticks(seps)
ax.set_yticklabels(squad, minor=False, rotation=45)
ax.set_xlim3d(0, 6)
ax.set_xlabel('$\omega / 2m_a$')
ax.set_ylabel('Separation Distance')
ax.set_zlabel('$S_{\Omega (\mathbf{r})} / S_{\Omega (\mathbf{r_0})}$')
plt.show()

# 25 sims in 4 mins = complete 500 in 20 * 4 mins = 80 mins...
