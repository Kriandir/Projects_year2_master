import numpy as np
import matplotlib.pyplot as plt

def hamiltonian_sym(p,q,v,u,j,k):
    """
    Calculate the Hamiltonian of the harmonic oscillator.

    @x: float value of x
    @y: float value of y
    @omega: int/float value of spring constant
    """
    # Compute energy of particle itself
    particle_element = 0.5*p**2 + 0.25*((q**2 - 1)**2)

    # Compute interaction terms with heat bath
    bath_element = np.sum(((v**2)*j**2)/2 + 0.5*(u-q)**2)

    # Return total energy
    return particle_element + bath_element


"""
The functions for the sv method
"""
def g_q_sv(q,u,k):
    q = -q*(q**2-1) - k*(np.sum(q-u))
    return q

def f_p_sv(p):
    p = p
    return p

def f_v_sv(v,j,k):
    v = v*j**2
    return v

def g_u_sv(q,u):
    u = q-u
    return u

def kz_symp_euler_integrate(p0,q0,u_list,v_list,N,t_min,t_max,dt):
    """
    Perform SV Integration to compute the time lapse for a
    distinguishable particle in a heat bath.
    """
    # initialize lists and k
    k=1
    j_list = (np.arange(1,N+1,1))
    t_list = np.arange(t_min,t_max,dt)
    p_list = np.empty(len(t_list)+1)
    q_list = np.empty(len(t_list)+1)
    H_list = np.empty(len(t_list)+1)
    p_list[0] = p0
    q_list[0] = q0

    for i, t in enumerate(t_list):
        #
		# # Compute current value of Hamiltonian
        H_list[i] = hamiltonian_sym(p_list[i], q_list[i], v_list,u_list,j_list,k)

		# Calculate next steps q
    	p_star = p_list[i] + 0.5 * dt * g_q_sv(q_list[i],u_list,k)
    	q_list[i+1] = q_list[i] + dt * f_p_sv(p_star)

        # calculate next step heathbath u
        for u,v,j in np.nditer([u_list,v_list,j_list], op_flags=["readwrite"]):
            v_star = v + 0.5 * dt * g_u_sv(q_list[i],u)
            u[...] = u + dt * f_v_sv(v_star,j,k)

        # calculate next steps for both v and p
        p_list[i+1] = p_star + 0.5 * dt * g_q_sv(q_list[i+1],u_list,k)
        for u,v,j in np.nditer([u_list,v_list,j_list], op_flags=["readwrite"]):
            v[...] = v_star + 0.5 *dt *g_u_sv(q_list[i+1],u)


	# Return p, q, Hamiltonian and time steps
    return p_list[:-1],q_list[:-1],H_list[:-1], t_list

"""
ANRUFE EXAMPLE
"""

N = 10
u = np.random.rand(N)
v = np.random.rand(N)
p, q, H, t = kz_symp_euler_integrate(p0=0, q0=0.5, u_list=u, v_list=v, N=N, t_min=0, t_max=2*np.pi, dt=0.01)

# Compute cummulative average over p and H
average_p = np.cumsum(p)/np.arange(1, len(p) + 1)
average_h = np.cumsum(H)/np.arange(1, len(H) + 1)
# Plot
plt.plot(t, q, label='q')
plt.plot(t, p, label='p')
plt.plot(t,average_p,label = 'average_p')
plt.axhline(0, color='black')
plt.ylabel(r'q & p (arb. units)')
plt.xlabel('Time (arb. units)')
plt.title("p(t) and q(t) with SV for N = 10 and dt = 0.01")
plt.legend()
plt.savefig("n10sv")
plt.show()
plt.plot(t, H, label='H')
plt.plot(t,average_h, label = "average_H")
plt.title("H(t) with SV for N = 10 and dt = 0.01")
plt.ylabel("Energy (arb. units)")
plt.xlabel("Time (arb. units)")
plt.savefig("n10Hsv")
plt.show()

# repeat for N = 100
N = 100
u = np.random.rand(N)
v = np.random.rand(N)

p, q, H, t = kz_symp_euler_integrate(p0=0, q0=0.5, u_list=u, v_list=v, N=N, t_min=0, t_max=2*np.pi, dt=0.01)
average_p = np.cumsum(p)/np.arange(1, len(p) + 1)
plt.plot(t, q, label='q')
plt.plot(t, p, label='p')
plt.plot(t,average_p,label = 'average_p ')
plt.axhline(0, color='black')
plt.ylabel(r'q & p (arb. units)')
plt.xlabel('Time (arb. units)')
plt.title("p(t) and q(t) with SV for N = 100 and dt = 0.01")
plt.legend()
plt.savefig("n100sv")
plt.show()


average_h = np.cumsum(H)/np.arange(1, len(H) + 1)
plt.plot(t, H, label='H')
plt.plot(t,average_h, label = "average_H")
plt.title("H(t) with SV for N = 100 and dt =0.01")
plt.ylabel("Energy (arb. units)")
plt.xlabel("Time (arb. units)")
plt.legend()
plt.savefig("n100Hsv")
plt.show()
