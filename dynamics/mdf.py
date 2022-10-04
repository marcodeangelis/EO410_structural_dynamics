import numpy
import numpy.linalg
from numpy import (asarray, arange, zeros, ndarray)

from scipy.linalg import eig # generalised eigenvalue problem

from .data import DATA

PI = numpy.pi
SORT=numpy.sort
SQRT=numpy.sqrt

#     K = [[k1+k2 ,-k2    , 0  ],
#          [-k2   , k2+k3 , -k3],
#          [ 0    ,-k3    , k3 ]]
#     C = [[c1+c2 ,-c2    , 0  ],
#          [-c2   , c2+c3 , -c3],
#          [ 0    ,-c3    , c3 ]]
#     M = [[m1    ,0      ,0  ],
#          [0     ,m2     ,0  ],
#          [0     ,0      ,m3 ]]

def mass_matrix_mdof(data:dict=None):
    '''
     :param m:   Mass vector 1xd (floats)

     :returns: values of assembled mass matrix
    '''
    if data is None: data=DATA
    d = data['DOF']
    mm = data['M']['value']
    return numpy.eye(d) * asarray(mm, dtype=float)

def assemble_equilibrium_matrix_mdof(k_vector:list): # works for both stiffness and damping matrix 
    d = len(k_vector) # degrees of freedom
    k_vector=numpy.asarray(k_vector, dtype=float)
    v = -k_vector[1:]
    w = numpy.concatenate((k_vector[1:],[0]))
    s=[numpy.insert(v,i,w[i]) for i in range(d)]
    one_off_diag=numpy.eye(d)+numpy.eye(d,k=1)+numpy.eye(d,k=-1)
    diag=numpy.eye(d)*k_vector
    return diag + one_off_diag * s   
    
def stiffness_matrix_mdof(data:dict=None):
    '''
     :param k:   Stiffness vector 1xd (floats)

     :returns: values of assembled stiffness matrix
    '''
    if data is None: data=DATA
    k = data['K']['value'] # vector of stiffness values [k1,k2,k3,...,kd]
    return assemble_equilibrium_matrix_mdof(k)

def damping_matrix_mdof(data:dict=None):
    '''
     :param m:   Damping vector 1xd (floats)

     :returns: values of assembled damping matrix
    '''
    if data is None: data=DATA
    c = data['C']['value']
    return assemble_equilibrium_matrix_mdof(c)  


# FORCED (HARMONIC) VIBRATIONS #

def system_matrix_mdof(w:float,data:dict=None):
    '''
    :param m:   Mass vector 1xd (floats)
    :param k:   Stiffness vector 1xd (floats)
    :param c:   Damping vector 1xd (floats)

    :returns: values of assembled system matrix
    '''
    if data is None: data=DATA
    d = data['DOF']
    M = mass_matrix_mdof(data)
    K = stiffness_matrix_mdof(data)
    C = damping_matrix_mdof(data)
    return -w**2*M + 1j*w*C + K 

def displacement_amplitude(w:float, data:dict=None):
    '''
    Displacement amplitude in the frequency domain due to harmonic and stationary excitation.
    '''
    if data is None: data=DATA
    s = system_matrix_mdof(w,data)
    inv_s = numpy.linalg.inv(s)
    exci = asarray(data['EXCITATION']['value'],dtype=float)
    return numpy.abs(inv_s@exci)


# FREE VIBRATIONS #

def roots_characteristic_polynomial_3dof(data:dict=None):
    '''
    :param m:   Mass vector 1xd (floats)
    :param k:   Stiffness vector 1xd (floats)

    :returns: natural frequencies of 3dof system
    '''
    if data is None: data=DATA
    m = data['M']['value']
    m1,m2,m3 = m[0],m[1],m[2]
    k = data['K']['value']
    k1,k2,k3 = k[0],k[1],k[2]
    P = 4*[0]
    P[0] = -m1*m2*m3
    P[1] = m1*m2*k3 + m2*m3*(k1+k2) + m1*m3*(k2+k3)
    P[2] = - ( m1*k2*k3 + m2*(k1*k3+k2*k3) + m3*(k1*k2+k1*k3+k2*k3) )
    P[3] = k1*k2*k3
    return numpy.roots(P)

def natural_frequencies_3dof(data:dict=None):
    if data is None: data=DATA
    return SORT(SQRT(asarray(roots_characteristic_polynomial_3dof(data=data),dtype=float))/(2*PI))

def eigenvalue_solver(data:dict=None): 
    if data is None: data=DATA
    m = mass_matrix_mdof(data)
    k = stiffness_matrix_mdof(data)
    c, Q = eig(k, b=m) # generalised eigenvalue problem |K - M c| = 0
    return numpy.real(c), Q

def natural_frequencies_mdof(data:dict=None): 
    if data is None: data=DATA
    c,_ = eigenvalue_solver(data)
    return SORT(SQRT(c)/(2*PI)) 

# SIMULATOR (step-wise time integration)
from scipy.integrate import odeint
from numpy import (exp, power)

# def simulator(data:dict=None):
#     if data is None: data = DATA
#     t_length = data['TIME_WINDOW']['value']
#     t_step = data['STEP_SIZE']['value']
#     n = data['MC_samples']
#     ex = data['EXCITATION']['type']
    
#     def derivs(X, t): # This function defines the ODEs for the 3DOF system 
#         # Here X is the state vector such that x1=X[0] and xdot1=X[N-1]. 
#         # This function should return [x1dot,...xNdot, xdotdot1,...xdotdotN]
#         x1, x2, x3, xdot1, xdot2, xdot3 = X
#         # compute ODE values
#         xdotdot1 = -(c[0] / m[0]) * (xdot1) -(c[1] / m[0]) * (xdot1 - xdot2) -(k[0] / m[0]) * x1 -(k[1] / m[0]) * (x1 - x2) -(b[0] / m[0]) * (x1 - x2) * (x1 - x2) * (x1 - x2) + f(t=t) / m[0] #(FORCEAMP/ MASS[0])*np.exp(-np.power(t - mu, 2) / (2 * np.power(sig, 2)))
#         xdotdot2 = -(c[1] / m[1]) * (xdot2 - xdot1) -(c[2] / m[1]) * (xdot2 - xdot3) -(k[1] / m[1]) * (x2 - x1) -(k[2] / m[1]) * (x2 - x3)-(b[1] / m[1]) * (x2 - x1) * (x2 - x1) * (x2 - x1)-(b[2] / m[1]) * (x2 - x3) * (x2 - x3) * (x2 - x3) 
#         xdotdot3 = -(c[2] / m[2]) * (xdot3 - xdot2) -(k[2] / m[2]) * (x3 - x2) -(b[2] / m[2]) * (x3 - x2) * (x3 - x2) * (x3 - x2) 
#         return [xdot1, xdot2, xdot3, xdotdot1, xdotdot2, xdotdot3]
    
#     # define the time base parameters for the ODE integration
#     ts = arange(0, t_length, t_step) 
#     xs = zeros((n, 5, len(ts)), dtype=float)
#     # generate random samples
#     if n==1:  # use default
#         mm = [data['M']['value']]
#         kk = [data['K']['value']]
#         cc = [data['C']['value']]
#         bb = [data['BETA']['value']]
#         ii = [numpy.concatenate((data['DISP_INIT']['value'],data['VELO_INIT']['value']))]
#     else:
#         mm = sample('M',data,seed=10) #M = MASS(Mi,dispersion=disp[0]).sample(N=N,seed=10) # MASS=np.random.normal(loc = MASSm, scale = MASSs)
#         kk = sample('K',data,seed=10) #K = STIFF(Ki,dispersion=disp[1]).sample(N=N,seed=10) # STIFF=np.random.normal(loc = STIFFm, scale = STIFFs)
#         bb = sample('BETA',data,seed=10) #B = BETA().sample(N=N,seed=10) # BETA=np.random.normal(loc = BETAm, scale = BETAs)
#         cc = sample('C',data,seed=10) #C = DAMP(Ci,dispersion=disp[2]).sample(N=N,seed=10) # DAMP=np.random.normal(loc = DAMPm, scale = DAMPs)
#         ii_d = sample('DISP_INIT',data,seed=10) #I = INIT(x=Xi,v=Vi,dispersion=d_init).sample(N=N,seed=10) # Init0=np.random.normal(loc = X0m, scale = X0s)
#         ii_v = sample('VELO_INIT',data,seed=10)
#         ii=[numpy.concatenate((ii_d[k,:],ii_v[k,:])) for k in range(n)]
#     if ex=='HAMMER':
#         f = hammer_force(data) #f = HFORCE(duration=1e-2)
#     elif ex=='WHITE_NOISE':
#         f = WFORCE(50,duration=t_length)
#     else: raise(Exception('Unknown Input'))
#     for i in range(n):
#         m,k,b,c,init0 = mm[i],kk[i],bb[i],cc[i],ii[i]
#         Xs = odeint(derivs, init0, ts)
#         # extract the displacements from the return vector
#         xs[i,:,:]= [ts, Xs[:,0], Xs[:,1], Xs[:,2],f(t=ts)]
#     return ts, xs

def hammer_force(data:dict):
    att = data['EXCITATION']['at_time'][0]
    amp = data['EXCITATION']['value'][0]
    dur = data['EXCITATION']['duration'][0]
    def fun(t:float):
        return amp * exp(-power(t - att, 2.) / (2 * power(dur, 2.)))
    return fun

def deterministic_solver(data:dict=None,m=3*[None],k=3*[None],c=3*[None],b=3*[None],i_d=3*[None],i_v=3*[None]):
    if data is None: data = DATA
    t_length = data['TIME_WINDOW']['value']
    t_step = data['STEP_SIZE']['value']
    ex = data['EXCITATION']['type']

    if type(m)!=ndarray: m=asarray(m)
    if type(k)!=ndarray: k=asarray(k)
    if type(c)!=ndarray: c=asarray(c)
    if type(i_d)!=ndarray: i_d=asarray(i_d)
    if type(i_v)!=ndarray: i_v=asarray(i_v)

    # define the time base parameters for the ODE integration
    ts = arange(0, t_length, t_step) 
    # xs = zeros((5, len(ts)), dtype=float)

    if ex=='IMPACT':
        f = hammer_force(data)
    elif ex=='WHITE_NOISE': # not yet implemented
        f = WFORCE(50,duration=t_length)

    def derivatives(fun):
        def derivs(X, t):
            # Here X is the state vector such that x1=X[0] and xdot1=X[N-1]. 
            # This function should return [x1dot,...xNdot, xdotdot1,...xdotdotN]
            x1, x2, x3, xdot1, xdot2, xdot3 = X
            # compute ODE values
            xdotdot1 = -(c[0] / m[0]) * (xdot1) -(c[1] / m[0]) * (xdot1 - xdot2) -(k[0] / m[0]) * x1 -(k[1] / m[0]) * (x1 - x2)  + fun(t=t) / m[0] 
            xdotdot2 = -(c[1] / m[1]) * (xdot2 - xdot1) -(c[2] / m[1]) * (xdot2 - xdot3) -(k[1] / m[1]) * (x2 - x1) -(k[2] / m[1]) * (x2 - x3) 
            xdotdot3 = -(c[2] / m[2]) * (xdot3 - xdot2) -(k[2] / m[2]) * (x3 - x2)
            return [xdot1, xdot2, xdot3, xdotdot1, xdotdot2, xdotdot3]
        return derivs

    if all(m==None):m=data['M']['value']
    if all(k==None):k=data['K']['value']
    if all(c==None):c=data['C']['value']
    if all(i_d==None):i_d=data['DISP_INIT']['value']
    if all(i_v==None):i_v=data['VELO_INIT']['value']
    init = numpy.concatenate((i_d,i_v))

    Xs = odeint(derivatives(f), init, ts)
    # extract the displacements from the return vector
    # xs[:,:]= [ts, Xs[:,0], Xs[:,1], Xs[:,2],f(t=ts)]
    return ts, Xs[:,:3], f(t=ts)

# def simulator_(data:dict=None):
#     if data is None: data = DATA
#     n = data['MC_samples']
#     xxss=[]
#     if n==0:  # use default
#         ts,xs,force = deterministic_solver(data)
#         xxss.append(xs)
#     else:
#         mm = sample('M',data,seed=data['M']['seed']) 
#         kk = sample('K',data,seed=data['K']['seed']) 
#         bb = sample('BETA',data,seed=data['BETA']['seed']) 
#         cc = sample('C',data,seed=data['C']['seed']) 
#         ii_d = sample('DISP_INIT',data,seed=data['DISP_INIT']['seed']) 
#         ii_v = sample('VELO_INIT',data,seed=data['VELO_INIT']['seed'])
#         # ii=[numpy.concatenate((ii_d[j,:],ii_v[j,:])) for j in range(n)]
#         for j in range(n):
#             ts,xs,force = deterministic_solver(data,m=mm[j],k=kk[j],c=cc[j],b=bb[j],i_d=ii_d[j],i_v=ii_v[j])
#             xxss.append(xs)
#     return ts,xxss,force


# Excitation = {
#     'Force':{
#         'intensity':{
#             'value':1,
#             'law':fun,
#         },
#     },
#     'Inertial':{
#         'intensity':{
#             'value':1,
#             'law':fun,
#         }
#     },
# }