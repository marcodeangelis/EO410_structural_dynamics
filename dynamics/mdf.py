import numpy
import numpy.linalg

from .data import DATA

from matplotlib import pyplot

def mass_matrix_mdof(data:dict=None):
    '''
     :param m:   Mass vector 1xd (floats)

     :returns: values of assembled mass matrix
    '''
    if data is None: data=DATA
    d = data['DOF']
    mm = data['M']['value']
    return numpy.eye(d) * numpy.asarray(mm, dtype=float)

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
    # return numpy.asarray([-w**2*M[i][j] + 1j*w*C[i][j] + K[i][j] for j in range(d) for i in range(d)], dtype=complex)

def displacement_amplitude(w:float, data:dict=None):
    '''
    Displacement amplitude in the frequency domain due to harmonic and stationary excitation.
    '''
    if data is None: data=DATA
    s = system_matrix_mdof(w,data)
    inv_s = numpy.linalg.inv(s)
    exci = numpy.asarray(data['EXCITATION']['value'],dtype=float)
    return numpy.abs(inv_s@exci)

# def system_matrix_3dof(w:float,data:dict=None):
#     '''
#     :param m:   Mass vector 1xd (floats)
#     :param k:   Stiffness vector 1xd (floats)
#     :param c:   Damping vector 1xd (floats)

#     :returns: values assembled system matrix
#     '''
#     if data is None: data=DATA
#     m1,m2,m3 = tuple(data['M']['value'])  #tuple(MASS().value())
#     k1,k2,k3 = tuple(data['K']['value'])  #tuple(STIFF().value())
#     c1,c2,c3 = tuple(data['C']['value'])  #tuple(DAMP().value())
#     K = [[k1+k2 ,-k2    , 0  ],
#          [-k2   , k2+k3 , -k3],
#          [ 0    ,-k3    , k3 ]]
#     C = [[c1+c2 ,-c2    , 0  ],
#          [-c2   , c2+c3 , -c3],
#          [ 0    ,-c3    , c3 ]]
#     M = [[m1    ,0      ,0  ],
#          [0     ,m2     ,0  ],
#          [0     ,0      ,m3 ]]
#     D = []
#     for i in range(3):
#         d = []
#         for j in range(3):
#             d.append(-w**2*M[i][j] + 1j*w*C[i][j] + K[i][j])
#         D.append(d)
#     return D

# def displacement_msd_numpy(w:float, data:dict=None):
#     if data is None: data=DATA
#     intensity = DATA['HAMMER']['value']
#     sm = system_matrix(w)
#     sm_arr = numpy.asarray(sm,dtype=complex)
#     invD = numpy.linalg.inv(sm_arr)
#     exci_floor = 1
#     if exci_floor==3: # excitation at floor 3 and so on
#         exci = numpy.array([0,0,intensity])
#     elif exci_floor==2:
#         exci = numpy.array([0,intensity,0])
#     elif exci_floor==1:
#         exci = numpy.array([intensity,0,0])
#     u = invD@exci
#     return u

# def psd(w=None,intensity=1,exci_floor=1):
#     u = displacement_msd_numpy(w=w,intensity=intensity,exci_floor=exci_floor)
#     return numpy.abs(u)