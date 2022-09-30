'''
This script defines the default globals. 

If nothing is specified the script will return values for a three-degree-of-freedom system.
'''

# Default values 
M = [5.05,     4.98,     4.95]  # 1 [kg] Mass 
K = [35000.0, 43000.0, 43000.0] # 10000 # [Ns] Stiffness 
C = [8.0,     4.0,     5.5]     # 2 # [Ns2]  Damping 

DOF = len(M)

DISP_INIT = [0.,0.,0.]   # Initial displacement 
VELO_INIT = [0.,0.,0.]   # Initial velocity 
INITIALS = [DISP_INIT,VELO_INIT]

F = [100.0, 100.0, 100.0]   # [N] Intensity of force
T = 5  # [s] observation time

# Numerical integration

# INTEGRATOR_STEPS = 1024
STEP_SIZE = 0.0001 # [s] 

DURATION = 3*[0.005] # [s] duration of impact force
AT = 3*[0.01] # [s] peak time of impact

EXCITATION_TYPES= ['IMPACT', 'WHITE_NOISE', 'NONE', 'HARMONIC']
# EXCITATION_TYPE = 'HAMMER' # 'WHITE_NOISE'

DATA={
    'DOF':DOF,
    'M': {
        'value':M,
        'dim':len(M),
        'unit':'kg',
        },
    'K': {
        'value':K,
        'dim':len(K),
        'unit':'N/m',
        },
    'C': {
        'value':C,
        'dim':len(C),
        'unit':'Ns/m',
        },
    'DISP_INIT': {
        'value':DISP_INIT,
        'dim':len(DISP_INIT),
        'unit':'m',
        },
    'VELO_INIT': {
        'value':VELO_INIT,
        'dim':len(VELO_INIT),
        'unit':'m/s',
    },
    'EXCITATION': {
        'value':F,
        'dim':len(F),
        'at_time':AT,
        'duration':DURATION,
        'type':EXCITATION_TYPES[0],
        'unit':'N',
        },
    'TIME_WINDOW': {
        'value':T,
        'unit':'s',
        },
    'STEP_SIZE':{
        'value':STEP_SIZE,
        'unit':'s'
        },
}


def generate_data(dof:int=None) -> dict:
    if dof is None: return DATA
    if dof < 3:
        m=M[0]
        k=K[0]
        c=[0]
        di=DISP_INIT[0]
        vi=VELO_INIT[0]
        f=F[0]
        dur=DURATION[0]
        at=DURATION[0]
    elif dof == 3: return DATA
    elif dof>3:
        ad=dof-3
        m=M+ad*[M[-1]]
        k=K+ad*[K[-1]]
        c=C+ad*[C[-1]]
        di=DISP_INIT+ad*[DISP_INIT[-1]]
        vi=VELO_INIT+ad*[VELO_INIT[-1]]
        f=F+ad*[F[-1]]
        dur=DURATION+ad*[DURATION[-1]]
        at=AT+ad*[AT[-1]]
    data = DATA.copy()
    data['DOF']=dof
    data['M']['value']=m
    data['M']['dim']=len(m)
    data['K']['value']=k
    data['K']['dim']=len(k)
    data['C']['value']=c
    data['C']['dim']=len(c)
    data['DISP_INIT']['value']=di
    data['DISP_INIT']['dim']=len(di)
    data['VELO_INIT']['value']=vi
    data['VELO_INIT']['dim']=len(di)
    data['EXCITATION']['value']=f
    data['EXCITATION']['dim']=len(f)
    data['EXCITATION']['at_time']=at
    data['EXCITATION']['duration']=dur
    return data