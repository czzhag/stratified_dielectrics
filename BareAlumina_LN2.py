import numpy as np
import diele_layer_model as dm
import matplotlib.pyplot as plt

def check_degeneracy_TETM_nln2Xtheta(theta=[45,46],n2=[dm.n_alumina,3.08283],n3=[dm.n_ln2,1.161]):
# check the degeneracy breaking between nln2 and theta with TE and TM light.
    #theta=0

    fs=np.linspace(220e9,320e9,1000)
    Rs =np.full((2,2,len(fs)),np.nan)
    Ts =np.full((2,2,len(fs)),np.nan)
    for i in range(2):
    
        n2i = n2[i]
        n3i = n3[i]
        thetai = theta[i]
        eps=[1, n2i**2, n3i**2]
        mu =[1, 1, 1]
        dz =[np.inf, 10.e-3, np.inf]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=dm.get_coeffs_RT(f,thetai,eps,mu,dz,'TE')
            Rs[0,i,ii]=R
            Ts[0,i,ii]=T
            r,t,R,T=dm.get_coeffs_RT(f,thetai,eps,mu,dz,'TM')
            Rs[1,i,ii]=R
            Ts[1,i,ii]=T

   
    fig, ax = plt.subplots(2)

    ax[0].plot(fs/1e9, Rs[0,0,:], color='b', linestyle='-', label='TE (%.1f deg, %.3f, %.3f)'%(theta[0],n2[0],n3[0]))
    ax[0].plot(fs/1e9, Rs[0,1,:], color='b', linestyle='--', label='TE (%.1f deg, %.3f, %.3f)'%(theta[1],n2[1],n3[1]))
    ax[0].plot(fs/1e9, Rs[1,0,:], color='m', linestyle='-', label='TM (%.1f deg, %.3f, %.3f)'%(theta[0],n2[0],n3[0]))
    ax[0].plot(fs/1e9, Rs[1,1,:], color='m', linestyle='--', label='TM (%.1f deg, %.3f, %.3f)'%(theta[1],n2[1],n3[1]))

    ax[0].set_xlim(215,325)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Reflection')
    ax[0].legend()

    ax[1].plot(fs/1e9, Rs[0,0,:]-Rs[0,1,:], color='b', linestyle='-', label=r'$\Delta$TE')
    ax[1].plot(fs/1e9, Rs[1,0,:]-Rs[1,1,:], color='m', linestyle='-', label=r'$\Delta$TM')


    ax[1].set_xlim(215,325)
    ax[1].set_ylabel('Reflection')
    ax[1].set_xlabel('GHz')
    ax[1].legend()
    
    plt.savefig('check_degeneracy_TETM_nln2Xtheta.png',dpi=180,format='png')
    plt.show()


