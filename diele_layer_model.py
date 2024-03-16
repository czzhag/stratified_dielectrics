import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from getdist import plots, gaussian_mixtures
g = plots.get_subplot_plotter()
g.settings.legend_fontsize = 18
g.settings.axes_labelsize = 18
g.settings.axes_fontsize = 16
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_y_rotation = 45
g.settings.title_limit_fontsize = 1
# This is a collection of functions for treating dielectric layers
# Most variables are in SI. Angles are in deg.

c=299792458.
n_alumina=3.08
n_1090=1.485
n_2850=2.231
n_e100=1.52
n_b3mix=1.745
n_ln2 =1.22

def snell(n_i,theta_i,n_t):
    if np.abs(theta_i)>=90:
        raise ValueError("Input angle must be in arrange (-90,90) degree.")
    sin_t = math.sin(theta_i*np.pi/180.)/(n_t/n_i)

    if np.abs(sin_t)>=1:
        if n_i>n_t:
            sin_c = math.sin(90*np.pi/180.)/(n_i/n_t)
            theta_c = math.asin(sin_c)*180/np.pi
            print("snell: total reflection occurs beyond %.2f, n_i/n_t=%.2f/%.2f"%(theta_c, n_i, n_t))
            return None
        else:
            raise ValueError("Check inputs insuring their physical consistency.")
    else:
        theta_t = math.asin(sin_t)*180/np.pi
        return theta_t

def get_diele_matrix(f, theta, dz, eps_r, mu_r, polar='TE'):
# f is the frequency in Hz
# dz is the thickness in m
# need relative permittivity and permeability
    n = np.sqrt(eps_r * mu_r)
    if polar == 'TE':
        p = np.sqrt(eps_r/mu_r)*math.cos(theta/180.*np.pi)
    elif polar == 'TM':
        p = np.sqrt(mu_r/eps_r)*math.cos(theta/180.*np.pi)
    else:
        raise ValueError('Polarization unrecognized - can only be TE or TM.')

    k0 = 2*np.pi/(c/f)
    beta = k0*n*dz*math.cos(theta/180.*np.pi)
    m = np.zeros((2,2),dtype=complex)
    m[0,0] = math.cos(beta)
    m[1,1] = math.cos(beta)
    m[0,1] = -1j/p*math.sin(beta)
    m[1,0] = -1j*p*math.sin(beta)

    return m

def get_coeffs_RT(f,theta=90,eps=[1],mu=[1],dz=[np.inf],polar='TE'):
# f is the frequency in Hz
# reflection and transmittance with the stratified dielectric layer model.
# theta is the incident angle in degree.
# eps, mu, dz are relative permittivities, permeabilities, and thickness of layers

    if not len(eps)==len(mu)==len(dz):
        raise ValueError('eps, mu, and dz must be in the same length.')
    elif len(dz)<=1:
        print('No interface has been found. R=0, T=1.')
        r,t = 0,1
        R,T = 0,1
    else:
        theta_l = snell(np.sqrt(eps[0]*mu[0]),theta,np.sqrt(eps[-1]*mu[-1]))
        if polar == 'TE':
            p_0 = np.sqrt(eps[0]/mu[0])*math.cos(theta/180.*np.pi)
            p_l = np.sqrt(eps[-1]/mu[-1])*math.cos(theta_l/180.*np.pi)
        elif polar == 'TM':
            p_0 = np.sqrt(mu[0]/eps[0])*math.cos(theta/180.*np.pi)
            p_l = np.sqrt(mu[-1]/eps[-1])*math.cos(theta_l/180.*np.pi)
        else:
            raise ValueError('Polarization unrecognized - can only be TE or TM.')

        m=np.array([[1,0],[0,1]], dtype=complex)
        if len(dz)==2:
            print('Only one interface has been found.')
        else:

            ilayer=0
            while True:
                ilayer = ilayer+1
                if ilayer == len(dz)-1:
                    break
                else:
                    epsi = eps[ilayer]
                    mui  = mu[ilayer]
                    dzi  = dz[ilayer]
                    thetai = snell(np.sqrt(eps[0]*mu[0]),theta,np.sqrt(epsi*mui))
                    mi = get_diele_matrix(f, thetai, dzi, epsi, mui, polar)
                    m = np.matmul(m, mi)

        denom = (m[0,0] + m[0,1]*p_l)*p_0 + (m[1,0] + m[1,1]*p_l)
        r = ((m[0,0] + m[0,1]*p_l)*p_0 - (m[1,0] + m[1,1]*p_l)) / denom
        t = 2*p_0 / denom
        R = np.abs(r)**2
        T = p_l/p_0*np.abs(t)**2

    return r,t,R,T

def rep_bornNwolf_plot1p18():
# reproduce plot 1.18 of born and wolf
# this looks good

    theta=0
    f=150.e9 #doesn't matter
    eps1=1
    eps2s=[3.0**2, 2.0**2, 1.7**2, 1.5**2, 1.4**2,1.2**2,1.0**2]
    eps3=1.5**2

    Rs = np.full((len(eps2s),25), np.nan)
    dzs= np.full((25), np.nan)
    for ieps2 in range(len(eps2s)):
        eps = [eps1, eps2s[ieps2], eps3]
        mu  = [1,1,1]
        for idz2 in range(25):
            dz = [np. inf, c/f/np.sqrt(eps[1]*mu[1])*0.05*idz2, np.inf]
            r,t,R,T = get_coeffs_RT(f,theta,eps,mu,dz)
            Rs[ieps2,idz2] = R
            dzs[idz2] = dz[1]
       
        plt.plot(dzs*np.sqrt(eps[1]*mu[1]), Rs[ieps2,:], label='n2 = %.1f'%np.sqrt(eps[1]*mu[1]))


    plt.xlabel(r'$n_2h$')
    plt.ylabel('R')
    plt.legend()
    plt.savefig('rep_bornNwolf_plot1p18.png',dpi=180,format='png')
    #plt.show()

def rep_alumina10mm(dz2=10e-3,plot_refl=True):
# bare filter transmittance
    theta=0
    fs=np.linspace(220e9,320e9,500)
    eps=[1, (n_alumina)**2, 1]
    mu =[1, 1, 1]
    dz =[np.inf, dz2, np.inf]
    Rs =np.full_like(fs,np.nan)
    Ts =np.full_like(fs,np.nan)

    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T

    plt.xlim(215,325)
    plt.xlabel('GHz')
    if plot_refl:
        plt.plot(fs/1e9, Rs)
        plt.ylabel('Reflectivity')
    else:
        plt.plot(fs/1e9, Ts)
        plt.ylabel('Transmittance')
    plt.title('Bare Alumina %.0fmm, in Air'%(dz2*1e3))
    plt.savefig('rep_alumina%.0fmm.png'%(dz2*1e3),dpi=180,format='png')
    plt.show()

def rep_alumina10mm_withba4coating(theta=0,polar='TE'):
# bare filter transmittance
    #theta=0
    fs=np.linspace(100e9,400e9,1000)
    eps=[1, n_1090**2, n_e100**2, n_2850**2, n_e100**2, n_alumina**2, n_e100**2, n_2850**2, n_e100**2, n_1090**2, 1]
    mu =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dz =[np.inf, 0.117e-3, 0.0254e-3, 0.056e-3, 0.0254e-3, 10.e-3, 0.0254e-3, 0.056e-3, 0.0254e-3, 0.117e-3, np.inf]
    Rs =np.full_like(fs,np.nan)
    Ts =np.full_like(fs,np.nan)

    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz,polar)
        Rs[ii]=R
        Ts[ii]=T

    plt.plot(fs/1e9, Ts)
    plt.xlim(180,320)
    plt.ylim(0.9, 1)
    plt.xlabel('GHz')
    plt.ylabel('Transmittance')
    plt.title('Alumina 10mm with BA4 coating in vacuum/air, %.1f-deg incidence, %s'%(theta,polar))
    plt.savefig('rep_alumina10mm_withba4coating_theta%.0f_%s.png'%(theta,polar),dpi=180,format='png')
    plt.show()

def make_alumina10mm_refXtrans(polar='TE'):
    fs=np.linspace(100e9,200e9,1000)
    thetas=np.linspace(10,50,9)
    Rs =np.full((len(thetas),len(fs)),np.nan)
    Ts =np.full((len(thetas),len(fs)),np.nan)

    n3 = 1
    d2 = 10.e-3
    eps=[1, n_alumina**2, n3**2]
    mu =[1, 1, 1]
    dz =[np.inf, d2, np.inf]

    for jj in range(len(thetas)):
        theta=thetas[jj]
        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz,polar)
            Rs[jj,ii]=R
            Ts[jj,ii]=T

    Rmax = np.max(Rs, axis=1)
    plt.plot(thetas, Rmax, marker='^', mfc='k', mec='k', ms=6, ls='-', c='k', lw=1, label='d2=%.0f mm, n3=%.1f'%(d2*1e3,n3))
    
    eps_ln2 = 1.4
    n3 = np.sqrt(eps_ln2)
    eps=[1, n_alumina**2, n3**2]

    for jj in range(len(thetas)):
        theta=thetas[jj]
        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz,polar)
            Rs[jj,ii]=R

    Rmax = np.max(Rs, axis=1)
    plt.plot(thetas, Rmax, marker='^', mfc='b', mec='b', ms=6, ls='-', c='b', lw=1, label='d2=%.0f mm, n3=%.1f'%(d2*1e3,n3))

    n3 = 1
    d2 = 5.e-3
    eps=[1, n_alumina**2, n3**2]
    dz =[np.inf, d2, np.inf]

    for jj in range(len(thetas)):
        theta=thetas[jj]
        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz,polar)
            Rs[jj,ii]=R
            Ts[jj,ii]=T

    Rmax = np.max(Rs, axis=1)
    #plt.plot(thetas, Rmax, marker='^', mfc='k', mec='k', ms=6, ls='--', c='k', lw=1, label='d2=%.0f mm, n3=%.1f'%(d2*1e3,n3))
    
    
    n3 = np.sqrt(eps_ln2)
    d2 = 10.e-3
    dz =[np.inf, d2, np.inf]
    scales = np.linspace(0.9,1.1,11)
    cmap = plt.get_cmap('Blues', len(scales))
    for iscale in range(len(scales)):
        scale = scales[iscale]
        eps=[1, (n_alumina*scale)**2, n3**2]

        for jj in range(len(thetas)):
            theta=thetas[jj]
            for ii in range(len(fs)):
                f=fs[ii]
                r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz,polar)
                Rs[jj,ii]=R

        Rmax = np.max(Rs, axis=1)
        plt.plot(thetas, Rmax, ls='-.', c=cmap(iscale), lw=1)#, label='d2=%.0f mm, n3=%.1f, %.1f*n_al2o3'%(d2*1e3,n3,scale))

    plt.xlabel('deg')
    plt.ylabel('Max Reflection')
    plt.title('alumina filter in air-LN2 interface, %s'%polar)
    plt.grid()

    norm = mpl.colors.Normalize(vmin=0.9, vmax=1.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0.9, 1.1, len(scales))) 
    plt.legend()
    plt.savefig('make_alumina10mm_refXtrans_%s.png'%polar,dpi=180,format='png')
    plt.show()

def make_alumina10mm_withba4coating_n3(theta=0,n3=[1],polar='TE'):
# bare filter transmittance
    #theta=0
    fs=np.linspace(100e9,400e9,1000)
    Rs =np.full((len(n3),len(fs)),np.nan)
    Ts =np.full((len(n3),len(fs)),np.nan)
    
    for i in range(len(n3)):
        n3i = n3[i]
        eps=[1, n_1090**2, n_e100**2, n_2850**2, n_e100**2, n_alumina**2, n_e100**2, n_2850**2, n_e100**2, n_1090**2, n3i**2]
        mu =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dz =[np.inf, 0.117e-3, 0.0254e-3, 0.056e-3, 0.0254e-3, 10.e-3, 0.0254e-3, 0.056e-3, 0.0254e-3, 0.117e-3, np.inf]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz,polar)
            Rs[i,ii]=R
            Ts[i,ii]=T

        plt.plot(fs/1e9, Rs[i,:], label='n3=%.1f'%n3i)

    plt.xlim(180,320)
    plt.ylim(-0.05, 0.3)
    plt.xlabel('GHz')
    plt.ylabel('Reflection')
    plt.legend()
    plt.grid()
    plt.title('Alumina 10mm with BA4 coating at interface between air and medium with n3, %.1f-deg incidence, %s'%(theta,polar))
    plt.savefig('make_alumina10mm_withba4coating_n3_theta%.0f_%s.png'%(theta,polar),dpi=180,format='png')
    plt.show()

def make_alumina10mm_n3_theta():
# bare filter transmittance
    #theta=0
    fs=np.linspace(220e9,320e9,500)
    mu =[1, 1, 1]
    dz =[np.inf, 10e-3, np.inf]
    Rs =np.full_like(fs,np.nan)
    Ts =np.full_like(fs,np.nan)

    plt.figure(figsize=(14,4))

    theta0=45
    theta=theta0
    eps=[1, (n_alumina)**2, n_ln2**2]
    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T
    plt.plot(fs/1e9, Rs, color='k', label='theta=%d, n3=%.1f'%(theta, np.sqrt(eps[2])))

    theta=theta0-2
    eps=[1, (n_alumina)**2, n_ln2**2]
    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T
    plt.plot(fs/1e9, Rs, color='k', linestyle='--',linewidth=0.5, label='theta=%d+-2, n3=%.1f'%(theta0, np.sqrt(eps[2])))

    theta=theta0+2
    eps=[1, (n_alumina)**2, n_ln2**2]
    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T
    plt.plot(fs/1e9, Rs, color='k', linestyle='--', linewidth=0.5)

    theta=theta0
    eps=[1, (n_alumina)**2, 1.3**2]
    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T
    plt.plot(fs/1e9, Rs, color='b', label='theta=%d, n3=%.1f'%(theta, np.sqrt(eps[2])))

    theta=theta0-2
    eps=[1, (n_alumina)**2, 1.3**2]
    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T
    plt.plot(fs/1e9, Rs, color='b', linestyle='--',linewidth=0.5, label='theta=%d+-2, n3=%.1f'%(theta0, np.sqrt(eps[2])))

    theta=theta0+2
    eps=[1, (n_alumina)**2, 1.3**2]
    for ii in range(len(fs)):
        f=fs[ii]
        r,t,R,T=get_coeffs_RT(f,theta,eps,mu,dz)
        Rs[ii]=R
        Ts[ii]=T
    plt.plot(fs/1e9, Rs, color='b', linestyle='--', linewidth=0.5)


    plt.legend()
    plt.xlim(215,325)
    plt.xlabel('GHz')
    plt.ylabel('Reflectivity')
    plt.title('Bare Alumina 10mm, Air-LN2')
    plt.savefig('make_alumina10mm_n3_theta.png',dpi=180,format='png')
    plt.show()

def fisher1(noise=0.01, polar='TE'):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['theta', 'n_1090', 'n_2850', 'n_e100', 'n_almina', 'n_ln2','d_1090A','d_1090B','d_e100','d_2850A','d_2850B']
    params0 = [45,  #0, theta
        n_1090,     #1,
        n_2850,     #2,
        n_e100,     #3,
        n_alumina,  #4,
        n_ln2,       #5, n_LN2
        0.117e-3,   #6, d_1090A
        0.117e-3,   #7, d_1090B
        0.0254e-3,  #8, d_e100
        0.056e-3,   #9, d_2850A
        0.056e-3]   #10, d_2850B
        #10e-3       #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[1]**2, #1090A
            params[3]**2, #e100
            params[2]**2, #2850A
            params[3]**2, #e100
            params[4]**2, #alumina
            params[3]**2, #e100
            params[2]**2, #2850B
            params[3]**2, #e100
            params[1]**2, #1090B
            params[5]**2] #LN2
        dz= [np.inf,#air
            params[6],  #1090A
            params[8],  #e100
            params[9],  #2850A
            params[8],  #e100
            10e-3,      #alumina
            params[8],  #e100
            params[10], #2850B
            params[8],  #e100
            params[7],  #1090B
            np.inf]     #LN2
        mu = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,params[0],eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0 = model(fs, params0, polar)

    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-12)
        Rsi = model(fs, paramsi, polar)
        dRsi= (Rsi - Rs0)/abs(params0[ii]*1e-12)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-12)
            Rsj = model(fs, paramsj, polar)
            dRsj= (Rsj - Rs0)/abs(params0[jj]*1e-12)

            fisher[ii,jj] = noise**-2 * np.sum(dRsi * dRsj)
            if not ii==jj:
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))
 
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    g.triangle_plot(mx,
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, %s'%(noise,polar)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue'],
      line_args = [{'lw':2, 'color':'darkblue'}]
      )
    g.export('testtri_noisep%.0E_%s.png'%(noise,polar))


    return params0, params_name, inv_fisher
    
def fisher2polar(noise=0.01):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['theta', 'n_1090', 'n_2850', 'n_e100', 'n_almina', 'n_ln2','d_1090A','d_1090B','d_e100','d_2850A','d_2850B']
    params0 = [45,  #0, theta
        n_1090,     #1,
        n_2850,     #2,
        n_e100,     #3,
        n_alumina,  #4,
        n_ln2,       #5, n_LN2
        0.117e-3,   #6, d_1090A
        0.117e-3,   #7, d_1090B
        0.0254e-3,  #8, d_e100
        0.056e-3,   #9, d_2850A
        0.056e-3]   #10, d_2850B
        #10e-3       #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[1]**2, #1090A
            params[3]**2, #e100
            params[2]**2, #2850A
            params[3]**2, #e100
            params[4]**2, #alumina
            params[3]**2, #e100
            params[2]**2, #2850B
            params[3]**2, #e100
            params[1]**2, #1090B
            params[5]**2] #LN2
        dz= [np.inf,#air
            params[6],  #1090A
            params[8],  #e100
            params[9],  #2850A
            params[8],  #e100
            10e-3,      #alumina
            params[8],  #e100
            params[10], #2850B
            params[8],  #e100
            params[7],  #1090B
            np.inf]     #LN2
        mu = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,params[0],eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0_TE = model(fs, params0, 'TE')
    Rs0_TM = model(fs, params0, 'TM')

    fisher_TE = np.full((len(params0),len(params0)), np.nan)
    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-12)
        Rsi_TE = model(fs, paramsi, 'TE')
        Rsi_TM = model(fs, paramsi, 'TM')
        dRsi_TE= (Rsi_TE - Rs0_TE)/abs(params0[ii]*1e-12)
        dRsi_TM= (Rsi_TM - Rs0_TM)/abs(params0[ii]*1e-12)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-12)
            Rsj_TE = model(fs, paramsj, 'TE')
            Rsj_TM = model(fs, paramsj, 'TM')
            dRsj_TE= (Rsj_TE - Rs0_TE)/abs(params0[jj]*1e-12)
            dRsj_TM= (Rsj_TM - Rs0_TM)/abs(params0[jj]*1e-12)

            fisher_TE[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE)
            fisher[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE + dRsi_TM * dRsj_TM)
            if not ii==jj:
                fisher_TE[jj,ii] = fisher_TE[ii,jj]
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher_TE=np.linalg.pinv(fisher_TE)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))

    
    mx_TE=gaussian_mixtures.GaussianND(params0,inv_fisher_TE,names=params_name)
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    
    g.triangle_plot([mx_TE, mx],
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, TE'%(noise), 'noise=%.1e, TE+TM'%(noise)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue','red'],
      line_args = [{'lw':2, 'color':'darkblue'},{'lw':2, 'color':'red'}]
      )
    g.export('testtri_noisep%.0E_TETM.png'%(noise))
   
    return params0, params_name, inv_fisher

def fisher1_barealumina(noise=0.01, polar='TE'):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    #params_name = ['theta', 'n_almina', 'n_ln2']
    params_name = ['theta', 'n_almina', 'n_ln2']
    params0 = [45,  #0, theta
        n_alumina,  #1,
        n_ln2]       #2, n_LN2
        #10e-3       #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[1]**2, #alumina
            params[2]**2] #LN2
        dz= [np.inf,#air
            10e-3,      #alumina
            np.inf]     #LN2
        mu = [1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,params[0],eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0 = model(fs, params0, polar)

    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-12)
        Rsi = model(fs, paramsi, polar)
        dRsi= (Rsi - Rs0)/abs(params0[ii]*1e-12)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-12)
            Rsj = model(fs, paramsj, polar)
            dRsj= (Rsj - Rs0)/abs(params0[jj]*1e-12)

            fisher[ii,jj] = noise**-2 * np.sum(dRsi * dRsj)
            if not ii==jj:
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))

    
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    g.triangle_plot(mx,
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, %s'%(noise,polar)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue'],
      line_args = [{'lw':2, 'color':'darkblue'}]
      )
    g.export('testtri_barealumina_noisep%.0E_%s.png'%(noise,polar))


    return params0, params_name, inv_fisher
 
def fisher2polar_barealumina(noise=0.01):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['theta', 'n_almina', 'n_ln2']
    params0 = [45,  #0, theta
        n_alumina,  #1,
        n_ln2]       #2, n_LN2
        #10e-3       #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[1]**2, #alumina
            params[2]**2] #LN2
        dz= [np.inf,#air
            10e-3,      #alumina
            np.inf]     #LN2
        mu = [1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,params[0],eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0_TE = model(fs, params0, 'TE')
    Rs0_TM = model(fs, params0, 'TM')

    fisher_TE = np.full((len(params0),len(params0)), np.nan)
    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-12)
        Rsi_TE = model(fs, paramsi, 'TE')
        dRsi_TE= (Rsi_TE - Rs0_TE)/abs(params0[ii]*1e-12)
        Rsi_TM = model(fs, paramsi, 'TM')
        dRsi_TM= (Rsi_TM - Rs0_TM)/abs(params0[ii]*1e-12)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-12)
            Rsj_TE = model(fs, paramsj, 'TE')
            dRsj_TE= (Rsj_TE - Rs0_TE)/abs(params0[jj]*1e-12)
            Rsj_TM = model(fs, paramsj, 'TM')
            dRsj_TM= (Rsj_TM - Rs0_TM)/abs(params0[jj]*1e-12)

            fisher_TE[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE)
            fisher[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE + dRsi_TM * dRsj_TM)
            if not ii==jj:
                fisher_TE[jj,ii] = fisher_TE[ii,jj]
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher_TE=np.linalg.pinv(fisher_TE)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))

    
    mx_TE=gaussian_mixtures.GaussianND(params0,inv_fisher_TE,names=params_name)
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    
    g.triangle_plot([mx_TE, mx],
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, TE'%(noise), 'noise=%.1e, TE+TM'%(noise)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue','red'],
      line_args = [{'lw':2, 'color':'darkblue'},{'lw':2, 'color':'red'}]
      )
    g.export('testtri_barealumina_noisep%.0E_TETM.png'%(noise))


    return params0, params_name, inv_fisher
 
def fisher1_lesspar(noise=0.01, polar='TE'):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['n_1090', 'n_2850', 'n_e100', 'd_1090A','d_1090B','d_e100','d_2850A','d_2850B']
    params0 = [# theta
        n_1090,     #0,
        n_2850,     #1,
        n_e100,     #2,
        0.117e-3,   #3, d_1090A
        0.117e-3,   #4, d_1090B
        0.0254e-3,  #5, d_e100
        0.056e-3,   #6, d_2850A
        0.056e-3]   #7, d_2850B
        #10e-3      #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[0]**2, #1090A
            params[2]**2, #e100
            params[1]**2, #2850A
            params[2]**2, #e100
            n_alumina**2, #alumina
            params[2]**2, #e100
            params[1]**2, #2850B
            params[2]**2, #e100
            params[0]**2, #1090B
            n_ln2**2] #LN2
        dz= [np.inf,#air
            params[3],  #1090A
            params[5],  #e100
            params[6],  #2850A
            params[5],  #e100
            10e-3,      #alumina
            params[5],  #e100
            params[7], #2850B
            params[5],  #e100
            params[4],  #1090B
            np.inf]     #LN2
        mu = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,45,eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0 = model(fs, params0, polar)

    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-12)
        Rsi = model(fs, paramsi, polar)
        dRsi= (Rsi - Rs0)/abs(params0[ii]*1e-12)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-12)
            Rsj = model(fs, paramsj, polar)
            dRsj= (Rsj - Rs0)/abs(params0[jj]*1e-12)

            fisher[ii,jj] = noise**-2 * np.sum(dRsi * dRsj)
            if not ii==jj:
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))
 
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    g.triangle_plot(mx,
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, %s'%(noise,polar)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue'],
      line_args = [{'lw':2, 'color':'darkblue'}]
      )
    g.export('testtri_ba4coated_lesspar_noisep%.0E_%s.png'%(noise,polar))

    return params0, params_name, inv_fisher
    
def fisher2polar_lesspar(noise=0.01):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['n_1090', 'n_2850', 'n_e100', 'd_1090A','d_1090B','d_e100','d_2850A','d_2850B']
    params0 = [# theta
        n_1090,     #0,
        n_2850,     #1,
        n_e100,     #2,
        0.117e-3,   #3, d_1090A
        0.117e-3,   #4, d_1090B
        0.0254e-3,  #5, d_e100
        0.056e-3,   #6, d_2850A
        0.056e-3]   #7, d_2850B
        #10e-3      #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[0]**2, #1090A
            params[2]**2, #e100
            params[1]**2, #2850A
            params[2]**2, #e100
            n_alumina**2, #alumina
            params[2]**2, #e100
            params[1]**2, #2850B
            params[2]**2, #e100
            params[0]**2, #1090B
            n_ln2**2] #LN2
        dz= [np.inf,#air
            params[3],  #1090A
            params[5],  #e100
            params[6],  #2850A
            params[5],  #e100
            10e-3,      #alumina
            params[5],  #e100
            params[7], #2850B
            params[5],  #e100
            params[4],  #1090B
            np.inf]     #LN2
        mu = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,45,eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0_TE = model(fs, params0, 'TE')
    Rs0_TM = model(fs, params0, 'TM')

    fisher_TE = np.full((len(params0),len(params0)), np.nan)
    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-12)
        Rsi_TE = model(fs, paramsi, 'TE')
        Rsi_TM = model(fs, paramsi, 'TM')
        dRsi_TE= (Rsi_TE - Rs0_TE)/abs(params0[ii]*1e-12)
        dRsi_TM= (Rsi_TM - Rs0_TM)/abs(params0[ii]*1e-12)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-12)
            Rsj_TE = model(fs, paramsj, 'TE')
            Rsj_TM = model(fs, paramsj, 'TM')
            dRsj_TE= (Rsj_TE - Rs0_TE)/abs(params0[jj]*1e-12)
            dRsj_TM= (Rsj_TM - Rs0_TM)/abs(params0[jj]*1e-12)

            fisher_TE[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE)
            fisher[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE + dRsi_TM * dRsj_TM)
            if not ii==jj:
                fisher_TE[jj,ii] = fisher_TE[ii,jj]
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher_TE=np.linalg.pinv(fisher_TE)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))

    
    mx_TE=gaussian_mixtures.GaussianND(params0,inv_fisher_TE,names=params_name)
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    
    g.triangle_plot([mx_TE, mx],
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, TE'%(noise), 'noise=%.1e, TE+TM'%(noise)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue','red'],
      line_args = [{'lw':2, 'color':'darkblue'},{'lw':2, 'color':'red'}]
      )
    g.export('testtri_ba4coated_lesspar_noisep%.0E_TETM.png'%(noise))

   
    return params0, params_name, inv_fisher

def fisher1_lesspar2(noise=0.01,polar='TE'):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['n_1090', 'n_2850', 'n_e100']
    params0 = [# theta
        n_1090,     #0,
        n_2850,     #1,
        n_e100]     #2,
        #0.117e-3,   #3, d_1090A
        #0.0254e-3,  #4, d_e100
        #0.056e-3]   #5, d_2850A
        #10e-3      #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[0]**2, #1090A
            params[2]**2, #e100
            params[1]**2, #2850A
            params[2]**2, #e100
            n_alumina**2, #alumina
            params[2]**2, #e100
            params[1]**2, #2850B
            params[2]**2, #e100
            params[0]**2, #1090B
            n_ln2**2] #LN2
        dz= [np.inf,#air
            0.117e-3,   #1090A
            0.0254e-3,  #e100
            0.056e-3,   #2850A
            0.0254e-3,  #e100
            10e-3,      #alumina
            0.0254e-3,  #e100
            0.056e-3,   #2850B
            0.0254e-3,  #e100
            0.117e-3,   #1090B
            np.inf]     #LN2
        mu = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,45,eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0 = model(fs, params0, polar)

    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-6)
        Rsi = model(fs, paramsi, polar)
        dRsi= (Rsi - Rs0)/abs(params0[ii]*1e-6)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-6)
            Rsj = model(fs, paramsj, polar)
            dRsj= (Rsj - Rs0)/abs(params0[jj]*1e-6)

            fisher[ii,jj] = noise**-2 * np.sum(dRsi * dRsj)
            if not ii==jj:
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))

    
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    
    g.triangle_plot([mx],
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, %s'%(noise,polar)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue'],
      line_args = [{'lw':2, 'color':'darkblue'}]
      )
    g.export('testtri_ba4coated_lesspar2_noisep%.0E_%s.png'%(noise,polar))

   
    return params0, params_name, inv_fisher




def fisher2polar_lesspar2(noise=0.01):
# Let's try constrain everything all together with a measurement done at a fixed angle (45 for example)
    params_name = ['n_1090', 'n_2850', 'n_e100']
    params0 = [# theta
        n_1090,     #0,
        n_2850,     #1,
        n_e100]     #2,
        #0.117e-3,   #3, d_1090A
        #0.0254e-3,  #4, d_e100
        #0.056e-3]   #5, d_2850A
        #10e-3      #11, d_alumina

    def model(fs,params,polar):
        Rs = np.full_like(fs, np.nan)
        eps= [1,    #air
            params[0]**2, #1090A
            params[2]**2, #e100
            params[1]**2, #2850A
            params[2]**2, #e100
            n_alumina**2, #alumina
            params[2]**2, #e100
            params[1]**2, #2850B
            params[2]**2, #e100
            params[0]**2, #1090B
            n_ln2**2] #LN2
        dz= [np.inf,#air
            0.117e-3,   #1090A
            0.0254e-3,  #e100
            0.056e-3,   #2850A
            0.0254e-3,  #e100
            10e-3,      #alumina
            0.0254e-3,  #e100
            0.056e-3,   #2850B
            0.0254e-3,  #e100
            0.117e-3,   #1090B
            np.inf]     #LN2
        mu = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for ii in range(len(fs)):
            f=fs[ii]
            r,t,R,T=get_coeffs_RT(f,45,eps,mu,dz,polar)
            Rs[ii]=R
        
        return Rs

    fs = np.linspace(220e9,320e9,1000)
    Rs0_TE = model(fs, params0, 'TE')
    Rs0_TM = model(fs, params0, 'TM')

    fisher_TE = np.full((len(params0),len(params0)), np.nan)
    fisher = np.full((len(params0),len(params0)), np.nan)
    for ii in range(len(params0)):
        paramsi = params0.copy()
        paramsi[ii] = params0[ii]+abs(params0[ii]*1e-6)
        Rsi_TE = model(fs, paramsi, 'TE')
        Rsi_TM = model(fs, paramsi, 'TM')
        dRsi_TE= (Rsi_TE - Rs0_TE)/abs(params0[ii]*1e-6)
        dRsi_TM= (Rsi_TM - Rs0_TM)/abs(params0[ii]*1e-6)

        for jj in range(ii, len(params0)):
            paramsj = params0.copy()
            paramsj[jj] = params0[jj]+abs(params0[jj]*1e-6)
            Rsj_TE = model(fs, paramsj, 'TE')
            Rsj_TM = model(fs, paramsj, 'TM')
            dRsj_TE= (Rsj_TE - Rs0_TE)/abs(params0[jj]*1e-6)
            dRsj_TM= (Rsj_TM - Rs0_TM)/abs(params0[jj]*1e-6)

            fisher_TE[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE)
            fisher[ii,jj] = noise**-2 * np.sum(dRsi_TE * dRsj_TE + dRsi_TM * dRsj_TM)
            if not ii==jj:
                fisher_TE[jj,ii] = fisher_TE[ii,jj]
                fisher[jj,ii] = fisher[ii,jj]

    #print(fisher)
    inv_fisher_TE=np.linalg.pinv(fisher_TE)
    inv_fisher=np.linalg.pinv(fisher)
    
    for ipar in range(len(params0)):
        print(' %s = %.2e +- %.2e'%(params_name[ipar], params0[ipar], np.sqrt(inv_fisher[ipar,ipar])))

    
    mx_TE=gaussian_mixtures.GaussianND(params0,inv_fisher_TE,names=params_name)
    mx=gaussian_mixtures.GaussianND(params0,inv_fisher,names=params_name)
    
    g.triangle_plot([mx_TE, mx],
      params_name,
      filled = True,
      legend_labels = ['noise=%.1e, TE'%(noise), 'noise=%.1e, TE+TM'%(noise)],
      legend_loc = 'upper right',
      #title = 'bear alumina 10mm at air-ln2 interface',
      contour_colors = ['darkblue','red'],
      line_args = [{'lw':2, 'color':'darkblue'},{'lw':2, 'color':'red'}]
      )
    g.export('testtri_ba4coated_lesspar2_noisep%.0E_TETM.png'%(noise))

   
    return params0, params_name, inv_fisher

# example codes of making the triangle plots:
# from getdist import plots, gaussian_mixtures
# g = plots.get_subplot_plotter()
# par0,parnames, invF=dm.fisher1(0.01,'TE')
# mx=gaussian_mixtures.GaussianND(par0,invF,names=params_name)
# g.triangle_plot(mx,
#   params_name,
#   filled = True,
#   legend_labels = ['noise=0.01, TE'],
#   legend_loc = 'upper right',
#   contour_colors = ['darkblue'],
#   line_args = [{'lw':2, 'color':'darkblue'}]
#   )
# g.export('testtri.png')

#g = plots.get_subplot_plotter()
#g.settings.figure_legend_frame = True
#g.settings.legend_fontsize = 24
#g.settings.axes_labelsize = 24
#g.settings.axes_fontsize = 20
#g.settings.axis_tick_x_rotation = 45
#g.settings.axis_tick_y_rotation = 45
#g.settings.alpha_filled_add = 0.9
#g.settings.title_limit_fontsize = 1
