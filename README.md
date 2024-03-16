diele_layer_model.py - 

  theta_t = snell(n_i,theta_i,n_t): Calculate refraction angles with snell law.
  
  m = get_diele_matrix(f, theta, dz, eps_r, mu_r, polar='TE'): Calculate characteristic matrices of thin layers. For dielectric, mu_r=1, eps_r=n**2. theta is the angle in the layer.
  
  r,t,R,T = get_coeffs_RT(f,theta=90,eps=[1],mu=[1],dz=[np.inf],polar='TE'): Get reflectivity and transmittance of amplitude and intensity (power). theta is the first incident angle.
    For example, a layer with refraction index n and thickness d is put in air. Light at frequency f comes to the layer with an incident angle theta. One can get the reflectivity and transmittance in TE polarization with
    
      eps = [1, n**2, 1]
      mu  = [1, 1, 1]
      dz  = [np.inf, d, np.inf]
      r,t,R,T = get_coeffs_RT(f,theta,eps,mu,dz)

  rep_...(): A set of sanity checks to reproduce well known results.
  make_...(): Functions to make interesting plots.
  fisher1_...(): fisher analysis using spectrum at a single polarization as observables.
  fisher2polar_...(): fisher analysis using spectra from both polarizations.

  
