#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 07:01:17 2018

@author: max
"""
import numpy as np
from scipy.interpolate import griddata, interp1d

class xsec():
    @staticmethod
    def calc_svione_st(ne,Te):
        """Calculates the ionization cross section using stacey thomas data
    
        Args:
            ne (float): takes ne in m^-3
            Te (float): takes Te in keV
    
        Returns:
            svione (float): 
            
        """
        #TODO: svione isn't actually a function of ne. Not sure why Mandrekas wrote it this way.
        #      We can simplify this function signficantly.
        if isinstance(ne, float):
            ne = np.array([ne])    
        if isinstance(Te, float):
            Te = np.array([Te])

        orig_shape = Te.shape     
        
        logne = np.log10(ne).flatten()
        logTe = np.log10(Te*1E3).flatten()
        
        
        #fix values that are outside the interpolation area
        logne[logne>22] = 22
        logne[logne<16] = 16
        logTe[logTe>3]  = 3
        logTe[logTe<-1] = -1       
        
        znint, tint,  = np.meshgrid(np.array([16,18,20,21,22]), np.array([-1,0,1,2,3]))        
        
        eion = np.array([[-2.8523E+01, -2.8523E+01, -2.8523E+01, -2.8523E+01, -2.8523E+01],
                         [-1.7745E+01, -1.7745E+01, -1.7745E+01, -1.7745E+01, -1.7745E+01],
                         [-1.3620E+01, -1.3620E+01, -1.3620E+01, -1.3620E+01, -1.3620E+01],
                         [-1.3097E+01, -1.3097E+01, -1.3097E+01, -1.3097E+01, -1.3097E+01],
                         [-1.3301E+01, -1.3301E+01, -1.3301E+01, -1.3301E+01, -1.3301E+01]])        
        
        svione = griddata(np.column_stack((znint.flatten(),tint.flatten())),eion.flatten(), np.column_stack((logne,logTe)), method='linear', rescale=False)
        svione  = np.reshape(svione,orig_shape)
        return 10**svione
    
    @staticmethod
    def calc_svione_janev (Te):
        """calculates the electron impact ionization reactivity (<sigma*v>)
        
        Args:
            Te (float): Electron temperature in eV
            
        Returns:
            svione (float): electron impact ionization rate (m^3/s)
        
        This function evaluates the electron impact ionization 
        reactivity (<sigma*v>) as a function of the electron temperature
        using the logarithmic polynomial approximation of Janev, et al.
    
        Reference:
        ---------
        R.K. Janev et al. 'Elementary Processes in Hydrogen-Helium Plasmas,
        Springer-Verlag, 1987 
    
        e + H(1s) --> e + H+ + e (Reaction 2.1.5)
        Notice that only six significant digits have been kept, per
        Janev's suggestion (page 256)
    
        Originally written by John Mandrekas for the GTNEUT code.
        Adapted for python by Max Hill
    
        Comments:
            --------
        The electron temperature range is: 0.1 eV <= te < 20 keV
        """
        
        if isinstance(Te, float):
            Te = np.array([Te])
        
        #convert from kev to ev
        Te = Te*1E3
        
        bn = np.array([-3.271396786375e+01,  1.353655609057e+01, -5.739328757388e+00,
                        1.563154982022e+00, -2.877056004391e-01,  3.482559773737e-02,
                       -2.631976175590e-03,  1.119543953861e-04, -2.039149852002e-06])
    
        Te[Te<1.0E-1] = 1.0E-1
        Te[Te>2.0E4]  = 2.0E4
        logTe = np.log(Te)
        svione = np.zeros(Te.shape)
        for i,lT in enumerate(logTe):
            for j,coef in enumerate(bn):
                svione[i] = svione[i] + coef * lT**j

        return np.exp(svione)*1E-6

    @staticmethod
    def calc_svioni_janev (Ti,Tn,aion,aneut):
        """
        This function evaluates the ion impact ionization reactivity
        (<sigma*v>) as a function of the ion temperature and the neutral
        energy using the double logarithmic polynomial approximation of 
        Janev, et al.
    
        Reference:
            ---------
            R.K. Janev et al. 'Elementary Processes in Hydrogen-Helium Plasmas,
            Springer-Verlag, 1987 
    
        p + H(1s) --> p + H+ + e (Reaction 3.1.6)
        
        Written by John Mandrekas, GIT, 10/05/2001 for the GTNEUT code
    
        Parameters:
        ----------
        aion  : ion mass (1 for H, 2 for D, etc.)
        ti    : ion temperature (eV)
        aneut : neutral mass (1 for H, 2 for D, etc.)
        e0    : ion temperature (eV)
    
        Output:
        ------
        svioni : ion impact ionization reaction rate (m^3/s)
    
        Comments:
        --------
        The ion temperature  and neutral energy ranges are: 
        0.1 eV <= Ti, E0 <= 20 keV
        Energies and temperatures are scaled by mass ratios
        """
        
        if isinstance(Ti, float):
            Ti = np.array([Ti])
        if isinstance(Tn, float):
            Tn = np.array([Tn])
            
        #convert from kev to ev
        Ti = Ti*1E3
        Tn = Tn*1E3
                    
        an = np.array([-1.617454916209e+02,  1.021458246570e+02, -5.712267930902e+01,
                        2.140540272484e+01, -4.767517412803e+00,  6.293295208376e-01,
                       -4.858173640838e-02,  2.031177914273e-03, -3.557982934756e-05,
                        1.767238902030e+01, -7.102574692619e+01,  4.246688953154e+01,
                       -1.128638171243e+01,  1.661679851896e+00, -1.476754423056e-01,
                        8.175790218529e-03, -2.732531913524e-04,  4.398387454014e-06,
                       -4.334843983767e+01,  3.855259623260e+01, -1.316883030631e+01,
                        2.145592145856e+00, -1.467281287038e-01, -2.915256218527e-03,
                        1.092542891192e-03, -6.205102802216e-05,  1.158798945435e-06,
                        2.464254915383e+01, -1.283426276878e+01,  2.369698902002e+00,
                       -1.506665823159e-01, -8.144926683660e-03,  2.231505500086e-03,
                       -2.210941355372e-04,  1.310924337643e-05, -3.431837053957e-07,
                       -5.439093405254e+00,  2.357085001656e+00, -2.961732508220e-01,
                       -9.917174972226e-04,  1.935894665907e-03, -1.679264493005e-05,
                        5.532386419162e-08, -1.121430499351e-06,  5.960280736984e-08,
                        5.959975304236e-01, -2.391382925527e-01,  2.789277301925e-02,
                        8.562387824450e-05, -1.340759667335e-04, -5.927455645560e-06,
                        5.820264508685e-07,  7.694068657107e-08, -4.972708712807e-09,
                       -3.361958123977e-02,  1.289667246580e-02, -1.858739201548e-03,
                        9.235982885753e-05,  9.875232214392e-06, -1.680823118052e-06,
                        3.019916624608e-08,  6.889325889968e-09, -3.171970185702e-10,
                        8.706597041685e-04, -3.140899683782e-04,  7.343984485463e-05,
                       -8.601564864429e-06, -6.467790579320e-07,  1.734797315767e-07,
                        2.523651535182e-09, -1.719633613108e-09,  7.332933714195e-11,
                       -6.359765062372e-06,  1.742836004704e-06, -1.235536456998e-06,
                        2.257852760280e-07,  1.608335682237e-08, -3.855914336143e-09,
                       -3.556222618473e-10,  7.627265694554e-11, -2.960493966948e-12]).reshape((9,9))
    
        #scale with isotope mass
        Ti = Ti / aion
        Tn = Tn / aneut
        
        #Make sure we are within limits of validity
        Ti[Ti<1.0E-1] = 1.0E-1
        Ti[Ti>2.0E4]  = 2.0E4
        Tn[Tn<1.0E-1] = 1.0E-1
        Tn[Tn>2.0E4]  = 2.0E4
    
        logTi = np.log(Ti)
        logTn = np.log(Tn)
    
        svioni = np.zeros(Ti.shape)
        for i,(lTi,lTn) in enumerate(zip(logTi,logTn)):
            for (ai,aj),coef in np.ndenumerate(an):
                svioni[i] = svioni[i] + coef * lTi**aj * lTn**ai
        
        return np.exp(svioni)*1E-6
        
    @staticmethod
    def calc_svrec_st(ne, Te):
        """Calculates the

        Args:
            ne (float): takes ne in m^-3
            Te (float): takes Te in keV

        Returns:
            svrec (float): 

        """
        orig_shape = Te.shape      
    
        logne = np.log10(ne)
        logTe = np.log10(Te*1000)
        
        #fix values that are outside the interpolation area
        logne[logne>22]  =  22
        logne[logne<16] = 16
        logTe[logTe>3]  =  3
        logTe[logTe<-1]  =  -1   

        znint, tint,  = np.meshgrid(np.array([16,18,20,21,22]), np.array([-1,0,1,2,3]))
        
        rec = np.array([[-1.7523E+01, -1.6745E+01, -1.5155E+01, -1.4222E+01, -1.3301E+01],
                        [-1.8409E+01, -1.8398E+01, -1.8398E+01, -1.7886E+01, -1.7000E+01],
                        [-1.9398E+01, -1.9398E+01, -1.9398E+01, -1.9398E+01, -1.9398E+01],
                        [-2.0155E+01, -2.0155E+01, -2.0155E+01, -2.0155E+01, -2.0155E+01],
                        [-2.1000E+01, -2.1000E+01, -2.1000E+01, -2.1000E+01, -2.1000E+01]])
                        
        svrec = griddata(np.column_stack((znint.flatten(),tint.flatten())), rec.flatten(), np.column_stack((logne,logTe)), method='linear', rescale=False)
        svrec = np.reshape(svrec,orig_shape)
        return 10**svrec
        
    @staticmethod
    def calc_svel(Ti,Tn):
        """Calculates the
    
        Args:
            Ti (float): takes Ti in keV
            Tn (float): takes Tn in keV
    
        Returns:
            svel (float): 
            
        """
        if isinstance(Ti, float):
            Ti = np.array([Ti])    
        if isinstance(Tn, float):
            Tn = np.array([Tn])
            
        orig_shape = Ti.shape     
        
        
        logTi = np.log10(Ti*1000).flatten()
        logTn = np.log10(Tn*1000).flatten()    
        
        #fix values that are outside the interpolation area
        logTi[logTi>3]  =  3
        logTi[logTi<-1] = -1
        logTn[logTn>2]  =  2
        logTn[logTn<0]  =  0
    
        tint, tnnt  = np.meshgrid(np.array([-1,0,1,2,3]), np.array([0,1,2]))
        elast = np.array([[-1.3569E+01, -1.3337E+01, -1.3036E+01, -1.3569E+01, -1.3337E+01],
                          [-1.3036E+01, -1.3337E+01, -1.3167E+01, -1.3046E+01, -1.3036E+01],
                          [-1.3046E+01, -1.2796E+01, -1.3036E+01, -1.3046E+01, -1.2796E+01]])
        svel = griddata(np.column_stack((tint.flatten(), tnnt.flatten())), elast.flatten(), np.column_stack((logTi,logTn)), method='linear', rescale=False)
        svel = np.reshape(svel,orig_shape)
        return 10**svel
    
    @staticmethod
    def calc_sveln(Tn):
        """Calculates the
    
        Args:
            Tn (float): takes Tn in keV
    
        Returns:
            sveln (float): 
        """
        if isinstance(Tn, float):
            Tn = np.array([Tn])
            
        logTn = np.log10(Tn*1000)
        
        logTn[logTn>2]  =  2
        logTn[logTn<0]  =  0
        
        f = interp1d([0, 1, 2], [-1.4569E+01, -1.4167E+01, -1.3796E+01])
        sveln = f(logTn)
        return 10**sveln
    
    @staticmethod
    def calc_svcxi_st(Ti,Tn):
        """
        Calculates the
    
        Args:
            Ti (float): takes Ti in keV
            Tn (float): takes Tn in keV
    
        Returns:
            svcx (float): 
        """  
        
        if isinstance(Ti, float):
            Ti = np.array([Ti])    

            
        orig_shape = Ti.shape     
        
        logTi = np.log10(Ti*1E3).flatten()
        logTn = np.log10(Tn*1E3).flatten()
        
        #fix values that are outside the interpolation area
        logTi[logTi>3]  =  3
        logTi[logTi<-1] = -1
        logTn[logTn>2]  =  2
        logTn[logTn<0]  =  0
    
        tint, tnnt  = np.meshgrid(np.array([-1,0,1,2,3]), np.array([0,1,2]))
        cx = np.array([[-1.4097E+01, -1.3921E+01, -1.3553E+01, -1.4097E+01, -1.3921E+01],
                       [-1.3553E+01, -1.3921E+01, -1.3824E+01, -1.3538E+01, -1.3553E+01],
                       [-1.3538E+01, -1.3432E+01, -1.3553E+01, -1.3538E+01, -1.3432E+01]])
        svcxi = griddata(np.column_stack((tint.flatten(), tnnt.flatten())), cx.flatten(), np.column_stack((logTi,logTn)), method='linear', rescale=False)
        svcxi = np.reshape(svcxi,orig_shape)
        return 10**svcxi

    @staticmethod
    def calc_svcxi_janev (Ti,Tn,aion,aneut):
        """
        This function evaluates the ion impact ionization reactivity
        (<sigma*v>) as a function of the ion temperature and the neutral
        energy using the double logarithmic polynomial approximation of 
        Janev, et al.
    
        Reference:
            ---------
            R.K. Janev et al. 'Elementary Processes in Hydrogen-Helium Plasmas,
            Springer-Verlag, 1987 
    
        p + H(1s) --> p + H+ + e (Reaction 3.1.6)
        
        Written by John Mandrekas, GIT, 10/05/2001 for the GTNEUT code
    
        Parameters:
        ----------
        aion  : ion mass (1 for H, 2 for D, etc.)
        ti    : ion temperature (eV)
        aneut : neutral mass (1 for H, 2 for D, etc.)
        e0    : ion temperature (eV)
    
        Output:
        ------
        svioni : ion impact ionization reaction rate (m^3/s)
    
        Comments:
        --------
        The ion temperature  and neutral energy ranges are: 
        0.1 eV <= Ti, E0 <= 20 keV
        Energies and temperatures are scaled by mass ratios
        """
        
        if isinstance(Ti, float):
            Ti = np.array([Ti])
        if isinstance(Tn, float):
            Tn = np.array([Tn])
        #convert from kev to ev
        Ti = Ti*1E3
        Tn = Tn*1E3
            
        an = np.array([-1.829079581680e+01,  2.169137615703e-01,  4.307131243894e-02,
                       -5.754895093075e-04, -1.552077120204e-03, -1.876800283030e-04,
                        1.125490270962e-04, -1.238982763007e-05,  4.163596197181e-07,
                        1.640252721210e-01, -1.106722014459e-01,  8.948693624917e-03,
                        6.062141761233e-03, -1.210431587568e-03, -4.052878751584e-05,
                        2.875900435895e-05, -2.616998139678e-06,  7.558092849125e-08,
                        3.364564509137e-02, -1.382158680424e-03, -1.209480567154e-02,
                        1.075907881928e-03,  8.297212635856e-04, -1.907025662962e-04,
                        1.338839628570e-05, -1.171762874107e-07, -1.328404104165e-08,
                        9.530225559189e-03,  7.348786286628e-03, -3.675019470470e-04,
                       -8.119301728339e-04,  1.361661816974e-04,  1.141663041636e-05,
                       -4.340802793033e-06,  3.517971869029e-07, -9.170850253981e-09,
                       -8.519413589968e-04, -6.343059502294e-04,  1.039643390686e-03,
                        8.911036876068e-06, -1.008928628425e-04,  1.775681984457e-05,
                       -7.003521917385e-07, -4.928692832866e-08,  3.208853883734e-09,
                       -1.247583860943e-03, -1.919569450380e-04, -1.553840717902e-04,
                        3.175388949811e-05,  1.080693990468e-05, -3.149286923815e-06,
                        2.318308730487e-07,  1.756388998863e-10, -3.952740758950e-10,
                        3.014307545716e-04,  4.075019351738e-05,  2.670827249272e-06,
                       -4.515123641755e-06,  5.106059413591e-07,  3.105491554749e-08,
                       -6.030983538280e-09, -1.446756795654e-10,  2.739558475782e-11,
                       -2.499323170044e-05, -2.850044983009e-06,  7.695300597935e-07,
                        2.187439283954e-07, -1.299275586093e-07,  2.274394089017e-08,
                       -1.755944926274e-09,  7.143183138281e-11, -1.693040208927e-12,
                        6.932627237765e-07,  6.966822400446e-08, -3.783302281524e-08,
                       -2.911233951880e-09,  5.117133050290e-09, -1.130988250912e-09,
                        1.005189187279e-10, -3.989884105603e-12,  6.388219930167e-14]).reshape((9,9))
    
        #scale with isotope mass
        Ti = Ti / aion
        Tn = Tn / aneut
        
        #Make sure we are within limits of validity
        Ti[Ti<1.0E-1] = 1.0E-1
        Ti[Ti>2.0E4]  = 2.0E4
        Tn[Tn<1.0E-1] = 1.0E-1
        Tn[Tn>2.0E4]  = 2.0E4
    
        logTi = np.log(Ti)
        logTn = np.log(Tn)
        svcxi = np.zeros(Ti.shape)
        for i,(lTi,lTn) in enumerate(zip(logTi,logTn)):
            for (ai,aj),coef in np.ndenumerate(an):
                svcxi[i] = svcxi[i] + coef * lTi**aj * lTn**ai
        return np.exp(svcxi)*1E-6    