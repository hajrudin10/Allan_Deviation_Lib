import pandas as pd
import numpy as np
import allantools
import mpmath as mp

#***************Calls that converts the freqcounter csv into allan deviation******************
class freq_count_to_AD:
    def __init__(self, filename, sampling_rate, k):
        self.sampling_rate = sampling_rate
        data = pd.read_csv(filename, delimiter=",", header=1, names=["Amplitude"], engine='python')
        self.raw_data = data["Amplitude"].to_numpy()
        self.raw_time = np.arange(0,len(self.raw_data))/self.sampling_rate
        self.tau = ( np.exp( np.log(len(self.raw_data)/2)/k )**range(k) )/sampling_rate
    
    def allan_dev(self):
        (self.allan_tau, self.allan_val_raw, _, _) = allantools.oadev(self.raw_data, rate=self.sampling_rate, data_type="freq", taus=self.tau)
        self.allan_val = self.allan_val_raw/np.mean(self.raw_data)
    
    def save_data(self,filename):
        allan_df =pd.DataFrame({'tau':self.allan_tau, 'adev':self.allan_val})
        allan_df.to_csv(filename+"_Allan_dev.csv")
     
        
#***************Calls that converts the ZI txt into allan deviation driven as a PLL******************     
class ZI_to_AD:
    def __init__(self, filename, k):
        data = pd.read_csv(filename, delimiter="; ", header=5, names=["Time", "Amplitude"], engine='python')
        self.raw_data = data["Amplitude"].to_numpy()
        self.raw_time = data["Time"].to_numpy()
        self.sampling_rate = 1/abs(data["Time"][1]-data["Time"][0])
        self.tau = ( np.exp( np.log(len(self.raw_data)/2)/k )**range(k) )/self.sampling_rate
    
    def allan_dev(self, data_type,f_0):
        if data_type=="freq":
            (self.allan_tau, self.allan_val_raw, _, _) = allantools.oadev(self.raw_data, rate=self.sampling_rate, data_type="freq", taus=self.tau)
            self.allan_val = self.allan_val_raw/np.mean(self.raw_data)
        elif data_type=="phase":
            self.freq_data = np.gradient(np.unwrap(np.deg2rad(self.raw_data)), 1/self.sampling_rate)
            (self.allan_tau, self.allan_val_raw, _, _) = allantools.oadev(self.freq_data, rate=self.sampling_rate, data_type="freq", taus=self.tau)
            self.allan_val = self.allan_val_raw/(np.mean(self.raw_data)+f_0)
        else:
            print("False datatype")
    
    def save_data(self,filename):
        allan_df =pd.DataFrame({'tau':self.allan_tau, 'adev':self.allan_val})
        allan_df.to_csv(filename+"_Allan_dev.csv")
        
#***************Calls that calculates the analytical solution for the allan deviation****************** 
class analytical:
    #constructor with iinput parameters****************************
    def __init__(self, Q, omega_r,tau_l, K_d, K_p, K_i, tau, S_th = 1):
        self.Q = Q
        self.omega_r= omega_r
        self.tau_l = tau_l
        self.K_d = K_d
        self.K_p = K_p
        self.K_i = K_i
        self.S_th = S_th
        self.tau = tau
        self.tau_r = mp.mpf(2*self.Q/self.omega_r) 
    
    #Transfer functions***********************************
    def H_R(self, s):
        return 1/(1+s*self.tau_r)

    def H_L(self, s):
        return 1/(1+s*self.tau_l)

    #TF for Feedbackfree drive
    def H_th_FF(self, s):
        return self.H_R(s)*self.H_L(s)/self.tau_r

    def H_d_FF(self, s):
        return self.H_L(s)/self.tau_r

    #TF for PLL
    def H_th_PLL(self, s):
        nominator = (s*self.K_p+self.K_i)*self.H_L(s)
        denominator = s**2+(s/self.tau_r)+nominator
        return (1/self.tau_r)*nominator/denominator

    def H_d_PLL(self,s):
        nominator = (s*self.K_p+self.K_i)*self.H_L(s)
        denominator = s**2+(s/self.tau_r)+nominator
        return (1/(self.tau_r*self.H_R(s)))*nominator/denominator

    #TF for SSO
    def H_th_SSO(self,s):
        return self.H_L(s)/self.tau_r

    def H_d_SSO(self,s):
        return self.H_L(s)/self.tau_r
    #Power spectral density***************************************
    def S_y(self,omega, H_th, H_d):
        d_therm=0
        return self.S_th*(np.absolute(H_th(1j*omega))**2+np.absolute(H_d(1j*omega))**2*self.K_d**2+d_therm)/self.omega_r**2
    
    #Definition of the functions to be integrated*****************
    def integrand_noiseless(self,omega, tau):
        return ((mp.sin(omega*tau/2)**4)/(omega**2))

    def integrand_FF(self,omega, tau):
        return self.integrand_noiseless(omega, tau)*self.S_y(omega,self.H_th_FF,self.H_d_FF)

    def integrand_PLL(self,omega, tau):
        return self.integrand_noiseless(omega, tau)*self.S_y(omega,self.H_th_PLL,self.H_d_PLL)

    def integrand_SSO(self,omega, tau):
        return self.integrand_noiseless(omega, tau)*self.S_y(omega,self.H_th_SSO,self.H_d_SSO)
    
    #functions that calculate the Allan deviation*****************
    #Feedback free drive
    def calculate_FF(self):
        self.sigma_FF = np.zeros(len(self.tau))
        for i, tau_val in enumerate(self.tau):
            self.sigma_FF[i] = mp.sqrt((4/(mp.pi*tau_val**2)) * mp.quad(lambda x: self.integrand_FF(x,tau_val),[0, mp.inf] ))
    
    #PLL drive        
    def calculate_PLL(self):
        self.sigma_PLL = np.zeros(len(self.tau))
        for i, tau_val in enumerate(self.tau):
            self.sigma_PLL[i] = mp.sqrt((4/(mp.pi*tau_val**2)) * mp.quad(lambda x: self.integrand_PLL(x,tau_val),[0, mp.inf] ))
    
    #Self sustaining oscillator
    def calculate_SSO(self):
        self.sigma_SSO = np.zeros(len(self.tau))
        for i, tau_val in enumerate(self.tau):
            self.sigma_SSO[i] = mp.sqrt((4/(mp.pi*tau_val**2)) * mp.quad(lambda x: self.integrand_SSO(x,tau_val),[0, mp.inf] ))

    def save_PLL(self,filename):
        allan_df =pd.DataFrame({'tau':self.tau, 'adev':self.sigma_PLL})
        allan_df.to_csv(filename+"_Allan_dev.csv")
