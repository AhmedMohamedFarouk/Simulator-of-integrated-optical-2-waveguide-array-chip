import numpy as np
from scipy.linalg import expm
import control
import matplotlib.pyplot as plt
class Two_waveguide_chip:
    def __init__(self, LTI_parameters = np.array([[[-8.55619048e+02, -5.28843537e+04], [1.0,  0.0]],
                                                  [[1.0], [0.0]],
                                                  [[  714.28571429, 49931.97278912]],
                                                  [[0.0]]])
                 
                 , light_wavelength = 808 * 10**(-9), chip_length = (3.6) * 10**(-2),
                 n0 = 2.1753, dn = 5 * 10**(-6), c0 = 100.0, dc1 = 1.5, dc2 = -1.3):
        self.lampda = light_wavelength     # lampda is the input laser wavelength
        self.l = chip_length               # l is the length of the chip
        self.n0 = n0                       # n0 is the intrinsic refractive index of the chip
        self.dn = dn                       # dn is the dynamical proportionality constant that determines how much the propagation constant changes by
                                           # changing the voltage across the waveguide v1
        self.c0 = c0                       # C0 is the intrinsic coupling between two adjacent waveguides 
       
        self.dc1 = dc1                     # dC1 and dC2 are dynamical proportionality constants that determine the amount of change of 
                                           # the coupling between waveguides by changing the voltages across them 
        self.dc2 = dc2
        self.A = LTI_parameters[0]
        self.B = LTI_parameters[1]
        self.C = LTI_parameters[2]
        self.D = LTI_parameters[3]
        
    def set_Hamiltonian_parameters(self):
        #print(v1, v2)
        # bi is the propagation constant across the ith waveguide 
        self.b1 = ((2* np.pi) / self.lampda) * (self.n0 + self.dn * self.v1)
        self.b2 = ((2*np.pi) / self.lampda) * (self.n0 + self.dn * self.v2)
        self.v12 = self.v2 - self.v1 # vij is the potential difference across the substrate between the two waveguides i and j
        self.c12 = self.c0 + self.dc1 * self.v12 + self.dc2 * (self.v1 + self.v2) # cij is the coupling coefficient between waveguide i and waveguide j (Cij = C0 + ∆C1∆Vi,j + ∆C2 (∆Vi + ∆Vj ))
        #print(b1, b2, c12)
        
    def construct_Hamiltonian(self):
        self.set_Hamiltonian_parameters()
        self.H = np.array([[self.b1, self.c12], [self.c12, self.b2]])   # H is the chip Hamiltonian which is described by the tridiagonal real-valued matrix
        #print(H)
        
    def evolve_the_unitary(self):
        self.construct_Hamiltonian()
        self.U = expm(((-1j*self.l))*self.H)  # the unitary evolution matrix
        #print(U)
     
    def chip_electrical_response(self, Input_Voltage, Time_range):
        Actual_voltage = []
        LTI_of_chip_electrical_response = control.ss(self.A, self.B, self.C, self.D)
        self.Input_voltage = np.array(Input_Voltage)
        self.Input_voltage = self.Input_voltage.transpose() 
        for Electrode in self.Input_voltage:
            t, V_electrode = control.forced_response(LTI_of_chip_electrical_response, Time_range, Electrode)
            Actual_voltage.append(V_electrode)
        Actual_voltage = np.array(Actual_voltage)
        return Actual_voltage.transpose() 
    
    def power_distribution_for_single_time_instant(self, PSI_initial):
        self.PSI_initial = PSI_initial  # PSI_initial is the initial input state
        self.evolve_the_unitary()
        self.PSI_final = np.dot(self.U , self.PSI_initial)
        #print(PSI_final)
        self.power_distribution_single_time_instant = np.square(np.absolute(self.PSI_final))
        return self.power_distribution_single_time_instant
        #print(Power_Distribution)
        
    
    def power_distribution(self, PSI_initial, Input_Voltage, Time_range):
        PSI_init = np.array(PSI_initial)
        PSI_init = PSI_init.transpose()
        self.Power_distribution = []
        self.Actual_voltage = np.array(self.chip_electrical_response(Input_Voltage, Time_range))
        for input_voltage_for_single_instant in self.Actual_voltage:
            self.v1 = input_voltage_for_single_instant[0]
            self.v2 = input_voltage_for_single_instant[1]
            #self.power_distribution_for_single_time_instant(PSI_initial)
            self.Power_distribution.append((self.power_distribution_for_single_time_instant(PSI_init)).transpose())
            #print(input_voltage_for_single_instant)
        self.Power_distribution = np.array(self.Power_distribution)
        return self.Power_distribution
    
   
        
