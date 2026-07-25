# Scrip to solve the 1D heat equation by Crank-Nicolson

#Import libraries

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from joblib import Parallel, delayed, cpu_count

# Dictionary of metals and thermal diffusivity 
metal_diffusivities = {
                    "Copper": 111, "Iron": 23, "Aluminum": 97, "Brass": 34, 
                    "Steel": 18, "Zinc": 42, "Lead": 22, "Titanium": 9.8,
                    "Silver": 165, "Gold": 127, "Nickel": 23, "Tungsten": 67,
                    "Magnesium": 88, "Tin": 40, "Platinum": 25, "Cobalt": 26}


class heatDiffusion:
    def __init__(self, metal, h=0.1, k=0.1, tol=1e-3):
        '''
        Define the heat diffusion class.        
        metal (str): The type of metal to analyze.
        h (float): The spatial step size in cm (default: 0.1).
        k (float): The time step size in seconds (default: 0.1).
        tol (float): The tolerance for thermal equilibrium (default: 1e-3). 
        '''
    
        # Error validation
        if metal not in metal_diffusivities:
            raise ValueError (f"Invalid metal:'{metal}'")
        # Define inputs
        self.metal = metal
        self.alpha = metal_diffusivities[metal] * 0.01 # convet in cm^2/s
        self.h = h
        self.k = k
        self.tol = tol
        self.l = 10.0 #Length of the metal cm
        self.max_steps = 100000

    def initialization_routine(self):
        '''
        Define the spatial grid, initial temperature distribution, 
        and BCs.        
        x (numpy array): Spatial grid points.
        T_o (numpy array): Initial temperature.
        n (int): Number of spatial points.
        T_b (float): Boundary temperature.
        '''
        # Define space vector
        x = np.arange(-self.l, self.l + self.h, self.h)
        # Number of points inside to the bar
        n = len(x) 
        # Set the initial condition 
        T_o = 175 - 50 * np.cos((np.pi * x) / 5) - x**2
        # Set boundary conditions
        T_b = 25
        T_o[0] = T_b
        T_o[-1] = T_b

        return x , T_o, n, T_b

    def crank_nicolson(self):
        '''
        Define the Crank-Nicolson discretization method.
        x (numpy array): Spatial grid points.
        T (numpy array): Temperature distribution over time.
        t (list): Time points.
        thermal_t (float): Time to reach thermal equilibrium in minutes.
        '''
        # Call the init function
        x, T_o, n, T_b = self.initialization_routine()
        # Define r factor             
        r_factor = (self.alpha * self.k) / (self.h**2)

        # Define the implicit matrix
        D1_matrix_0 = np.diag([2 + 2*r_factor] * (n - 2), 0)
        D1_matrix_n = np.diag([-r_factor] * (n - 3), -1)
        D1_matrix_p = np.diag([-r_factor] * (n - 3), +1)
        D1_matrix = D1_matrix_0 + D1_matrix_n + D1_matrix_p

        # Define the explicit matrix
        D2_matrix_0 = np.diag([2 - 2*r_factor] * (n - 2), 0)
        D2_matrix_n = np.diag([r_factor] * (n - 3), -1)
        D2_matrix_p = np.diag([r_factor] * (n - 3), +1)
        D2_matrix = D2_matrix_0 + D2_matrix_n + D2_matrix_p

        # Define a T matrix with ICs/BC
        T = np.zeros((len(x), self.max_steps))
        T[:, 0] = T_o # ICs
        T[0, :] = T_b # BC
        T[-1, :] = T_b # BC

        t = [0]
        thermal_t = 0
        equilibrium = False
        j = 0
        # Run the loop until the approach to the thermal equilibrium
        while not equilibrium and j < self.max_steps:
            
            b = T[1:-1, j].copy()     
            
            b = np.dot(D2_matrix, b)
            b[0] = b[0] + r_factor * (T[0, j+1] + T[0, j])
            b[-1] = b[-1] + r_factor * (T[-1, j+1] + T[-1, j])        
            
            # Solve implicit matrix
            T[1:-1, j+1] = np.linalg.solve(D1_matrix, b)           
            
            # Equilibrium check
            if np.max(np.abs(T[:, j+1] - T[:, j])) < self.tol:  
                thermal_t = t[-1] 
                equilibrium = True
                #print(f'Thermal Equilibrium at t = {thermal_t:.3f} [s]')

            # Store into the t list  
            t.append(t[-1] + self.k)
            j = j + 1

        # Update T matrix
        T = T[:, :j + 1]  

        return  x, T, t, thermal_t
    
def execution_rutine(metal):
    '''
    Define the execution routine for each metal.
    metal (str): The type of metal to analyze.
    '''

    # Call the function 
    _, _, _, thermal_t = heatDiffusion(metal).crank_nicolson()
    print({'metal': metal, 'thermal_t': round(thermal_t, 4)})
    return metal, thermal_t

if __name__ == "__main__":

    if len(sys.argv) > 1:
        n_cores = int(sys.argv[1])
    else:
        n_cores = 1

    list_metals = list(metal_diffusivities.keys())        
    print(f'Number of cores{n_cores}')   

    # Time stamp
    start_time = time.time()
    
    if n_cores == 1:
        results = [execution_rutine(m) for m in list_metals]
    else:
        results = Parallel(n_jobs=n_cores)(delayed(execution_rutine)(m) for m in list_metals)

    # Time stamp
    end_time = time.time()
    total_time = end_time - start_time        
    
    print(f'Execution time: {total_time:.4f}')      




        
        







    


    
    