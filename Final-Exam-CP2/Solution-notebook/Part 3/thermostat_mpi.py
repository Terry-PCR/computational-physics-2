# !/usr/bin/python
"""
A molecular dynamics solver that simulates the motion of non-interacting particles
in the canonical ensemble using a Langevin thermostat.
Reference: https://github.com/Comp-science-engineering/Tutorials/tree/master/MolecularDynamics
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# Define global physical constants
from scipy.constants import Avogadro, Boltzmann

def wallHitCheck(pos, vels, box):
    """ This function enforces reflective boundary conditions.
    All particles that hit a wall  have their velocity updated
    in the opposite direction.
    @pos: atomic positions (ndarray)
    @vels: atomic velocity (ndarray, updated if collisions detected)
    @box: simulation box size (tuple)
    """
    ndims = len(box)

    for i in range(ndims):
        vels[((pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])),i] *= -1

def integrate(pos, vels, forces, mass,  dt):
    """ A simple forward Euler integrator that moves the system in time 
    @pos: atomic positions (ndarray, updated)
    @vels: atomic velocity (ndarray, updated)
    """
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T

def computeForce(mass, vels, temp, relax, dt):
    """ Computes the Langevin force for all particles
    @mass: particle mass (ndarray)
    @vels: particle velocities (ndarray)
    @temp: temperature (float)
    @relax: thermostat constant (float)
    @dt: simulation timestep (float)
    returns forces (ndarray)
    """
    natoms, ndims = vels.shape

    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T

    force = - (vels * mass[np.newaxis].T) / relax + noise

    return force

def run(**args):
    """ This is the main function that solves Langevin's equations for
    a system of natoms usinga forward Euler scheme, and returns an output
    list that stores the time and the temperture.
    @natoms (int): number of particles
    @temp (float): temperature (in Kelvin)
    @mass (float): particle mass (in Kg)
    @relax (float): relaxation constant (in seconds)
    @dt (float): simulation timestep (s)
    @nsteps (int): total number of steps the solver performs
    @box (tuple): simulation box size (in meters) of size dimensions x 2
    e.g. box = ((-1e-9, 1e-9), (-1e-9, 1e-9)) defines a 2D square
    @ofname (string): filename to write output to
    @freq (int): write output every 'freq' steps
    @[radius]: particle radius (for visualization)
    Returns a list (of size nsteps x 2) containing the time and temperature.
    
    """

    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps   = args['mass'], args['relax'], args['steps']
    ofname, freq, radius = args['ofname'], args['freq'], args['radius']

    dim = len(box)
    pos = np.random.rand(natoms,dim)

    for i in range(dim):
        pos[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * pos[:,i]

    vels = np.random.rand(natoms,dim)
    mass = np.ones(natoms) * mass / Avogadro
    radius = np.ones(natoms) * radius
    step = 0

    output = []

    while step <= nsteps:

        step += 1

        # Compute all forces
        forces = computeForce(mass, vels, temp, relax, dt)

        # Move the system in time
        integrate(pos, vels, forces, mass, dt)

        # Check if any particle has collided with the wall
        wallHitCheck(pos,vels,box)

        # Compute output (temperature)
        ins_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2)) / (Boltzmann * dim * natoms)
        output.append([step * dt, ins_temp])

        if not step%freq:
            #dump.writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
            writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
    return np.array(output)


def writeOutput(filename, natoms, timestep, box, **data):
    """ Writes the output (in dump format) """

    axis = ('x', 'y', 'z')

    with open(filename, 'a') as fp:

        fp.write('ITEM: TIMESTEP\n')
        fp.write('{}\n'.format(timestep))

        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write('{}\n'.format(natoms))

        fp.write('ITEM: BOX BOUNDS' + ' f' * len(box) + '\n')
        for box_bounds in box:
            fp.write('{} {}\n'.format(*box_bounds))

        for i in range(len(axis) - len(box)):
            fp.write('0 0\n')

        keys = list(data.keys())

        for key in keys:
            isMatrix = len(data[key].shape) > 1

            if isMatrix:
                _, nCols = data[key].shape

                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else:
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]

                del data[key]

        keys = data.keys()

        fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')

        output = []
        for key in keys:
            output = np.hstack((output, data[key]))

        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))

# -----------------------------------------------------------------------------
# Your MPI parallelization code should start here. Do not modify the code above.
# -----------------------------------------------------------------------------

# Run code wih desired parameters
# Import libraries 
from mpi4py import MPI
import csv
import os
# Define a list of temperatures to simulate in K
temperatures = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800]
N_atoms = 1000

if __name__ == '__main__':

    # a) Get basic information about MPI communicator
    # World communicator = pool of processors
    world_comm = MPI.COMM_WORLD   
    # This is the number of processors retrieved from -np xxxx
    world_size = world_comm.Get_size()
    # This is the ID of a specific processor
    my_rank = world_comm.Get_rank()

    # b) Distribute a range of temperatures among the available MPI
    # Point-to-Point communication
    # Determine the workload for each rank
    workloads = [len(temperatures) // world_size for i in range(world_size)]
    for i in range(len(temperatures) % world_size):
        workloads[i] += 1
    my_start = 0
    for i in range(my_rank):
        my_start += workloads[i]
    my_end = my_start + workloads[my_rank]
    my_temperatures = temperatures[my_start:my_end]
    # Initial parameters for the simulation
    params = {
        'natoms': N_atoms,
        'temp': 300,
        'mass': 0.001,
        'radius': 120e-12,
        'relax': 1e-13,
        'dt': 1e-15,
        'steps': 10000,
        'freq': 100,
        'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
        #'ofname': 'traj-hydrogen-3D-{N_atoms}-T{temp}.dump'.format(N_atoms, temp=params['temp'])
        }
    # Create a dicctionary to store the output
    output_dict = {}
    # Adding time stamp
    start_time = MPI.Wtime()
    
    # Loop over the temperatures 
    for t in my_temperatures:
        params['temp'] = t
        params['ofname'] = f'./outsimulation/traj-hydrogen-3D-{N_atoms}-T{t}.dump'    
        print(f'Core {my_rank} is simulating temperature {t} K.')   
        output = run(**params)       
        
        # Store the output in the dicctionary
        output_dict[t] = output
        # Settings for the plot
        plt.figure(figsize=(6,5))
        plt.plot(output[:,0] * 1e12, output[:,1], label=f'T={t} K')
        plt.xlabel('Time (ps)')
        plt.ylabel('Temp (K)')
        plt.title(f'Temperature Evolution for N={N_atoms} atoms at (T={t}K)')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f'./outsimulation/temperature-{N_atoms}-T{t}.png')
        plt.close()
    end_time = MPI.Wtime()
    # Time elapsed for each core
    elapsed_time = end_time - start_time
    # Finalize MPI 
    print(f'Core {my_rank} finished simulating temperature {t} K in {elapsed_time:.2f} seconds.')

    # d) Collect the simulation results from all other processes
    collect_results = world_comm.gather(output_dict, root=0)   
    # e) Generate the temperature plot and trajectory Hfile.
    # If the current process is the root process
    if my_rank == 0:
        # Maximum time for all processes
        total_time = MPI.Wtime() - start_time
        # Merge the output dictionaries from all processes
        merged_output = {}
        for output in collect_results:
            merged_output.update(output)
        # Generate the temperature plot
        plt.figure(figsize=(6,5))
        for t, output in merged_output.items():
            plt.plot(output[:,0] * 1e12 , output[:,1], label=f'T={t} K')
        plt.xlabel('Time (ps)')
        plt.ylabel('Temp (K)')
        plt.title(f'Combined Temperature Evolution for N={N_atoms} atoms')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f'./outsimulation/combined_temperature-{N_atoms}.png')
        plt.close()

        # f) Store the data in CSV file
        filename = './outsimulation/mpi_scaling.csv'
        # Check if the file already exists
        file_exists = os.path.isfile(filename)
        # Write the header
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Cores','Total_Time'])
            writer.writerow([world_size, total_time])
