The runtime is proportional to the number of particles, as the calculations are done for each particle. The runtime is also proportional to the length of the time interval, as the calculations are done for each time step.

$$ \text{Runtime} \propto \text{Length of time array}  \times \text{Number of particles}$$

Where $T$ is the length of the array created from the time interval, which is always constant, $T =constant$, and $N_p$ is the number of particles, we get that the runtime is proportional to the number of particles.
$$ \mathcal{O}(T \cdot N_p) = \mathcal{O}(c\cdot N_p) = \mathcal{O}(N_p)$$



### Langrangian method
The Langrangian method is a numerical method used to simulate the movement of particles in a fluid. The method is based on the idea of representing the particles as numerical points and simulating the individual particles trajectories as they are traveling with the wind and current. The result is a simulation that calculates the probability of finding particles at specific locations at a given time. We do not need to model the fluid itself, as this information is produced and simulated by the Meteorological Institute [1]. Using this data, we access a more realistic understanding of the movement of the particles as well as accumulation zones.

### Implementation of the Langrangian method with wind and current data
The implementation of the Langrangian method is done by first loading the wind and current data from the Meteorological Institute. The data is then used to calculate the velocity field at each point in time. The velocity field is then used to calculate the position of the particles at each point in time. The particles are then plotted on a map to visualize their movement.

