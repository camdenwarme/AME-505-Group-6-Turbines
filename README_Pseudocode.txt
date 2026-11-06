FDynSim([Policies], W, W^\theta)
	\lambda = 0.1
	-calculation each action trajectory recursively: a^i_t = policy^i(W(t), W^\theta(t), \theta^i_t), theta^i_{t+1} = a^i_t*delta_t + theta^i_t
	-insert W, W^\theta into current .yaml file for wind conditions, insert Theta^i = [theta^i_t]_0^T into .yaml file for yaw controls
	-call run_simulation.py
	-Pull total power across all turbines at each timestep from simulation results, set equal to vector R, and set R(T) = \lambda*(integral of total power over [0,T])
return: interweave([W, W^\theta, Theta], A^i, R) sequentially for each turbine %the output is a state, action, reward trajectory for use in training in an outer loop.

%Note that capital letters denote vectors (dimension is number of timesteps), W is wind strength and W^\theta is wind orientation.
