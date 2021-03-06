#############################################################################################
##
## Author: 			Vineeth C M
## Date: 			15th February 2019
## Description: 	Epsilon greedy based multi armed bandit
##
#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

def pull_arm(arm_idx):
	'''
	simulates the action of pulling an arm (arm_idx) and returns the reward
	'''
	return np.random.randn() + arm_means[arm_idx]

def update_value_fn(arm_idx,reward,value_fn,arm_pull_counts):
	'''
	updates the value function and arm pull count after the action of pulling an arm is performed
	'''
	value_fn[arm_idx] = (value_fn[arm_idx]*arm_pull_counts[arm_idx] + reward)/(arm_pull_counts[arm_idx] + 1)
	arm_pull_counts[arm_idx] += 1

def select_arm_egreedy(value_fn,eps):
	'''
	selects an arm based on epsilon greedy algorithm 
	'''
	if(np.random.rand() > eps):
		return value_fn.index(max(value_fn))
	else:
		return np.random.randint(num_arms)

num_arms=10
time_steps=1000
num_sims = 2000

arm_means = []

# Generate the true value function Q*(a) for each of the arms using N(0,1)
for i in range(num_arms):
	arm_means.append(np.random.randn())

#Optimal arm is the one with the highest Q*(a)
optimal_arm_idx = arm_means.index(max(arm_means))

# Different epsilon values
eps_ranges = [0.01,0.1]

qt_a_eps = []

opt_act_eps = []

# Run simulation for different epsilon values
for eps in eps_ranges:
	
	value_fn_db = []
	opt_act_frac = []
	# Run simulation
	for sim in range(num_sims):
		value_fn_db.append([])
		opt_act_frac.append([])
		value_fn = [0 for i in range(num_arms)]
		arm_pull_counts = [0 for i in range(num_arms)]
		# Loop across time steps
		for t in range(time_steps):
			arm_idx = select_arm_egreedy(value_fn,eps)
			reward = pull_arm(arm_idx)
			update_value_fn(arm_idx,reward,value_fn,arm_pull_counts)
			opt_act_frac[sim].append(arm_pull_counts[optimal_arm_idx]/(t+1))
			value_fn_db[sim].append(value_fn[:])

	avg_value_fn_across_sims=[[] for i in range(num_arms)]
	for a in range(num_arms):
		for i in range(time_steps):
			total = 0
			for sim in range(num_sims):
				total += value_fn_db[sim][i][a]
			avg = total/num_sims
			avg_value_fn_across_sims[a].append(avg)
	
	avg_opt_act_frac = []
	for i in range(time_steps):
		total = 0
		for sim in range(num_sims):
			total += opt_act_frac[sim][i]
		avg_opt_act_frac.append(total/num_sims)


	qt_a_eps.append([x[:] for x in avg_value_fn_across_sims])
	opt_act_eps.append(avg_opt_act_frac[:])

# Plotting Qt(a) as a function of time for all arms
for a in range(num_arms):
	plt.figure()
	for i in range(len(eps_ranges)):
		plt.plot(qt_a_eps[i][a],label='eps = ' +str(eps_ranges[i]))
		plt.legend()
	plt.title("Value function for arm " + str(a))
	plt.plot([0,1000],[arm_means[a],arm_means[a]],label = 'true value')
	plt.ylabel("Qt(%s)"%str(a))
	plt.xlabel("time steps (sim. time)")
	plt.legend()
	plt.show()

# Plotting Optimal action % as a function of time 
plt.figure()
for i in range(len(eps_ranges)):
	plt.plot(opt_act_eps[i],label='eps = ' +str(eps_ranges[i]))
	plt.legend()
plt.title("Optimal action % ")
plt.ylabel("Optimal action % ")
plt.xlabel("time steps (sim. time)")
plt.show()