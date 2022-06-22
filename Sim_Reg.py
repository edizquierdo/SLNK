import SLNK as slnk
import numpy as np
import sys

trials = 2500
psize = 100
reps = 100
n = 12
solutions = 2**n

SHARE_RATE_PARTIAL = 0.5 # Partial share
SHARE_RATE_FULL = 1.0 # Full share
SHARE_RADIUS_GLOBAL = psize - 1 # Global
SHARE_RADIUS_LOCAL = 1 # Local

# Main four conditions: Full and Partial sharing, Global and Local populations.
conditions = 4
exp_condition = np.zeros((conditions,2))
exp_condition[0] = [SHARE_RATE_FULL, SHARE_RADIUS_GLOBAL]
exp_condition[1] = [SHARE_RATE_FULL, SHARE_RADIUS_LOCAL]
exp_condition[2] = [SHARE_RATE_PARTIAL, SHARE_RADIUS_GLOBAL]
exp_condition[3] = [SHARE_RATE_PARTIAL, SHARE_RADIUS_LOCAL]

class SLSim:
    def __init__(self):
        pass

    def simulateReg(self,n,k,reps):
        '''For a single k, run the simulation many times for each condition'''
        avg = np.zeros((conditions,reps))
        for rep in range(reps):
            nk = slnk.NKLandscape(n,k)
            pop = slnk.Population(psize,n,nk,False)
            initial_genotypes = pop.genotypes.copy()
            for condition in range(conditions):
                pop.set_pop(initial_genotypes)
                pop.share_rate = exp_condition[condition][0]
                pop.share_radius = exp_condition[condition][1]
                for trial_num in range(1,trials+1):
                    pop.share()
                    pop.learn()
                avg[condition,rep] = pop.avgfit()
        return avg

    def exp(self,minK,maxK,step,id):
        '''Run the experiment for all values of K and all conditions and save the data'''
        for k in range(minK,maxK+1,step):
            print("K: ",k)
            avg=self.simulateReg(n,k,reps)
            np.save("N12/avg_"+str(k)+"_"+str(id)+".npy", avg)

id = int(sys.argv[1])
sim = SLSim()
sim.exp(0,14,1,id)
