import SLNK as slnk
import numpy as np
import sys

trials = 2500
psize = 100
reps = 50
n = 15
solutions = 2**n

### Community experiment
ncomm = 5
conditions = 5
ig = np.array([0.2,0.8,0.95,0.9875,1.0])
og = (1-ig)/(ncomm-1)
sh = [0.5,1.0] # Partial and Full

class SLSim:
    def __init__(self):
        pass

    def simulateComm(self,n,k,reps):
        '''For a single k, run the simulation many times for each condition'''
        avg = np.zeros((int(len(ig)*len(sh)),reps))
        for rep in range(reps):
            nk = slnk.NKLandscape(n,k)
            pop = slnk.Population(psize,n,nk,True)
            initial_genotypes = pop.genotypes.copy()
            condition = 0
            for comm in range(len(ig)):
                pop.set_community(ncomm, ig[comm], og[comm])
                for share in range(len(sh)):
                    pop.set_pop(initial_genotypes)
                    pop.share_rate = sh[share]
                    for trial_num in range(1,trials+1):
                        pop.share()
                        pop.learn()
                    avg[condition,rep] = pop.avgfit()
                    condition += 1
        return avg

    def exp(self,minK,maxK,step,id):
        '''Run the experiment for all values of K and all conditions and save the data'''
        for k in range(minK,maxK+1,step):
            print("K: ",k)
            avg=self.simulateComm(n,k,reps)
            np.save("avg_"+str(k)+"_"+str(id)+".npy", avg)

id = int(sys.argv[1])
sim = SLSim()
sim.exp(0,14,1,id)
