import SLNK as slnk
import numpy as np
import sys

trials = 2500
psize = 100
reps = 50
n = 15
solutions = 2**n

### Small World experiment
conditions = 1
degree = 4
p = 0.1
sh = [0.5,1.0]

class SLSim:

    def __init__(self):
        pass

    def simulateSmall(self,n,k,reps):
        avg = np.zeros((len(sh),reps))
        for rep in range(reps):
            nk = slnk.NKLandscape(n,k)
            pop = slnk.Population(psize,n,nk,True)
            initial_genotypes = pop.genotypes.copy()
            condition = 0
            pop.set_small_world(k = degree, p = p)
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
        for k in range(minK,maxK+1,step):
            print("K: ",k)
            avg=self.simulateSmall(n,k,reps)
            np.save("SMALL/avg_"+str(k)+"_"+str(id)+".npy", avg)

id = int(sys.argv[1])
sim = SLSim()
sim.exp(0,14,1,id)
