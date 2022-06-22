import numpy as np
import matplotlib.pyplot as plt

Ks = 12
Cs = 4
Rs = 1000
avg = np.zeros((Ks,Cs))
std = np.zeros((Ks,Cs))
all = np.zeros((Ks,Cs,Rs))
for k in range(Ks):
    i = 0
    d = np.load("N12/avg_"+str(k)+"_"+str(i)+".npy")
    print(d.shape)
    for i in range(1,10):
        newd = np.load("N12/avg_"+str(k)+"_"+str(i)+".npy")
        d = np.concatenate((d,newd),axis=1)
    all[k] = d
    avg[k] = np.mean(d,axis=1)
    std[k] = np.std(d,axis=1)
np.save("Viz/n12p100.npy",all)
np.savetxt("Viz/n12p100_avg.csv",avg,delimiter=",")
np.savetxt("Viz/n12p100_std.csv",std/np.sqrt(Rs),delimiter=",")

print(avg.shape)
plt.plot(avg)
plt.show()
