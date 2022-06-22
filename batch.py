import os
import sys

for id in range(10):
   os.system('python Sim_Reg.py '+str(id)+"&")

# for id in range(6,20):
#     os.system('python Sim_Comm.py '+str(id)+"&")

# for id in range(10,20):
#     os.system('python Sim_Small.py '+str(id)+"&")
#
# for id in range(10,20):
#     os.system('python Sim_Scale.py '+str(id)+"&")
