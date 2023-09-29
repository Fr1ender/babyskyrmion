import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def main():
  mx = 20
  my = 20
  region = 10
  dim = mx
  hx   = 2 * region / (mx + 1)
  hy   = 2 * region / (my + 1)

  #a = np.zeros(dim*dim*3)

  data = np.loadtxt('takosu.dat')

  phi1 = np.zeros((dim,dim))
  #output = np.zeros((dim,dim,dim))

  for j in range(dim):
    for i in range (dim):
      phi1[i][j] = data[i + j*dim]

  with open ("goodd.dat",'w') as out:
   for i in range(dim):
    for j in range(dim):
      print( -region + (i+1)*hx,-region + (j+1)*hy, phi1[i][j], sep=" ", file=out)

  

if __name__ == "__main__":
  main()
 