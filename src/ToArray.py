import numpy as np

def main():
  hx = 0.1
  hy = 0.1

  #a = np.zeros(dim*dim*3)

  data = np.loadtxt('energydensity.dat',skiprows=2)
  print(data.shape[0])
  dim = np.sqrt(float(data.shape[0]))
  print(dim)
  dim = int(dim)

  region = (dim - 1) / 2.0 * hx

  phi1 = np.zeros((dim,dim))
  #output = np.zeros((dim,dim,dim))
  filename = "densityplotN="

  for j in range(dim):
    for i in range (dim):
      phi1[i][j] = data[i + j*dim]

  with open (filename + str(dim) + ".dat",'w') as out:
   for j in range(dim):
    for i in range(dim):
      print( -region + i*hx,-region + j*hy, phi1[i][j], sep=" ", file=out)
      if(i == dim-1):
        print(" ", file = out)
      

  

if __name__ == "__main__":
  main()
 