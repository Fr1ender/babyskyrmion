import numpy as np

def main():
  data = np.loadtxt('initialvec.dat',skiprows=2)
  filename = "initialN="

  nextfield = float(data.shape[0]) / 3.0
  dim       = np.sqrt(nextfield)
  nextfield = int(nextfield)
  dim       = int(dim)
  print(dim) 
  print(nextfield)

  hx = 0.1
  hy = 0.1
  region = (dim - 1) / 2.0 * hx

  phi1 = np.zeros((dim,dim))
  phi2 = np.zeros((dim,dim))
  phi3 = np.zeros((dim,dim))

  for j in range(dim):
    for i in range(dim):
      phi1[i][j] = data[i + j*dim] 
      phi2[i][j] = data[i + j*dim + nextfield]
      phi3[i][j] = data[i + j*dim + nextfield * 2]

  with open (r"../data/" + filename + str(dim) + ".dat",'w') as out:
    for j in range(dim):
      for i in range(dim):
        #if(j == (dim - 1)/2 and i % 10 == 0):
        if(j % 15 == 0 and i % 15 == 0):
          print( -region + i*hx,-region + j*hy, "0.0" ,phi1[i][j], phi2[i][j], phi3[i][j], sep=" ", file=out)
    
  
if __name__ == "__main__":
  main()