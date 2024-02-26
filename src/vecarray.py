import numpy as np

def main():
  data_field = np.loadtxt('resultfield.dat',skiprows=2)
  #data_field = np.loadtxt('initialvec.dat',skiprows=2)
  data_grad = np.loadtxt('gradient.dat',skiprows=2)
  filename_g = "gradN="
  filename_f = "solN="
  filename_p3 = "phi3N="
  #filename_f = "initialN="

  nextfield = float(data_field.shape[0]) / 3.0
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
  g1   = np.zeros((dim,dim))
  g2   = np.zeros((dim,dim))
  g3   = np.zeros((dim,dim))

  for j in range(dim):
    for i in range(dim):
      phi1[i][j] = data_field[i + j*dim] 
      phi2[i][j] = data_field[i + j*dim + nextfield]
      phi3[i][j] = data_field[i + j*dim + nextfield * 2]
      g1[i][j] = data_grad[i + j*dim]
      g2[i][j] = data_grad[i + j*dim + nextfield]
      g3[i][j] = data_grad[i + j*dim + nextfield * 2]

  with open (r"../data/" + filename_g + str(dim) + ".dat",'w') as outgrad:
    with open (r"../data/" + filename_f + str(dim) + ".dat",'w') as outfield:
      with open (r"../data/" + filename_p3 + str(dim) + ".dat",'w') as outp3:
        for j in range(dim):
          for i in range(dim):
            #if(j == (dim - 1)/2 and i % 10 == 0):
            if(j % 10 == 0 and i % 10 == 0):
            #if(i % 10 == 0 and j == 100):
              #print( "0.0",-region + i*hx,-region + j*hx ,phi1[i][j], phi2[i][j], phi3[i][j], sep=" ", file=outfield)
              print( -region + i*hx,-region + j*hx,"0.0",phi1[i][j], phi2[i][j], phi3[i][j], sep=" ", file=outfield)
              print( -region + i*hx,-region + j*hy, "0.0" ,g1[i][j], g2[i][j], g3[i][j], sep=" ", file=outgrad)
              print( -region + i*hx,-region + j*hy,g3[i][j], sep=" ", file=outp3)
              if(i == dim-1):
                print(" ", file = outp3)
    
  
if __name__ == "__main__":
  main()