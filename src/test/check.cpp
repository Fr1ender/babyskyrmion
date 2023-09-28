#include <iostream>


int main(){
  int i,j,nx,ny;
  double hx,hy,x,y;
  double bound = 10.0;

  nx = 5; ny = 5;

  hx = 2 * bound / (1 + nx);
  hy = 2 * bound / (1 + ny);

  for(j = 0; j < ny; j++){
    y = -1 * bound + (j + 1) * hy;
    for(i = 0; i < nx; i++){
      x = -1 * bound + (i + 1) *hx;
      std::cout << x << " " << y << "\n";
    }
  }
}