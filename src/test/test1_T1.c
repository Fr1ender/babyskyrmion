#include "petscsys.h"
#include "petscsystypes.h"
#include "petsctao.h"
#include "petscvec.h"
#include "petscviewer.h"

typedef struct {
  PetscReal lambda;     /* parameter for initial configuration */
  PetscReal region;     /* Region size parameter */
  PetscInt  mx, my;     /* discretization in x- and y-directions */
  PetscInt  ndim;       /* problem dimension */
  Vec       s, y, xvec; /* work space for computing Hessian (?)*/
  PetscReal hx, hy;     /* mesh spacing in x- and y-directions */
} AppCtx;

PetscReal      FuncComm(PetscReal, PetscReal, PetscReal );
PetscErrorCode FormInitialGuess(AppCtx *, Vec );

static char help[] = "sugoi test1!";

int main(int argc, char **argv){
  AppCtx user;
  Vec    x;
  PetscMPIInt size;
  PetscInt    mx = 10; /* discretization in x-direction */
  PetscInt    my = 10; /* discretization in y-direction */
  PetscBool flg;
  PetscViewer miyouvectorwo;


  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.region   = 10.0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &user.lambda, &flg));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n---- This is test for vector viewer -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "mx: %" PetscInt_FMT "     my: %" PetscInt_FMT "   \n\n", mx, my));
  user.ndim = mx * my;
  user.mx   = mx;
  user.my   = my;
  user.hx   = 2 * user.region / (mx + 1); /* lattice size */
  user.hy   = 2 * user.region / (my + 1);
  user.lambda = 1.0;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim*3, &user.y)); // *3 
  PetscCall(VecDuplicate(user.y, &x));
  PetscCall(FormInitialGuess(&user, x));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "takosu.dat", &miyouvectorwo));
  PetscCall(PetscViewerPushFormat(miyouvectorwo, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(x, miyouvectorwo));
  PetscCall(PetscViewerPopFormat(miyouvectorwo));
  PetscCall(PetscViewerDestroy(&miyouvectorwo));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}


PetscReal FuncComm(PetscReal Lambda, PetscReal x, PetscReal y){
  return 1/(Lambda * Lambda * (x * x + y * y) + 1.0); 
}
PetscErrorCode FormInitialGuess(AppCtx *user, Vec X)
{
  PetscReal hx = user->hx, hy = user->hy, bound = user->region;
  PetscReal val1,val2,val3, x, y;
  PetscInt  i, j, k, n, m, nx = user->mx, ny = user->my, l = user->lambda;

  /* Compute initial guess */
  PetscFunctionBeginUser;
  for(j = 0; j < ny; j++){
    y = -1 * bound + (j + 1) * hy;
    for(i = 0; i < nx; i++){

      k = ny * j + i;
      n = k + nx * ny;
      m = n + nx * ny; /* combine \phi1 \phi2 \phi3 into one vector X*/

      x = -1 * bound + (i + 1) *hx; //except bound temporarily

      val1 = FuncComm(l, x, y) * l * l * (x * x + y * y) - 1.0; 
      val2 = FuncComm(l,x,y) * 2.0 * l * x;
      val3 = FuncComm(l,x,y) * -2.0 * l * y;
      PetscCall(VecSetValues(X, 1, &k, &val1, ADD_VALUES));
      PetscCall(VecSetValues(X, 1, &n, &val2, ADD_VALUES));
      PetscCall(VecSetValues(X, 1, &m, &val3, ADD_VALUES));
    }
  } 

  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}