#include "petsctao.h"
#include "petscviewer.h"

typedef struct {
  PetscReal lambda;     /* parameter for initial configuration */
  PetscReal region;     /* Region size parameter */
  PetscInt  mx, my;     /* discretization in x- and y-directions */
  PetscInt  ndim;       /* problem dimension */
  Vec       s, y, xvec; /* work space for computing Hessian (?)*/
  PetscReal hx, hy;     /* mesh spacing in x- and y-directions */
  PetscReal param_c2,param_c4,param_lag;
} AppCtx;

PetscReal      FuncComm(PetscReal, PetscReal, PetscReal );
PetscErrorCode FormInitialGuess(AppCtx *, Vec );
PetscErrorCode EnergyDensity(Vec ,Vec ,void *);

static char help[] = "sugoi test1!";

int main(int argc, char **argv){
  AppCtx user;
  Vec    x,E;
  PetscMPIInt size;
  PetscInt    mx = 10; /* discretization in x-direction */
  PetscInt    my = 10; /* discretization in y-direction */
  PetscBool flg;
  PetscViewer Eout;


  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.region   = 10.0;
  user.param_lag = 1.0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &user.lambda, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-paramlag", &user.param_lag, &flg));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n---- This is test for vector viewer -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "mx: %" PetscInt_FMT "     my: %" PetscInt_FMT "   \n\n", mx, my));
  user.ndim = mx * my;
  user.mx   = mx;
  user.my   = my;
  user.hx = 0.1;
  user.hy = 0.1;
  user.region = (mx - 1) / 2.0 * user.hx;
  /*
  user.hx   = 2 * user.region / (mx + 1);  lattice size
  user.hy   = 2 * user.region / (my + 1);
  */
  user.lambda = 1.0;
  user.param_c2 = 1.0;
  user.param_c4 = 1.0;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim*3, &user.y)); // *3 
  PetscCall(VecDuplicate(user.y, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim, &E)); // *3 
  PetscCall(FormInitialGuess(&user, x));

  /*
  PetscViewer miyouyovectorwo
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "takosu.dat", &miyouvectorwo));
  PetscCall(PetscViewerPushFormat(miyouvectorwo, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(x, miyouvectorwo));
  PetscCall(PetscViewerPopFormat(miyouvectorwo));
  PetscCall(PetscViewerDestroy(&miyouvectorwo));
  */

  PetscCall(EnergyDensity(x,E, (void *)&user));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "initialenergy.dat", &Eout));
  PetscCall(PetscViewerPushFormat(Eout, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(E, Eout));
  PetscCall(PetscViewerPopFormat(Eout));
  PetscCall(PetscViewerDestroy(&Eout));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

PetscReal PotentialTerm(AppCtx *user ,PetscReal phi_3){
  return  0.5 * 1.0 * phi_3 * phi_3;
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
    y = -1 * bound + j * hy;
    for(i = 0; i < nx; i++){

      k = ny * j + i;
      n = k + nx * ny;
      m = n + nx * ny; /* combine \phi1 \phi2 \phi3 into one vector X*/

      x = -1 * bound + i * hx; //except bound temporarily

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

PetscErrorCode EnergyDensity(Vec X,Vec E,void *ptr){
  AppCtx            *user = (AppCtx *)ptr;
  PetscReal          hx = user->hx, hy = user->hy, p5 = 0.5;
  PetscReal          zero = 0.0, vr, vt, dp1dx, dp1dy,dp2dx,dp2dy,dp3dx,dp3dy, fquad = 0.0, f12 = 0.0 ,fskyrm = 0.0, fpot = 0.0,flag = 0.0;
  PetscReal          v,v1,v2,v3,val;//, cdiv3 = user->param / three;
  const PetscScalar *x;
  PetscInt           nx = user->mx, ny = user->my, i, j, k1,k2,k3;
  PetscInt           dim,ind;

  PetscFunctionBeginUser;
  /* Get pointer to vector data */
  PetscCall(VecGetArrayRead(X,&x));
  dim = nx * ny;

  /* 
      Compute energy density
  */

  for (j = -1; j < ny; j++) {
    for (i = -1; i < nx; i++) {
      k1  = nx * j + i;
      k2  = nx * j + i + dim;
      k3  = nx * j + i + dim * 2;
      /* ||\nabla /phi_1||^2 */
      v  = 1.0;
      vr = 1.0;
      vt = 1.0;
      if (i >= 0 && j >= 0) v = x[k1];
      if (i < nx - 1 && j > -1) vr = x[k1 + 1];
      if (i > -1 && j < ny - 1) vt = x[k1 + nx];
      dp1dx = (vr - v) / hx;
      dp1dy = (vt - v) / hy;
      fquad = dp1dx * dp1dx + dp1dy * dp1dy;
      v1 = v;

      /* ||\nabla /phi_2||^2 */
      v  = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0) v = x[k2];
      if (i < nx - 1 && j > -1) vr = x[k2 + 1];
      if (i > -1 && j < ny - 1) vt = x[k2 + nx];
      dp2dx = (vr - v) / hx;
      dp2dy = (vt - v) / hy;
      fquad += dp2dx * dp2dx + dp2dy * dp2dy;
      v2 = v;

      /* ||\nabla /phi_3||^2 */
      v  = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0) v = x[k3];
      if (i < nx - 1 && j > -1) vr = x[k3 + 1];
      if (i > -1 && j < ny - 1) vt = x[k3 + nx];
      dp3dx = (vr - v) / hx;
      dp3dy = (vt - v) / hy;
      v3 = v;

      fquad += dp3dx * dp3dx + dp3dy * dp3dy;
      PetscReal pnorm;
      pnorm  = v1 * v1 + v2 * v2 + v3 * v3;
      flag = (pnorm - 1) * (pnorm - 1);
      // fquad = user->param_c2 * fquad / 2.0; after summing all, do this

      f12 = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12 -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;
      fskyrm = f12 * f12;
      //f12 = 0.0;

      fpot = PotentialTerm(user, v3);
      fquad  = user->param_c2 * fquad / 2.0;
      fskyrm = p5 * user->param_c4 * fskyrm;
      flag = user->param_lag * flag;

      val = 0;
      ind = k1;
      if( j >= 0 && i >= 0){
        val = fquad + fskyrm + fpot + flag;
        PetscCall(VecSetValues(E, 1, &ind, &val, INSERT_VALUES));
      }

      fquad = zero;
    }
  }
  /* assemble vector */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecAssemblyBegin(E));
  PetscCall(VecAssemblyEnd(E));

  //area = p5 * hx * hy;
  //*f   = area * (fquad + fskyrm + fpot);

  PetscFunctionReturn(PETSC_SUCCESS);
}

