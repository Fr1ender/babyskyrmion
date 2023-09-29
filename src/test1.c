/*
My object is minimizing static energy functional for baby Skyrm model to get the field \phi.
\phi is a SO(2) triplet scaler field.

Baby skyrm model is 
\mathcal{L} = 1/2 c_2\partial_\mu\phi^{a}\partial^{\mu}\phi_a \frac{1}{4}F_{\mu}{\nu}F^{\mu}{\nu} -c_0\frac{1}{2}U(\phi)^2

So, this program compute the solution of the following problem

min_\phi {\int 1/2 c_2 ||\nabla\phi||^2 - 1/4 c_4 F_{12}^2 - c_0 1/2 U(\phi)^2}
\lim_{r \to \infty} \phi(r) = (1,0,0) .

Second term is the Skyrm term and c_i are model parameters.

See Juha Jaykka and Martin Speight PHYSICAL REVIEW D 82, 125030 (2010) and PETSc/TAO websites.
PETSc library has many example for numerical problem. "eptorsion1.c", which is one of the examples is very helpful for me to code this program.

*/
#include "petscsys.h"
#include "petscsystypes.h"
#include "petsctao.h"
#include "petscvec.h"

static char help[] = "Waai!";

typedef struct {
  PetscReal param_c2, param_c4, param_c0;      /* model parameters */
  PetscReal lambda;     /* parameter for initial configuration */
  PetscReal region;     /* Region size parameter */
  PetscInt  mx, my;     /* discretization in x- and y-directions */
  PetscInt  ndim;       /* problem dimension */
  Vec       s, y, xvec; /* work space for computing Hessian (?)*/
  PetscReal hx, hy;     /* mesh spacing in x- and y-directions */
} AppCtx;

/* User-defined routines */

PetscReal      PotentialTerm(AppCtx *, PetscReal );
PetscReal      SkyrmTerm(PetscReal ,PetscReal ,PetscReal, PetscReal, PetscReal, AppCtx *);
PetscErrorCode FormInitialGuess(AppCtx * ,Vec ); /*\phi_1 \phi_2 \phi_3*/
PetscErrorCode FormFunction(Tao, Vec, PetscReal *, void *);
PetscReal      FuncComm(PetscReal, PetscReal, PetscReal );
PetscErrorCode FormGradient(Tao, Vec, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode HessianProductMat(Mat, Vec, Vec);
PetscErrorCode HessianProduct(void *, Vec, Vec);
//PetscErrorCode MatrixFreeHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);


int main(int argc, char **argv){
  PetscInt    mx = 10; /* discretization in x-direction */
  PetscInt    my = 10; /* discretization in y-direction */
  Vec         x;       /* solution, gradient vectors */
  PetscBool   flg;     /* A return value when checking for use options */
  Tao         tao;     /* Tao solver context */
  Mat         H;       /* Hessian matrix */
  AppCtx      user;    /* application context */
  PetscMPIInt size;    /* number of processes */
  PetscReal one = 1.0;

    
  /* Initialize TAO,PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");


  /* Specify default parameters for the problem, check for command-line overrides */
  user.param_c0 = 1.0;
  user.param_c2 = 1.0;
  user.param_c4 = 1.0;
  user.lambda   = 1.0;
  user.region   = 10.0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parc0", &user.param_c0, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parc2", &user.param_c2, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parc4", &user.param_c4, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &user.lambda, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-region", &user.region, &flg));
  //PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_lmvm", &test_lmvm, &flg));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n---- Minimizing the energy functional -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "mx: %" PetscInt_FMT "     my: %" PetscInt_FMT "   \n\n", mx, my));
  user.ndim = mx * my;
  user.mx   = mx;
  user.my   = my;
  user.hx   = 2 * user.region / (mx + 1); /* lattice size */
  user.hy   = 2 * user.region / (my + 1);


  /* Allocate vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim*3, &user.y)); // *3 
  PetscCall(VecDuplicate(user.y, &user.s));
  PetscCall(VecDuplicate(user.y, &x));
  /*
  PetscCall(VecDuplicate(user.y, &y));
  PetscCall(VecDuplicate(user.y, &z));
  */
  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Set solution vector with an initial guess */
  PetscCall(FormInitialGuess(&user, x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));

  /* Set Hessian */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, user.ndim, user.ndim, 5, NULL, &H));
  PetscCall(MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, (void *)&user));


  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&user.s));
  PetscCall(VecDestroy(&user.y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));

  PetscCall(PetscFinalize());
  return 0;
}

/* Evaluate potential term. In this problem, U(\phi) = \phi_3*/

PetscReal Potentialterm(AppCtx *user ,PetscReal phi_3){
  return -0.5 * user->param_c0 * phi_3 * phi_3;
}

/* for evaluating skyrm term
  NOT USE THIS
PetscReal SkyrmTerm(PetscReal entry1, PetscReal entry2l, PetscReal entry2r, PetscReal entry3l, PetscReal entry3r, AppCtx *user){
  PetscReal    nx = user->hx, ny = user->hy;
  PetscReal    dp2dx,dp3dy;
  
  dp2dx = (entry2r - entry2l) / nx;
  dp3dy = (entry3r - entry3l) / ny;

  return entry1 * dp2dx * dp3dy;
}
*/

/*
  FormInitialGuess
  (X Y Z) is (\phi_1 \phi_2 \phi_3)


*/
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
      m = l + nx * ny; /* combine \phi1 \phi2 \phi3 into one vector X*/

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

PetscReal FuncComm(PetscReal Lambda, PetscReal x, PetscReal y){
  return 1/(Lambda * Lambda * (x * x + y * y) + 1.0); 
}

/* ------------------------------------------------------------------- */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetFunction()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscFunctionBeginUser;
  PetscCall(FormFunction(tao, X, f, ptr));
  PetscCall(FormGradient(tao, X, G, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
FormFunction - Evaluate our energy integral;

*/

PetscErrorCode FormFunction(Tao tao, Vec X, PetscReal *f,void *ptr )
{

  AppCtx            *user = (AppCtx *)ptr;
  PetscReal          hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5;
  PetscReal          zero = 0.0, vb, vl, vr, vt, dp1dx, dp1dy,dp2dx,dp2dy,dp3dx,dp3dy, flin = 0.0, fquad = 0.0, f12 = 0.0 ,fskyrm = 0.0, fpot = 0.0;
  PetscReal          v;//, cdiv3 = user->param / three;
  const PetscScalar *x;
  PetscInt           nx = user->mx, ny = user->my, i, j, k1,k2,k3;
  PetscInt           dim;

  PetscFunctionBeginUser;
  /* Get pointer to vector data */
  PetscCall(VecGetArrayRead(X,&x));
  dim = nx * ny;

  /* Compute function 

    Evaluate integrate by summing value at volume of two triangular prism
  
  */

  /* lower triangle */
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
      fquad += dp1dx * dp1dx + dp1dy * dp1dy;

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


      /* ||\nabla /phi_3||^2 */
      v  = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0) v = x[k3];
      if (i < nx - 1 && j > -1) vr = x[k3 + 1];
      if (i > -1 && j < ny - 1) vt = x[k3 + nx];
      dp3dx = (vr - v) / hx;
      dp3dy = (vt - v) / hy;

      fquad += dp3dx * dp3dx + dp3dy * dp3dy;
      // fquad = user->param_c2 * fquad / 2.0; after summing all, do this

      f12 = x[k1]*dp2dx*dp3dy + x[k3]*dp1dx*dp2dy + x[k2]*dp3dx*dp1dy;
      f12 -= x[k2]*dp1dx*dp3dy + x[k3]*dp2dx*dp1dy + x[k1]*dp3dx*dp2dy;
      fskyrm += f12 * f12;
      f12 = 0.0;

      fpot += PotentialTerm(user, x[k3]);

      // flin -= (v + vr + vt) / 3.0 ;
    }
  }
  /* upper triangle*/
  for (j = 0; j <= ny; j++) {
    for (i = 0; i <= nx; i++) {
      k1  = nx * j + i;
      k2  = nx * j + i + dim;
      k3  = nx * j + i + dim * 2;
      /* ||\nabla /phi_1||^2 */
      vb = 1.0;
      vl = 1.0;
      v  = 1.0;
      if (i < nx && j > 0) vb = x[k1 - nx];
      if (i > 0 && j < ny) vl = x[k1 - 1];
      if (i < nx && j < ny) v = x[k1];
      dp1dx = (v - vl) / hx;
      dp1dy = (v - vb) / hy;
      fquad += dp1dx * dp1dx + dp1dy * dp1dy;

      /* ||\nabla /phi_2||^2 */
      vb = zero;
      vl = zero;
      v  = zero;
      if (i < nx && j > 0) vb = x[k2 - nx];
      if (i > 0 && j < ny) vl = x[k2 - 1];
      if (i < nx && j < ny) v = x[k2];
      dp2dx = (v - vl) / hx;
      dp2dy = (v - vb) / hy;
      fquad += dp2dx * dp2dx + dp2dy * dp2dy;


      /* ||\nabla /phi_3||^2 */
      vb = zero;
      vl = zero;
      v  = zero;
      if (i < nx && j > 0) vb = x[k3 - nx];
      if (i > 0 && j < ny) vl = x[k3 - 1];
      if (i < nx && j < ny) v = x[k3];
      dp3dx = (v - vl) / hx;
      dp3dy = (v - vb) / hy;
      fquad += dp3dx * dp3dx + dp3dy * dp3dy;

      f12 = x[k1]*dp2dx*dp3dy + x[k3]*dp1dx*dp2dy + x[k2]*dp3dx*dp1dy;
      f12 -= x[k2]*dp1dx*dp3dy + x[k3]*dp2dx*dp1dy + x[k1]*dp3dx*dp2dy;
      fskyrm = f12*f12;
      f12 = 0;
      fpot += PotentialTerm(user, x[k3]);
    }
  }

  fquad  = user->param_c2 * fquad / 2.0;
  fskyrm = p5 * user->param_c4 * fskyrm;

  PetscCall(VecRestoreArrayRead(X, &x));

  area = p5 * hx * hy;
  *f   = area * (fquad + fskyrm + fpot);

  PetscFunctionReturn(PETSC_SUCCESS);
}

