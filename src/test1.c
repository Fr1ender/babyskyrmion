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
  Vec       s, y, xvec; /* work space for computing Hessian */
  PetscReal hx, hy;     /* mesh spacing in x- and y-directions */
} AppCtx;

/* User-defined routines */

PetscErrorCode FormInitialGuess(AppCtx * ,Vec , Vec , Vec ); /*\phi_1 \phi_2 \phi_3*/
PetscErrorCode FormFunction(Tao, Vec, Vec, Vec, PetscReal *, void *);
PetscReal      FuncComm(PetscReal, PetscReal, PetscReal );
PetscErrorCode FormGradient(Tao, Vec, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode HessianProductMat(Mat, Vec, Vec);
PetscErrorCode HessianProduct(void *, Vec, Vec);
//PetscErrorCode MatrixFreeHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormFunctionGradient(Tao, Vec, Vec, Vec, PetscReal *, Vec, void *);


int main(int argc, char **argv){
  PetscInt    mx = 10; /* discretization in x-direction */
  PetscInt    my = 10; /* discretization in y-direction */
  Vec         x,y,z;       /* solution, gradient vectors */
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
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim*3, &user.y)); 
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
  PetscCall(FormInitialGuess(&user, x, y, z));
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

/*
  FormInitialGuess
  (X Y Z) is (\phi_1 \phi_2 \phi_3)


*/
PetscErrorCode FormInitialGuess(AppCtx *user, Vec X , Vec Y, Vec Z)
{
  PetscReal hx = user->hx, hy = user->hy, bound = user->region;
  PetscReal val1,val2,val3, x, y;
  PetscInt  i, j, k, nx = user->mx, ny = user->my, l = user->lambda;

  /* Compute initial guess */
  PetscFunctionBeginUser;
  for(j = 0; j < ny; j++){
    y = -1 * bound + (j + 1) * hy;
    for(i = 0; i < nx; i++){
      k = nx * j + i;
      x = -1 * bound + (i + 1) *hx; //except bound temporarily
      val1 = FuncComm(l, x, y) * l * l * (x * x + y * y) - 1.0; 
      val2 = FuncComm(l,x,y) * 2.0 * l * x;
      val3 = FuncComm(l,x,y) * -2.0 * l * y;
      PetscCall(VecSetValues(X, 1, &k, &val1, ADD_VALUES));
      PetscCall(VecSetValues(Y, 1, &k, &val2, ADD_VALUES));
      PetscCall(VecSetValues(Z, 1, &k, &val3, ADD_VALUES));
    }
  } 

  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscCall(VecAssemblyBegin(Z));
  PetscCall(VecAssemblyEnd(Z));
  
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
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, Vec Y, Vec Z, PetscReal *f, Vec G, void *ptr)
{
  PetscFunctionBeginUser;
  PetscCall(FormFunction(tao, X, Y,Z, f, ptr));
  PetscCall(FormGradient(tao, X, G, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
FormFunction - Evaluate our energy integral;

*/

PetscErrorCode FormFunction(Tao tao, Vec X, Vec Y, Vec Z, PetscReal *f,void *ptr )
{

  AppCtx            *user = (AppCtx *)ptr;
  PetscReal          hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5;
  PetscReal          zero = 0.0, vb, vl, vr, vt, dvdx, dvdy, flin = 0.0, fquad = 0.0;
  PetscReal          v;//, cdiv3 = user->param / three;
  const PetscScalar *x,*y,*z;
  PetscInt           nx = user->mx, ny = user->my, i, j, k;

  PetscFunctionBeginUser;
  /* Get pointer to vector data */
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(Y,&y));
  PetscCall(VecGetArrayRead(Z,&z));

  /* Compute function */
  for( j = -1; j < ny; j++){
    for(i = -1; i < nx; i++){

    }
  }



}

