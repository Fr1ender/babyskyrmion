/*
My object is minimizing static energy functional for baby Skyrm model to get the field \phi.
\phi is a SO(2) triplet scaler field.    Last update '23 10 21

Baby skyrm model is 
\mathcal{L} = 1/2 c_2\partial_\mu\phi^{a}\partial^{\mu}\phi_a \frac{1}{4}F_{\mu}{\nu}F^{\mu}{\nu} -c_0\frac{1}{2}U(\phi)^2

So, this program compute the solution of the following problem

min_\phi {\int 1/2 c_2 ||\nabla\phi||^2 - 1/4 c_4 F_{12}^2 - c_0 1/2 U(\phi)^2}
\lim_{r \to \infty} \phi(r) = (1,0,0) .

Second term is the Skyrm term and c_i are model parameters.

See Juha Jaykka and Martin Speight PHYSICAL REVIEW D 82, 125030 (2010) and PETSc/TAO websites.
PETSc library has many example for numerical problem. "eptorsion1.c", which is one of the examples is very helpful for me to code this program.

mpiexec -n -1 speight 
command line argument
-log_view
-tao_view  (./configure --with-debugging=no)
-tao_smonitor
-tao_gatol 1.e-4
-tao_type lmvm
-tao_max_funcs 200000
-tao_max_it 5000
-start_in_debugger

*/
#include "mpi.h"
#include "petsctao.h"
#include "petscviewer.h"
#include <omp.h>

static char help[] = "this program minimumize some energy functional. -mx and -my is option for lattice size";

typedef struct {
  PetscReal param_c2, param_c4, param_c0;      /* model parameters */
  PetscReal lambda;     /* parameter for initial configuration */
  PetscReal region;     /* Region size parameter */
  PetscInt  mx, my;     /* discretization in x- and y-directions */
  PetscInt  ndim,itrmax,itr;       /* problem dimension and itration number*/
  Vec       chargeitr,derrick;       /* charge monitor & derrick monitor */
  Vec       gradient;              /* monitor gradient vec */
  //Vec       s, y, xvec; /* work space for computing Hessian (?)*/
  PetscReal hx, hy;     /* mesh spacing in x- and y-directions */
  PetscReal param_lag,normglobal,newlag;  /* lagrange multiplier */
  PetscBool periodic;
} AppCtx;

/* User-defined routines */

PetscReal      PotentialTerm(AppCtx *, PetscReal );
PetscReal      SkyrmTerm(PetscReal ,PetscReal ,PetscReal, PetscReal, PetscReal, AppCtx *);
PetscErrorCode FormInitialGuess(AppCtx * ,Vec ); /*\phi_1 \phi_2 \phi_3*/
PetscErrorCode FormFunction(Tao, Vec, PetscReal *, void *);
PetscReal      FuncComm(PetscReal, PetscReal, PetscReal );
PetscErrorCode FormGradient(Tao, Vec, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *); // TaoLMVM doesn't need Hessian!!!
PetscErrorCode HessianProductMat(Mat, Vec, Vec);
PetscErrorCode HessianProduct(void *, Vec, Vec);
//PetscErrorCode MatrixFreeHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode EnergyDensity(Vec ,Vec ,void *);
PetscReal DerivPot(PetscReal,AppCtx*);
PetscReal DerivPotp1(PetscReal,AppCtx*);
PetscReal Potp1(PetscReal,AppCtx*);


int main(int argc, char **argv){
  PetscInt    mx = 10; /* discretization in x-direction */
  PetscInt    my = 10; /* discretization in y-direction */
  Vec         x,E;       /* solution, energy density */
  PetscBool   flg;     /* A return value when checking for use options */
  Tao         tao;     /* Tao solver context */
  //Mat         H;       /* Hessian matrix */
  AppCtx      user;    /* application context */
  PetscMPIInt size;    /* number of processes */
  PetscReal one = 1.0;
  PetscViewer Eout;

    
  /* Initialize TAO,PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");


  /* Specify default parameters for the problem, check for command-line overrides */
  user.param_c0 = one;
  user.param_c2 = one;
  user.param_c4 = one;
  user.lambda   = one;
  user.region   = 10.0;
  user.hx   = 0.1;
  user.hy   = 0.1;
  user.param_lag = 1.0;
  user.itr      = 0;
  user.itrmax   = 100;
  user.normglobal = 0.0;
  user.newlag = 1.0;
  user.periodic = PETSC_FALSE;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-itrmax", &user.itrmax, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-h", &user.hx, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parc0", &user.param_c0, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parc2", &user.param_c2, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parc4", &user.param_c4, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &user.lambda, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-region", &user.region, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-paramlag", &user.param_lag, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-paramnewlag", &user.newlag, &flg));
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-periodic",&user.periodic, &flg));
  //PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_lmvm", &test_lmvm, &flg));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n---- Minimizing the energy functional -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "mx: %" PetscInt_FMT "     my: %" PetscInt_FMT "   \n\n", mx, my));
  user.ndim = mx * my;
  user.mx   = mx;
  user.my   = my;
  user.hy = user.hx;

  user.region = (mx - 1.0) / 2.0 * user.hx;

  /* old version
  user.hx   = 2 * user.region / (mx + 1);   //lattice size 
  user.hy   = 2 * user.region / (my + 1);
  */ 



  /* Allocate vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim*3, &x)); /* *3 for three fields */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ndim, &E)); 
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.itrmax, &user.chargeitr));
  PetscCall(VecDuplicate(user.chargeitr, &user.derrick));
  PetscCall(VecDuplicate(x, &user.gradient));
  //PetscCall(VecDuplicate(x, &user.y));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Set solution vector with an initial guess */
  PetscCall(FormInitialGuess(&user, x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjective(tao, FormFunction, (void *)&user));
  PetscCall(TaoSetGradient(tao, NULL, FormGradient , (void *)&user));
  //PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));

  /* Set Hessian */
  //PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, user.ndim, user.ndim, 5, NULL, &H));
  //PetscCall(MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE));
  //PetscCall(TaoSetHessian(tao, H, H, FormHessian, (void *)&user));


  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));
  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));

  /* compute energy density */

  PetscCall(EnergyDensity(x,E, (void *)&user));

  //PetscPrintf(MPI_COMM_WORLD,"nekoneko%o\n",i);

  /* Set Output files */
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "energydensity.dat", &Eout));
  PetscCall(PetscViewerPushFormat(Eout, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(E, Eout));
  PetscCall(PetscViewerPopFormat(Eout));
  PetscCall(PetscViewerDestroy(&Eout));

  PetscViewer resultfield,derrickview,chargeview,gradview;

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "resultfield.dat", &resultfield));
  PetscCall(PetscViewerPushFormat(resultfield, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(x, resultfield));
  PetscCall(PetscViewerPopFormat(resultfield));
  PetscCall(PetscViewerDestroy(&resultfield));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "chargeitr.dat", &chargeview));
  PetscCall(PetscViewerPushFormat(chargeview, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(user.chargeitr, chargeview));
  PetscCall(PetscViewerPopFormat(chargeview));
  PetscCall(PetscViewerDestroy(&chargeview));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "derrickitr.dat", &derrickview));
  PetscCall(PetscViewerPushFormat(derrickview, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(user.derrick, derrickview));
  PetscCall(PetscViewerPopFormat(derrickview));
  PetscCall(PetscViewerDestroy(&derrickview));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "gradient.dat", &gradview));
  PetscCall(PetscViewerPushFormat(gradview, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(user.gradient, gradview));
  PetscCall(PetscViewerPopFormat(gradview));
  PetscCall(PetscViewerDestroy(&gradview));
  /*
  char neko[10] = "taoresults",inu[5] = ".dat",*file;
  file = strcat(neko, inu);

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "taoresults.dat", &taoout));
  //PetscCall(PetscViewerPushFormat(taoout, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(TaoView(tao, taoout));
  PetscCall(PetscViewerPopFormat(taoout));
  PetscCall(PetscViewerDestroy(&taoout));
  */

  /*
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&user.s));
  PetscCall(VecDestroy(&user.y));
  */

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&E));

  PetscCall(PetscFinalize());
  return 0;
}

/* Evaluate potential term. In this problem, U(\phi) = \phi_3*/

PetscReal PotentialTerm(AppCtx *user ,PetscReal phi_3){
  //return  0.5 * user->param_c0 * (1 - phi_3) * (1 - phi_3);
  return 0.5 * user->param_c0 * phi_3 * phi_3 * 0.0;
}

PetscReal DerivPot(PetscReal phi3,AppCtx* user){
  //return user->param_c0 * (1 - phi3) * (-1);
  return user->param_c0 * phi3 * 0.0;
}

PetscReal Potp1(PetscReal phi1, AppCtx* user){
  return user->param_c0 * (1 - phi1);
}

PetscReal DerivPotp1(PetscReal phi1,AppCtx* user){
  return user->param_c0 * (-1); // initial boundary忘れずに
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
  PetscReal hx = user->hx, hy = user->hy, bound = user->region,l = user->lambda;
  PetscReal val1,val2,val3, x, y;
  PetscInt  i, j, k, n, m, nx = user->mx, ny = user->my;
  PetscViewer iniout;
  PetscRandom r;
  Vec rand;

  /* Compute initial guess */
  PetscFunctionBeginUser;

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  PetscCall(PetscRandomSetInterval(r,-1.0,1.0));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->ndim * 3, &rand)); 
  VecSetRandom(rand,r);
  VecScale(rand,0.01);

  /* randvec viewer
  PetscViewer randout;
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "randomvec.dat", &randout));
  PetscCall(PetscViewerPushFormat(randout, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(rand, randout));
  PetscCall(PetscViewerPopFormat(randout));
  PetscCall(PetscViewerDestroy(&randout));
  */

  /*  random test
  PetscRandom r;
  PetscReal value1;
  PetscRandomCreate(PETSC_COMM_WORLD,&r);
  PetscRandomSetType(r,PETSCRANDER48);
  PetscRandomGetValue(r,&value1);
  //PetscRandomGetValueReal(r,&value1);
  PetscRandomDestroy(&r); 
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"rand : %3.3e\n", (double)value1));
  */

  //bound = - nx / 2.0 * hx;
  for(j = 0; j < ny; j++){
    y = -1 * bound + j * hy;
    for(i = 0; i < nx; i++){

      k = ny * j + i;
      n = k + nx * ny;
      m = n + nx * ny; /* combine \phi1 \phi2 \phi3 into one vector X*/

      x = -1 * bound + i * hx; //except bound temporarily
      val1 = FuncComm(l,x,y) * (l * l * (x * x + y * y) - 1.0); 
      val2 = FuncComm(l,x,y) * 2.0 * l * x;
      val3 = FuncComm(l,x,y) * -2.0 * l * y;
      PetscCall(VecSetValues(X, 1, &k, &val1, ADD_VALUES));
      PetscCall(VecSetValues(X, 1, &n, &val2, ADD_VALUES));
      PetscCall(VecSetValues(X, 1, &m, &val3, ADD_VALUES));
      //PetscCall(PetscPrintf(MPI_COMM_WORLD,"val1 : %3.3e\n", (double)FuncComm(l,x,y)));
    }
  } 

  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "initialvec.dat", &iniout));
  PetscCall(PetscViewerPushFormat(iniout, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(VecView(X, iniout));
  PetscCall(PetscViewerPopFormat(iniout));
  PetscCall(PetscViewerDestroy(&iniout));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"X[20] : %3.5e\n", (double)val1));

  PetscCall(VecAXPY(X,0.0,rand));

  PetscCall(VecDestroy(&rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FuncComm(PetscReal Lambda, PetscReal x, PetscReal y){
  return (1.0 / (Lambda * Lambda * (x * x + y * y) + 1.0)); 
}

PetscReal boundary(PetscInt i, PetscInt j, PetscInt fieldNUM,Vec X,AppCtx *user) {
  PetscReal x,y,l,bound,hx,hy,temp;
  const PetscScalar *vec;
  PetscInt dim,nx,ny;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&vec));
  hx = user->hx;
  hy = user->hy;
  bound = user->region;
  l = user->lambda;
  x = -1 * bound + i * hx;
  y = -1 * bound + j * hy;
  nx  = user->mx;
  ny  = user->my;
  dim = nx * ny;

  if(j == -1){
    j = j + 1;
  }
  if(i == -1){
    i = i + 1;
  }
  if(j == ny){
    j = j - 1;
  }
  if(i == nx){
    i = i - 1;
  }

  if (fieldNUM == 1) {
    temp = vec[i + j * nx];
    PetscCall(VecRestoreArrayRead(X, &vec));
    //return FuncComm(l, x, y) * (l * l * (x * x + y * y) - 1.0);
    //return temp;
    return 1.0;
  } else if (fieldNUM == 2) {
    temp = vec[i + j * nx + dim];
    PetscCall(VecRestoreArrayRead(X, &vec));
    //return FuncComm(l, x, y) * 2.0 * l * x;
    //return temp;
    return 0.0;
  } else if (fieldNUM == 3) {
    temp = vec[i + j * nx + dim*2];
    PetscCall(VecRestoreArrayRead(X, &vec));
    //return FuncComm(l, x, y) * -2.0 * l * y;
    //return temp;
    return 0.0;
  } else {
    PetscPrintf(MPI_COMM_WORLD, "ERROR boundary");
    PetscCall(VecRestoreArrayRead(X, &vec));
    return dim + l + x + y;
  }
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
  AppCtx          *user = (AppCtx *)ptr;
  PetscReal         zero = 0.0, p5 = 0.5, val;
  PetscInt          nx = user->mx, ny = user->my, dim, i,j,k1,k2,k3,ind;
  PetscReal         hx = user->hx, hy = user->hy, lagmul = user->param_lag;
  PetscReal         vb, vl, vr, vt,v,v1,v2,v3;
  PetscReal         fskyrm = 0.0,fpot = 0.0,chargelc = 0.0,derrickCHK,flag = 0.0,fquad = 0.0;
  PetscReal         dp1dx, dp1dy,dp2dx,dp2dy,dp3dx,dp3dy,f12 = 0.0, area,pnorm;
  PetscScalar *x;

  PetscFunctionBeginUser;
  /* Initialize vector for gradient*/
  PetscCall(VecSet(G, zero));
  dim = nx * ny;
  area = hx * hy * p5;

  PetscCall(VecGetArray(X, &x));

  //lagmul = 0.0;
  PetscReal norm,regular;
  for(int i=0;i<dim;i++){
    norm = x[i] * x[i] + x[i+dim] * x[i+dim] + x[i+dim*2] * x[i+dim*2];
    regular = 1.0 / sqrt(norm);
    x[i] = x[i] * regular;
    x[i+dim] = x[i+dim] * regular;
    x[i+dim*2] = x[i+dim*2] * regular;
    norm = x[i] * x[i] + x[i+dim] * x[i+dim] + x[i+dim*2] * x[i+dim*2];
   //PetscCall(PetscPrintf(MPI_COMM_WORLD,"%2.3e\n",norm));
  }

  /*Lower triangular contiribution*/
  for (j = -1; j < ny; j++) {
    for (i = -1; i < nx; i++) {
      k1  = nx * j + i;
      k2  = nx * j + i + dim;
      k3  = nx * j + i + dim * 2;
      /* ||\nabla /phi_1||^2 */
      v   = boundary(i, j, 1, X,user);
      vr  = boundary(i, j, 1, X,user);
      vt  = boundary(i, j, 1, X,user);
      
      if (i >= 0 && j >= 0) v = x[k1];
      if (i < nx - 1 && j > -1) vr = x[k1 + 1];
      if (i > -1 && j < ny - 1) vt = x[k1 + nx];
      dp1dx = (vr - v) / hx;
      dp1dy = (vt - v) / hy;
      v1 = v;
      fquad += dp1dx * dp1dx + dp1dy * dp1dy;

      /* ||\nabla /phi_2||^2 */
      v   = boundary(i, j, 2, X, user);
      vr  = boundary(i, j, 2, X, user);
      vt  = boundary(i, j, 2, X, user);

      if (i >= 0 && j >= 0) v = x[k2];
      if (i < nx - 1 && j > -1) vr = x[k2 + 1];
      if (i > -1 && j < ny - 1) vt = x[k2 + nx];
      dp2dx = (vr - v) / hx;
      dp2dy = (vt - v) / hy;
      v2 = v;
      fquad += dp2dx * dp2dx + dp2dy * dp2dy;

      /* ||\nabla /phi_3||^2 */
      v   = boundary(i, j,3,X, user);
      vr  = boundary(i, j,3,X, user);
      vt  = boundary(i, j,3,X, user);

      if (i >= 0 && j >= 0) v = x[k3];
      if (i < nx - 1 && j > -1) vr = x[k3 + 1];
      if (i > -1 && j < ny - 1) vt = x[k3 + nx];
      dp3dx = (vr - v) / hx;
      dp3dy = (vt - v) / hy;
      v3 = v;
      fquad += dp3dx * dp3dx + dp3dy * dp3dy;

      pnorm = v1 * v1 + v2 * v2 + v3 * v3;
      flag += 1 - pnorm;

      f12 = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12 -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;

      // following variable is derivative of skyrm term
      PetscReal ddeddp1dx,ddeddp1dy,ddeddp2dx,ddeddp2dy,ddeddp3dx,ddeddp3dy;
      PetscReal dedp1,dedp2,dedp3;
      PetscReal ddlagp1dx,ddlagp1dy,ddlagp2dx,ddlagp2dy,ddlagp3dx,ddlagp3dy,dlagp1,dlagp2,dlagp3;
      ddeddp1dx = (user->param_c4 * f12 * (v3 * dp2dy - v2 * dp3dy) ) / hx;
      ddeddp1dy = (user->param_c4 * f12 * (v2 * dp3dx - v3 * dp2dx) ) / hy;
      ddeddp2dx = (user->param_c4 * f12 * (v1 * dp3dy - v3 * dp1dy) ) / hx;
      ddeddp2dy = (user->param_c4 * f12 * (v3 * dp1dx - v1 * dp3dx) ) / hy;
      ddeddp3dx = (user->param_c4 * f12 * (v2 * dp1dy - v1 * dp2dy) ) / hx;
      ddeddp3dy = (user->param_c4 * f12 * (v1 * dp2dx - v2 * dp1dx) ) / hy;
      dedp1     =  user->param_c4 * f12 * (dp2dx * dp3dy - dp3dx * dp2dy);
      dedp2     =  user->param_c4 * f12 * (dp3dx * dp1dy - dp1dx * dp3dy);
      dedp3     =  user->param_c4 * f12 * (dp1dx * dp2dy - dp2dx * dp1dy);
      ddlagp1dx = user->newlag * (ddeddp1dx + dp1dx / hx); 
      ddlagp1dy = user->newlag * (ddeddp1dy + dp1dy / hy);
      ddlagp2dx = user->newlag * (ddeddp2dx + dp2dx / hx);
      ddlagp2dy = user->newlag * (ddeddp2dy + dp2dy / hy);
      ddlagp3dx = user->newlag * (ddeddp3dx + dp3dx / hx);
      ddlagp3dy = user->newlag * (ddeddp3dy + dp3dy / hy);
      dlagp1    = user->newlag * dedp1 / 3.0;
      dlagp2    = user->newlag * dedp2 / 3.0;
      dlagp3    = user->newlag * (dedp3 + DerivPot(v3, user)) / 3.0;
      /* phi_1 */ /* param-c2 is not working for this gradient TODO */
      if (i != -1 && j != -1) {
        ind = k1;
        val = -dp1dx / hx - dp1dy / hy - ddeddp1dx - ddeddp1dy + (dedp1 + /*v1 * (v1 * (dp1dx / hx + dp1dy / hy) + v2 * (dp2dx / hx + dp2dy / hy) + v3 * (dp3dx / hx + dp3dy / hy)) */2 * lagmul * v1 + DerivPotp1(v1, user)) / 3.0 + v1 * ( v1 * (ddlagp1dx + ddlagp1dy - dlagp1) + v2 * (ddlagp2dx + ddlagp2dy - dlagp2) + v3 * (ddlagp3dx + ddlagp3dy - dlagp3)); /* last term is lagrange multiplier term */
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != nx - 1 && j != -1) {
        ind = k1 + 1;
        val = dp1dx / hx + ddeddp1dx + (dedp1 + /* v1 * (- v1 * (dp1dx / hx) - v2 * (dp2dx / hx) - v3 * (dp3dx / hx ))*/ 2 * lagmul * v1 + DerivPotp1(v1, user)) / 3.0 + v1 * ( v1 * (-ddlagp1dx  - dlagp1) + v2 * (-ddlagp2dx - dlagp2) + v3 * (-ddlagp3dx - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != -1 && j != ny - 1) {
        ind = k1 + nx;
        val = dp1dy / hy + ddeddp1dy + (dedp1 + /* v1 * (- v1 * (dp1dy / hy) - v2 * (dp2dy / hy) - v3 * (dp3dy / hy )) */2 * lagmul * v1 + DerivPotp1(v1, user)) / 3.0 + v1 * ( v1 * (-ddlagp1dy - dlagp1) + v2 * (-ddlagp2dy - dlagp2) + v3 * (-ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      /* phi_2 */
      if (i != -1 && j != -1) {
        ind = k2;
        val = -dp2dx / hx - dp2dy / hy - ddeddp2dx - ddeddp2dy + (dedp2 + 2 * lagmul * v2) / 3.0 + v2 * ( v1 * (ddlagp1dx + ddlagp1dy - dlagp1) + v2 * (ddlagp2dx + ddlagp2dy - dlagp2) + v3 * (ddlagp3dx + ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != nx - 1 && j != -1) {
        ind = k2 + 1;
        val = dp2dx / hx + ddeddp2dx + (dedp2 + 2 * lagmul * v2) / 3.0 + v2 * ( v1 * (-ddlagp1dx - dlagp1) + v2 * (-ddlagp2dx - dlagp2) + v3 * (-ddlagp3dx- dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != -1 && j != ny - 1) {
        ind = k2 + nx;
        val = dp2dy / hy + ddeddp2dy + (dedp2 + 2 * lagmul * v2) / 3.0 + v2 * ( v1 * (-ddlagp1dy - dlagp1) + v2 * (-ddlagp2dy - dlagp2) + v3 * (-ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      /* phi_3 */
      if (i != -1 && j != -1) {
        ind = k3;
        val = -dp3dx / hx - dp3dy / hy - ddeddp3dx - ddeddp3dy + (dedp3 + DerivPot(v3,user) + 2 * lagmul * v3) /3.0 + v3 * ( v1 * (ddlagp1dx + ddlagp1dy - dlagp1) + v2 * (ddlagp2dx + ddlagp2dy - dlagp2) + v3 * (ddlagp3dx + ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != nx - 1 && j != -1) {
        ind = k3 + 1;
        val = dp3dx / hx + ddeddp3dx + (dedp3 + DerivPot(v3,user) + 2 * lagmul * v3) / 3.0 + v3 * ( v1 * (-ddlagp1dx - dlagp1) + v2 * (-ddlagp2dx - dlagp2) + v3 * (-ddlagp3dx - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != -1 && j != ny - 1) {
        ind = k3 + nx;
        val = dp3dy / hy + ddeddp3dy + (dedp3 + DerivPot(v3,user) + 2 * lagmul * v3) / 3.0 + v3 * ( v1 * (-ddlagp1dy - dlagp1) + v2 * (-ddlagp2dy - dlagp2) + v3 * (-ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      fskyrm += f12*f12;
      chargelc += f12;
      f12 = 0;
      fpot += PotentialTerm(user, v3) + Potp1(v1, user);
    }
  }

  /* upper triangle*/

  for (j = 0; j <= ny; j++) {
    for (i = 0; i <= nx; i++) {
      k1  = nx * j + i;
      k2  = nx * j + i + dim;
      k3  = nx * j + i + dim * 2;
      /* ||\nabla /phi_1||^2 */
      vb = boundary(i, j,1,X, user);
      vl = boundary(i, j,1,X, user);
      v  = boundary(i, j,1,X, user);

      if (i < nx && j > 0) vb = x[k1 - nx];
      if (i > 0 && j < ny) vl = x[k1 - 1];
      if (i < nx && j < ny) v = x[k1];
      dp1dx = (v - vl) / hx;
      dp1dy = (v - vb) / hy;
      v1 = v;
      fquad += dp1dx * dp1dx + dp1dy * dp1dy;

      /* ||\nabla /phi_2||^2 */
      vb = boundary(i, j, 2, X, user);
      vl = boundary(i, j, 2, X, user);
      v  = boundary(i, j, 2, X,user);
      

      if (i < nx && j > 0) vb = x[k2 - nx];
      if (i > 0 && j < ny) vl = x[k2 - 1];
      if (i < nx && j < ny) v = x[k2];
      dp2dx = (v - vl) / hx;
      dp2dy = (v - vb) / hy;
      v2 = v;
      fquad += dp2dx * dp2dx + dp2dy * dp2dy;


      /* ||\nabla /phi_3||^2 */
      vb = boundary(i, j, 3, X,user);
      vl = boundary(i, j, 3, X,user);
      v  = boundary(i, j, 3, X,user);
      
      if (i < nx && j > 0) vb = x[k3 - nx];
      if (i > 0 && j < ny) vl = x[k3 - 1];
      if (i < nx && j < ny) v = x[k3];
      dp3dx = (v - vl) / hx;
      dp3dy = (v - vb) / hy;
      v3 = v;
      fquad += dp3dx * dp3dx + dp3dy * dp3dy;

      pnorm = v1 * v1 + v2 * v2 + v3 * v3;

      flag += 1 - pnorm;

      f12 = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12 -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;

      // following variable stands derivative of skyrm term
      PetscReal ddeddp1dx,ddeddp1dy,ddeddp2dx,ddeddp2dy,ddeddp3dx,ddeddp3dy;
      PetscReal dedp1,dedp2,dedp3;
      PetscReal ddlagp1dx,ddlagp1dy,ddlagp2dx,ddlagp2dy,ddlagp3dx,ddlagp3dy,dlagp1,dlagp2,dlagp3;
      ddeddp1dx = (user->param_c4 * f12 * (v3 * dp2dy - v2 * dp3dy) ) / hx;
      ddeddp1dy = (user->param_c4 * f12 * (v2 * dp3dx - v3 * dp2dx) ) / hy;
      ddeddp2dx = (user->param_c4 * f12 * (v1 * dp3dy - v3 * dp1dy) ) / hx;
      ddeddp2dy = (user->param_c4 * f12 * (v3 * dp1dx - v1 * dp3dx) ) / hy;
      ddeddp3dx = (user->param_c4 * f12 * (v2 * dp1dy - v1 * dp2dy) ) / hx;
      ddeddp3dy = (user->param_c4 * f12 * (v1 * dp2dx - v2 * dp1dx) ) / hy;
      dedp1     =  user->param_c4 * f12 * (dp2dx * dp3dy - dp3dx * dp2dy);
      dedp2     =  user->param_c4 * f12 * (dp3dx * dp1dy - dp1dx * dp3dy);
      dedp3     =  user->param_c4 * f12 * (dp1dx * dp2dy - dp2dx * dp1dy);
      ddlagp1dx = user->newlag * (ddeddp1dx + dp1dx / hx); 
      ddlagp1dy = user->newlag * (ddeddp1dy + dp1dy / hy);
      ddlagp2dx = user->newlag * (ddeddp2dx + dp2dx / hx);
      ddlagp2dy = user->newlag * (ddeddp2dy + dp2dy / hy);
      ddlagp3dx = user->newlag * (ddeddp3dx + dp3dx / hx);
      ddlagp3dy = user->newlag * (ddeddp3dy + dp3dy / hy);
      dlagp1    = user->newlag * dedp1 / 3.0;
      dlagp2    = user->newlag * dedp2 / 3.0;
      dlagp3    = user->newlag * (dedp3 + v3) / 3.0;
      /* phi_1 */
      if (i != nx && j != 0) {
        ind = k1 - nx;
        val = -dp1dy / hy - ddeddp1dy + (dedp1 + /* v1 * (v1 * (dp1dy / hy) + v2 * (dp2dy / hy) + v3 * (dp3dy / hy))*/ + 2 * v1 + DerivPotp1(v1, user)) / 3.0 + v1 * ( v1 * (ddlagp1dy - dlagp1) + v2 * (ddlagp2dy - dlagp2) + v3 * (ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != 0 && j != ny) {
        ind = k1 - 1;
        val = -dp1dx / hx - ddeddp1dx + (dedp1 + /* v1 * (v1 * (dp1dx / hx ) + v2 * (dp2dx / hx) + v3 * (dp3dx / hx))*/ + 2 * lagmul * v1  + DerivPotp1(v1, user)) / 3.0 + v1 * ( v1 * (ddlagp1dx - dlagp1) + v2 * (ddlagp2dx - dlagp2) + v3 * (ddlagp3dx - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != nx && j != ny) {
        ind = k1;
        val = dp1dx / hx + dp1dy / hy + ddeddp1dx + ddeddp1dy + (dedp1 + /* - v1 * (v1 * (dp1dx / hx + dp1dy / hy) + v2 * (dp2dx / hx + dp2dy / hy) + v3 * (dp3dx / hx + dp3dy / hy)) */ + 2 * lagmul * v1 + DerivPotp1(v1, user)) / 3.0 + v1 * ( v1 * (-ddlagp1dx - ddlagp1dy - dlagp1) + v2 * (-ddlagp2dx - ddlagp2dy - dlagp2) + v3 * (-ddlagp3dx - ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }

      /* phi_2 */

      if (i != nx && j != 0) {
        ind = k2 - nx;
        val = -dp2dy / hy - ddeddp2dy + (dedp2 + 2 * lagmul * v2) / 3.0 + v2 * ( v1 * (ddlagp1dy - dlagp1) + v2 * (ddlagp2dy - dlagp2) + v3 * (ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != 0 && j != ny) {
        ind = k2 - 1;
        val = -dp2dx / hx - ddeddp2dx + (dedp2 + 2 * lagmul * v2) / 3.0 + v2 * ( v1 * (ddlagp1dx - dlagp1) + v2 * (ddlagp2dx - dlagp2) + v3 * (ddlagp3dx - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != nx && j != ny) {
        ind = k2;
        val = dp2dx / hx + dp2dy / hy + ddeddp2dx + ddeddp2dy + (dedp2 + 2 * lagmul * v2) / 3.0 + v3 * ( v1 * (-ddlagp1dx - ddlagp1dy - dlagp1) + v2 * (-ddlagp2dx - ddlagp2dy - dlagp2) + v3 * (-ddlagp3dx - ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }

      /* phi_3 */

      if (i != nx && j != 0) {
        ind = k3 - nx;
        val = -dp3dy / hy - ddeddp3dy + (dedp3 + DerivPot(v3, user) + 2 * lagmul * v3) / 3.0 + v3 * ( v1 * (ddlagp1dy - dlagp1) + v2 * (ddlagp2dy - dlagp2) + v3 * (ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != 0 && j != ny) {
        ind = k3 - 1;
        val = -dp3dx / hx - ddeddp3dx + (dedp3 + DerivPot(v3,user) + 2 * lagmul * v3) / 3.0 + ( v1 * (ddlagp1dx - dlagp1) + v2 * (ddlagp2dx - dlagp2) + v3 * (ddlagp3dx - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      if (i != nx && j != ny) {
        ind = k3;
        val = dp3dx / hx + dp3dy / hy + ddeddp3dx + ddeddp3dy + (dedp3 + DerivPot(v3,user) + 2 * lagmul * v3) / 3.0 + v3 * ( v1 * (-ddlagp1dx - ddlagp1dy - dlagp1) + v2 * (-ddlagp2dx - ddlagp2dy - dlagp2) + v3 * (-ddlagp3dx - ddlagp3dy - dlagp3));
        PetscCall(VecSetValues(G, 1, &ind, &val, ADD_VALUES));
      }
      fskyrm += f12*f12;
      chargelc += f12;
      f12 = 0;
      fpot += PotentialTerm(user, v3) + Potp1(v1, user);
    }
  }  
  const double PI = 3.1415926535;
  fquad  = user->param_c2 * fquad / 2.0;
  fskyrm = p5 * user->param_c4 * fskyrm;
  flag = lagmul * flag;
  area = p5 * hx * hy;
  *f   = area * (fquad + fskyrm + fpot + flag);
  chargelc = area * chargelc / ( 4.0 * PI );
  derrickCHK = fskyrm - fpot;
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"charge : %2.3e\n", (double)chargeCheck));

  PetscCall(PetscPrintf(MPI_COMM_WORLD,"----------------------------\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"fpot(E0)           : %2.3e\n", (double)fpot * (double)area));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"fskyrme(E4)        : %2.3e\n", (double)fskyrm * (double)area));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"fskyrme - fpot : %2.3e\n", (double)derrickCHK));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"flag           : %2.3e\n", (double)flag * (double)area));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"fquad          : %2.3e\n", (double)fquad * (double)area));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"charge         : %e\n", (double)chargelc));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"funcvalue      : %2.3e\n", (double)*f));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"pnorm : %2.3e\n", (double)pnorm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"----------------------------\n"));
  /* Assemble Vector */
  PetscCall(VecAssemblyBegin(G));
  PetscCall(VecAssemblyEnd(G));
  PetscCall(VecRestoreArray(X, &x));

  area = p5 * hx * hy;
  PetscCall(VecScale(G, area));
  //PetscCall(VecScale(G, 0.0001));
  PetscCall(VecCopy(G,user->gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  EnergyDensity(Vec X,Vec E,void *ptr){
    compute energy density of this system;
    Input 
    X    : input Vector (value of \phi on each lattice)
    *ptr : user context 
    Output
    E : output vector (energydensity of each point)


*/

PetscErrorCode EnergyDensity(Vec X,Vec E,void *ptr){
  AppCtx            *user = (AppCtx *)ptr;
  PetscReal          hx = user->hx, hy = user->hy, p5 = 0.5;
  PetscReal          vr, vt,vl,vb, dp1dx, dp1dy,dp2dx,dp2dy,dp3dx,dp3dy, fquad = 0.0, f12 = 0.0 ,fskyrm = 0.0, fpot = 0.0,flag = 0.0;
  PetscReal          v,v1,v2,v3,val,pnorm,charge = 0.0,area, PI = 3.1415926535;//, cdiv3 = user->param / three;
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
      v  = boundary(i, j, 1, X,user);
      vr = boundary(i, j, 1, X,user);
      vt = boundary(i, j, 1, X,user);
      /* boundary new (periodic)*/
      if(user->periodic){ 
      if(i != -1 && j == -1){
        v = x[k1 + nx * nx];
      }
      else if(i == -1 && j == -1){
        v = x[nx * nx - 1]; //例外
      }
      else if(i == -1 && j != -1){
        v = x[k1 + nx];
      }

      if(i != nx - 1 && j == -1){
        vr = x[k1 + nx * nx + 1];
      }
      else if(i == nx - 1 && j == -1){
        vr = x[nx * nx - nx]; //ここが例外
      }
      else if(i == nx - 1 && j != -1){
        vr = x[k1 + 1 - nx];
      }

      if(i == -1 && j != ny - 1){
        vt = x[k1 + nx * 2]; // ?
      }
      else if(i == -1 && j == ny - 1){
        vt = x[nx - 1];
      }
      else if(i != -1 && j == ny - 1){
        vt = x[k1 - nx * (nx - 1)];
      }
      }
      if (i >= 0 && j >= 0) v = x[k1];
      if (i < nx - 1 && j > -1) vr = x[k1 + 1];
      if (i > -1 && j < ny - 1) vt = x[k1 + nx];
      dp1dx = (vr - v) / hx;
      dp1dy = (vt - v) / hy;
      fquad = dp1dx * dp1dx + dp1dy * dp1dy;
      v1 = v;

      /* ||\nabla /phi_2||^2 */
      v  = boundary(i, j, 2, X,user);
      vr = boundary(i, j, 2, X,user);
      vt = boundary(i, j, 2, X,user);
      if(user->periodic){
      if(i != -1 && j == -1){
        v = x[k2 + nx * nx];
      }
      else if(i == -1 && j == -1){
        v = -x[nx * nx - 1 + dim]; //例外
      }
      else if(i == -1 && j != -1){
        v = -x[k2 + nx];
      }

      if(i != nx - 1 && j == -1){
        vr = x[k2 + nx * nx + 1];
      }
      else if(i == nx - 1 && j == -1){
        vr = -x[nx * nx - nx + dim]; //ここが例外
      }
      else if(i == nx - 1 && j != -1){
        vr = -x[k2 + 1 - nx];
      }

      if(i == -1 && j != ny - 1){
        vt = -x[k2 + nx * 2]; // ?
      }
      else if(i == -1 && j == ny - 1){
        vt = -x[nx - 1 + dim];
      }
      else if(i != -1 && j == ny - 1){
        vt = x[k2 - nx * (nx - 1)];
      }
      }

      if (i >= 0 && j >= 0) v = x[k2];
      if (i < nx - 1 && j > -1) vr = x[k2 + 1];
      if (i > -1 && j < ny - 1) vt = x[k2 + nx];
      dp2dx = (vr - v) / hx;
      dp2dy = (vt - v) / hy;
      fquad += dp2dx * dp2dx + dp2dy * dp2dy;
      v2 = v;

      /* ||\nabla /phi_3||^2 */
      v  = boundary(i, j, 3, X,user);
      vr = boundary(i, j, 3, X,user);
      vt = boundary(i, j, 3, X,user);
      if(user->periodic){
      if(i != -1 && j == -1){
        v = -x[k3 + nx * nx];
      }
      else if(i == -1 && j == -1){
        v = -x[nx * nx - 1 + dim*2]; //例外
      }
      else if(i == -1 && j != -1){
        v = x[k3 + nx];
      }

      if(i != nx - 1 && j == -1){
        vr = -x[k3 + nx * nx + 1];
      }
      else if(i == nx - 1 && j == -1){
        vr = -x[nx * nx - nx + dim*2]; //ここが例外
      }
      else if(i == nx - 1 && j != -1){
        vr = x[k3 + 1 - nx];
      }

      if(i == -1 && j != ny - 1){
        vt = x[k3 + nx * 2]; // ?
      }
      else if(i == -1 && j == ny - 1){
        vt = -x[nx - 1 + dim*2];
      }
      else if(i != -1 && j == ny - 1){
        vt = -x[k3 - nx * (nx - 1)];
      }
      }

      if (i >= 0 && j >= 0) v = x[k3];
      if (i < nx - 1 && j > -1) vr = x[k3 + 1];
      if (i > -1 && j < ny - 1) vt = x[k3 + nx];
      dp3dx = (vr - v) / hx;
      dp3dy = (vt - v) / hy;
      v3 = v;

      fquad += dp3dx * dp3dx + dp3dy * dp3dy;
      pnorm  = v1 * v1 + v2 * v2 + v3 * v3;
      flag = (pnorm - 1) * (pnorm - 1);
      // fquad = user->param_c2 * fquad / 2.0; after summing all, do this

      f12 = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12 -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;
      fskyrm = f12 * f12;
      //f12 = 0.0;

      fpot = PotentialTerm(user, v3) + Potp1(v1, user);
      fquad  = user->param_c2 * fquad / 2.0;
      fskyrm = p5 * user->param_c4 * fskyrm;
      flag = user->param_lag * flag;

      val = 0;
      ind = k1;
      if( j >= 0 && i >= 0){
        val = fquad + fskyrm + fpot + flag;
        PetscCall(VecSetValues(E, 1, &ind, &val, INSERT_VALUES));
      }
      if(i == 0 && j == 0){
       //PetscCall(PetscPrintf(MPI_COMM_WORLD,"lowertest leftbottom: %2.3e\n", fquad));
      }
      if(i == nx - 1 && j == ny - 1){
        PetscCall(PetscPrintf(MPI_COMM_WORLD,"lowertest righttop : %2.3e\n", fquad));
      }

      charge += f12;
    }
  }
  #pragma omp parallel for private(i,j)
  for (j = 0; j <= ny; j++) {
    for (i = 0; i <= nx; i++) {
      k1  = nx * j + i;
      k2  = nx * j + i + dim;
      k3  = nx * j + i + dim * 2;
      /* ||\nabla /phi_1||^2 */

      /*old boundary*/
      /*
      vb = boundary(i, j, 1, user);
      vl = boundary(i, j, 1, user);
      v  = boundary(i, j, 1, user);
      */

      v   = boundary(i, j, 1, X,user);
      vl  = boundary(i, j, 1, X,user);
      vb  = boundary(i, j, 1, X,user);

      /* periodic condition */
      if(user->periodic){
      if(j == 0 && i != nx){
        vb = x[k1 + ny * (ny - 1)];
      }
      else if(j == 0 && i == nx){
        vb = x[ny * (ny - 1)];
      } 
      else if(j != 0 && i == nx){
        vb = x[k1 - nx];
      }

      if(i == 0 && j != ny){
        vl = x[k1 + nx - 1];
      }
      else if(i == 0 && j == ny){
        vl = x[nx - 1];
      }
      else if(i != 0 && j == ny){
        vl = x[k1 - ny * ny - 1];
      }

      if(i == nx && j != ny){
        v = x[k1 - nx];
      }
      else if(i != nx && j == ny){
        v = x[k1 - ny * ny];
      }
      else if(i == nx && j == ny){
        v = x[0];
      }
      }

      if (i < nx && j > 0) vb = x[k1 - nx];
      if (i > 0 && j < ny) vl = x[k1 - 1];
      if (i < nx && j < ny) v = x[k1];
      dp1dx = (v - vl) / hx;
      dp1dy = (v - vb) / hy;
      fquad = dp1dx * dp1dx + dp1dy * dp1dy;
      v1 = v;

      /* ||\nabla /phi_2||^2 */
      v   = boundary(i, j, 2, X,user);
      vl  = boundary(i, j, 2, X,user);
      vb  = boundary(i, j, 2, X,user);
      /* periodic condition */
      if(user->periodic){
      if(j == 0 && i != nx){
        vb = x[k2 + ny * (ny - 1)];
      }
      else if(j == 0 && i == nx){
        vb = -x[ny * (ny - 1) + dim];
      } 
      else if(j != 0 && i == nx){
        vb = -x[k2 - nx];
      }

      if(i == 0 && j != ny){
        vl = -x[k2 + nx - 1];
      }
      else if(i == 0 && j == ny){
        vl = -x[nx - 1 + dim];
      }
      else if(i != 0 && j == ny){
        vl = x[k2 - ny * ny - 1];
      }

      if(i == nx && j != ny){
        v = -x[k2 - nx];
      }
      else if(i != nx && j == ny){
        v = x[k2 - ny * ny];
      }
      else if(i == nx && j == ny){
        v = -x[dim];
      }
      }

      if (i < nx && j > 0) vb = x[k2 - nx];
      if (i > 0 && j < ny) vl = x[k2 - 1];
      if (i < nx && j < ny) v = x[k2];
      dp2dx = (v - vl) / hx;
      dp2dy = (v - vb) / hy;
      fquad += dp2dx * dp2dx + dp2dy * dp2dy;
      v2 = v;

      /* ||\nabla /phi_3||^2 */
      v   = boundary(i, j, 3, X,user);
      vl  = boundary(i, j, 3, X,user);
      vb  = boundary(i, j, 3, X,user);
      /* periodic condition */
      if(user->periodic){
      if(j == 0 && i != nx){
        vb = -x[k3 + ny * (ny - 1)];
      }
      else if(j == 0 && i == nx){
        vb = -x[ny * (ny - 1) + dim*2];
      } 
      else if(j != 0 && i == nx){
        vb = x[k3 - nx];
      }

      if(i == 0 && j != ny){
        vl = x[k3 + nx - 1];
      }
      else if(i == 0 && j == ny){
        vl = -x[nx - 1 + dim*2];
      }
      else if(i != 0 && j == ny){
        vl = -x[k3 - ny * ny - 1];
      }

      if(i == nx && j != ny){
        v = x[k3 - nx];
      }
      else if(i != nx && j == ny){
        v = -x[k3 - ny * ny];
      }
      else if(i == nx && j == ny){
        v = -x[dim*2];
      }
      }

      if (i < nx && j > 0) vb = x[k3 - nx];
      if (i > 0 && j < ny) vl = x[k3 - 1];
      if (i < nx && j < ny) v = x[k3];
      dp3dx = (v - vl) / hx;
      dp3dy = (v - vb) / hy;
      fquad += dp3dx * dp3dx + dp3dy * dp3dy;
      v3 = v;

      pnorm  = v1 * v1 + v2 * v2 + v3 * v3;
      flag += (pnorm - 1) * (pnorm - 1);
      //chargelc += v1 * (dp2dx * dp3dy - dp3dx * dp2dy) + v2 * (dp3dx * dp1dy - dp1dx * dp3dy) + v3 * (dp1dx * dp2dy - dp2dx * dp1dy);
      if(j % 10 == 0 && i % 10 == 0){
        //PetscCall(PetscPrintf(MPI_COMM_WORLD,"pnorm : %2.3e\n", (double)pnorm));
      }

      f12 = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12 -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;
      fskyrm = f12*f12;
      f12 = 0;
      fpot = PotentialTerm(user, v3) + Potp1(v1, user);

      fquad  = user->param_c2 * fquad / 2.0;
      fskyrm = p5 * user->param_c4 * fskyrm;
      flag = user->param_lag * flag;

      val = 0;
      ind = k1;
      if( j < nx && i < ny){
        val = fquad + fskyrm + fpot + flag;
        PetscCall(VecSetValues(E, 1, &ind, &val, ADD_VALUES));
      }
      if(i == 0 && j == 0){
        PetscCall(PetscPrintf(MPI_COMM_WORLD,"uppertest leftbottom: %2.3e-%2.3e\n", vl,v));
        PetscCall(PetscPrintf(MPI_COMM_WORLD,"uppertest leftbottom: %2.3e\n", fquad));
      }
      if(i == nx - 1 && j == ny - 1){
        //PetscCall(PetscPrintf(MPI_COMM_WORLD,"uppertest righttop : %2.3e\n", fquad));
      }
      fpot = 0;
      flag = 0;
      fquad= 0;
    }
  }

  PetscCall(VecScale(E,p5));

  /* charge calculation */
  area = p5 * hx * hy * 2;
  charge = area * charge / ( 4 * PI );
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"charge : %2.3e\n", (double)charge));
  /*
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"fquad : %2.3e\n", (double)fquad));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"dp3dx : %2.3e\n", (double)dp3dx));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"dp3dy : %2.3e\n", (double)dp3dy));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"dp1dx : %2.3e\n", (double)dp1dx));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"dp1dy : %2.3e\n", (double)dp1dy));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"dp2dx : %2.3e\n", (double)dp2dx));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"dp2dy : %2.3e\n", (double)dp2dy));
  */

  /* assemble vector */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecAssemblyBegin(E));
  PetscCall(VecAssemblyEnd(E));

  //area = p5 * hx * hy;
  //*f   = area * (fquad + fskyrm + fpot);

  PetscFunctionReturn(PETSC_SUCCESS);
}
