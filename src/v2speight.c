/*
Now we can use mpi in this program...
*/

#include "mpi.h"
#include "petscdm.h"
#include "petscdmtypes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petsctao.h"
#include "petscdmda.h"
#include "petscvec.h"
#include "petscviewer.h"

static char help[] = "this program minimumize some energy functional. -mx and -my is option for lattice size";

typedef struct {
  PetscReal param_c2, param_c4, param_c0;      /* model parameters */
  PetscReal lambda;     /* parameter for initial configuration */
  PetscReal region;     /* Region size parameter */
  PetscInt  mx, my;     /* discretization in x- and y-directions */
  PetscInt  ndim;
  Vec       gradient;              /* viewer for gradient vec */
  //Vec       s, y, xvec; /* work space for computing Hessian (?)*/
  PetscReal hx, hy;     /* mesh spacing in x- and y-directions */
  PetscReal param_lag,normglobal,newlag;  /* lagrange multiplier */
  PetscBool periodic;
  DM        dm,dmE;        /* dm object */
  Vec       localX,localE;    /* local sol vec*/
} AppCtx;

PetscErrorCode FormInitialGuess(AppCtx * ,Vec ); /*\phi_1 \phi_2 \phi_3*/
PetscErrorCode FormFunction(Tao, Vec, PetscReal *, void *);
PetscReal      FuncComm(PetscReal, PetscReal, PetscReal );
PetscErrorCode FormGradient(Tao, Vec, Vec, void *);
//PetscErrorCode MatrixFreeHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode EnergyDensity(Vec ,Vec ,void *);
PetscReal DerivPot(PetscReal,AppCtx*);
PetscReal DerivPotp1(PetscReal,AppCtx*);
PetscReal Potp1(PetscReal,AppCtx*);

int main(int argc, char **argv){
  Vec         x,E;       /* solution, energy density */
  PetscBool   flg;     /* A return value when checking for use options */
  Tao         tao;     /* Tao solver context */
  //Mat         H;       /* Hessian matrix */
  AppCtx      user;    /* application context */
  PetscViewer Eout;    /*viewer for energy density*/

    
  /* Initialize TAO,PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Set parameters */ 
  PetscInt    Nx = PETSC_DECIDE,Ny = PETSC_DECIDE; /* for dm */
  PetscReal one = 1.0;
  user.param_c0 = one;
  user.param_c2 = one;
  user.param_c4 = one;
  user.lambda   = one;
  user.mx       = 10;
  user.my       = 10;
  user.region   = 10.0;
  user.hx   = 0.1;
  user.hy   = 0.1;
  user.param_lag = 1.0;
  user.normglobal = 0.0;
  user.newlag = 1.0;
  user.periodic = PETSC_FALSE;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &user.my, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &user.mx, &flg));
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

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n---- Minimumizing the energy functional -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mx: %" PetscInt_FMT "     my: %" PetscInt_FMT "   \n\n", user.mx, user.my));
  user.ndim = user.mx * user.my;
  user.hy = user.hx;
  user.region = (user.mx - 1.0) / 2.0 * user.hx;

  /* Set up DMDA */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, user.mx, user.my, Nx, Ny, 3, 1, NULL, NULL, &user.dm));
  PetscCall(DMSetFromOptions(user.dm));
  PetscCall(DMSetUp(user.dm));
  PetscCall(DMDASetFieldName(user.dm,0,"phi1"));
  PetscCall(DMDASetFieldName(user.dm,1,"phi2"));
  PetscCall(DMDASetFieldName(user.dm,2,"phi3"));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, user.mx, user.my, Nx, Ny, 1, 1, NULL, NULL, &user.dmE));

  /* Create vectors */
  PetscCall(DMCreateGlobalVector(user.dm,&x));
  PetscCall(DMCreateLocalVector(user.dm,&user.localX));
  PetscCall(DMCreateGlobalVector(user.dmE,&E));
  PetscCall(DMCreateLocalVector(user.dmE,&user.localE));

  /* Set up TAO */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao,TAOBQNLS));

  /* Set solution vector */
  PetscCall(FormInitialGuess(&user, x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routine for function and gradient evalution */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));

  /* Check tao options from command line */
  PetscCall(TaoSetFromOptions(tao));

  /* Solve!! */
  PetscCall(TaoSolve(tao));

  /* Free tao data */
  PetscCall(TaoDestroy(&tao));

  /* Compute energy density */
  PetscCall(EnergyDensity(x, E, (void *)&user));

  /* Free other data */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&user.localX));
  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&user.localE));
  PetscCall(DMDestroy(&user.dm));
  PetscCall(DMDestroy(&user.dmE));

  /* Finalize */
  PetscCall(PetscFinalize());
  return 0;
}


PetscErrorCode FormInitialGuess(AppCtx *user,Vec X){
  PetscInt i,j,xs,ys,xm,ym,xe,ye;
  PetscInt gxs,gys,gxm,gym;
  PetscInt k[3],DoF=3;
  //PetscInt nx = user->mx, ny = user->my;
  PetscReal hx = user->hx, hy = user->hy, bound = user->region, l = user->lambda;
  PetscReal corX,corY,val[3]; /* coordinates */
  PetscViewer iniout;
  //Vec localX = user->localX;

  /* get vector info from dm */
  //PetscCall(DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, localX));
  //PetscCall(DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, localX));

  PetscFunctionBegin;
  /* get corner index info */
  PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAGetGhostCorners(user->dm, &gxs, &gys, NULL, &gxm, &gym, NULL));

  /* get array from vec */
  //PetscCall(DMDAVecGetArrayDOF(user->dm, xlocal, &x));
  
  /* calculate initial field values*/
  xe = xs + xm;
  ye = ys + ym;
  
  for(j = ys; j < ye; j++){
    corY = -1 * bound + j * hy;
    for(i = xs; i < xe; i++){
      corX = -1 * bound + i * hx; //except bound temporarily
      k[0] = xm * j * DoF + i * DoF; //jのDOF考慮 <- OK!
      k[1] = k[0] + 1;
      k[2] = k[0] + 2;
      val[0] = FuncComm(l,corX,corY) * (l * l * (corX * corX + corY * corY) - 1.0); 
      val[1] = FuncComm(l,corX,corY) * 2.0 * l * corX;
      val[2] = FuncComm(l,corX,corY) * -2.0 * l * corY;
      PetscCall(VecSetValuesLocal(X, 3, k, val, INSERT_VALUES));
      /*
      x[j][i][0] = FuncComm(l,corX,corY) * (l * l * (corX * corX + corY * corY) - 1.0); 
      x[j][i][1] = FuncComm(l,corX,corY) * 2.0 * l * corX;
      x[j][i][2] = FuncComm(l,corX,corY) * -2.0 * l * corY;
      */
      //PetscCall(PetscPrintf(MPI_COMM_WORLD,"val1 : %3.3e\n", (double)FuncComm(l,corX,corY)));
    }
  } 
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"count : %d\n", count));

  /* restore array */
  //PetscCall(DMDAVecRestoreArrayDOF(user->dm, xlocal, &x));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  /* setting viewer */
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "initialvecv2.dat", &iniout));
  PetscCall(PetscViewerPushFormat(iniout, PETSC_VIEWER_ASCII_DENSE));
  PetscCall(VecView(X, iniout));
  PetscCall(PetscViewerPopFormat(iniout));
  PetscCall(PetscViewerDestroy(&iniout));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal FuncComm(PetscReal Lambda, PetscReal x, PetscReal y){
  return (1.0 / (Lambda * Lambda * (x * x + y * y) + 1.0)); 
}

PetscReal PotentialTerm(AppCtx *user ,PetscReal phi_3){
  //return  0.5 * user->param_c0 * (1 - phi_3) * (1 - phi_3);
  return 0.5 * user->param_c0 * phi_3 * phi_3 * 0.0;
}

PetscReal DerivPot(PetscReal phi3,AppCtx* user){
  //return user->param_c0 * (1 - phi3) * (-1);
  return user->param_c0 * phi3 * 0.0;
}

PetscReal Potp1(PetscReal phi1, AppCtx* user){
  return user->param_c0 * (1 - phi1) * (1 - phi1);
}

PetscReal DerivPotp1(PetscReal phi1,AppCtx* user){
  return user->param_c0 * (1 - phi1) * (-1); // initial boundary忘れずに
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

