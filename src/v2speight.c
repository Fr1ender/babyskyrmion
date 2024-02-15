/*
Now we can use mpi in this program...
*/

#include "mpi.h"
#include "petscdm.h"
#include "petscdmtypes.h"
#include "petscerror.h"
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
/* -------------------- */

  /* TEST */
  /*
  PetscReal f;
  Vec       testG;
  PetscCall(DMCreateLocalVector(user.dm, &testG));
  PetscCall(FormFunctionGradient(tao, x, &f, testG, &user));
  PetscCall(VecDestroy(&testG));

  PetscViewer testview;
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "testtest.dat", &testview));
  PetscCall(PetscViewerPushFormat(testview, PETSC_VIEWER_ASCII_DENSE));
  PetscCall(VecView(x, testview));
  PetscCall(PetscViewerPopFormat(testview));
  PetscCall(PetscViewerDestroy(&testview));
  */
/* -------------------- */
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
  //PetscCall(EnergyDensity(x, E, (void *)&user));

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

PetscReal boundary(PetscInt i, PetscInt j, PetscInt fieldNUM,Vec X,AppCtx *user) {
  PetscReal x,y,l,bound,hx,hy;
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

  if (fieldNUM == 0) {
    //return FuncComm(l, x, y) * (l * l * (x * x + y * y) - 1.0) * 0.0;
    //return vec[i + j * nx];
    return 1.0;
  } else if (fieldNUM == 1) {
    //return FuncComm(l, x, y) * 2.0 * l * x * 0.0;
    //return vec[i + j * nx + dim];
    return 0.0;
  } else if (fieldNUM == 2) {
    //return FuncComm(l, x, y) * -2.0 * l * y * 0.0;
    //return vec[i + j * nx + dim * 2];
    return 0.0;
  } else {
    PetscPrintf(MPI_COMM_WORLD, "ERROR boundary");
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
  AppCtx *user = (AppCtx *)ptr;
  PetscReal         zero = 0.0, p5 = 0.5;
  PetscInt          xe, ye, xsm, ysm, xep, yep;         /* xe,ye : corner index  */
  PetscInt          xs, ys, xm, ym, gxm, gym, gxs, gys; /* xm,ym : local array length */
  PetscInt          mx = user->mx, my = user->my,i,j;//,dim,k,k1,k2,k3,ind;
  PetscReal         hx = user->hx, hy = user->hy, lagmul = user->param_lag;
  PetscReal         vb1, vl1, vr1, vt1,v1,vb2,vl2,vr2,vt2,v2,vb3,vl3,vr3,vt3,v3;
  PetscReal         dp1dx, dp1dy,dp2dx,dp2dy,dp3dx,dp3dy,fquad = 0.0,flag = 0.0,f12 = 0.0, area,pnorm,chargeCheck = 0.0,fskyrme = 0.0,fpot = 0.0;
  PetscReal         chargelocal = 0.0,flocal;
  Vec               localX = user->localX;
  const PetscScalar ***x;
  PetscReal ***g;
  const PetscReal PI = 3.1415926535;
  PetscFunctionBeginUser;

  /* Initialize */  
  PetscCall(VecSet(G, zero));

  /* Set Dm */ 
  PetscCall(DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, localX));

  /* Get pointer to vector data */
  PetscCall(DMDAVecGetArrayDOF(user->dm, localX, &x));
  PetscCall(DMDAVecGetArrayDOF(user->dm, G, &g));

  /* Get local mesh boundaries */
  PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAGetGhostCorners(user->dm, &gxs, &gys, NULL, &gxm, &gym, NULL));

  /* Set local loop dimensions */
  xe = xs + xm;
  ye = ys + ym;
  if (xs == 0) xsm = xs - 1;
  else xsm = xs;
  if (ys == 0) ysm = ys - 1;
  else ysm = ys;
  if (xe == mx) xep = xe + 1;
  else xep = xe;
  if (ye == my) yep = ye + 1;
  else yep = ye;

  /* Compute function and gradient */
  for(j = ysm; j < ye; j++){
    for(i = xsm; i < xe; i++){
      //k = (j - gys) * gxm + i - gxs;
      v1  = boundary(i, j, 0, localX, user);
      vr1 = boundary(i, j, 0, localX, user);
      vt1 = boundary(i, j, 0, localX, user);
      v2  = boundary(i, j, 1, localX, user);
      vr2 = boundary(i, j, 1, localX, user);
      vt2 = boundary(i, j, 1, localX, user);
      v3  = boundary(i, j, 2, localX, user);
      vr3 = boundary(i, j, 2, localX, user);
      vt3 = boundary(i, j, 2, localX, user);
      if (i >= 0 && j >= 0){
        v1 = x[j][i][0];
        v2 = x[j][i][1];
        v3 = x[j][i][2];
      }
      if(i < mx - 1 && j > -1){
        vr1 = x[j][i + 1][0];
        vr2 = x[j][i + 1][1];
        vr3 = x[j][i + 1][2];
      }
      if(i > -1 && j != my - 1){
        vt1 = x[j + 1][i][0];
        vt2 = x[j + 1][i][1];
        vt3 = x[j + 1][i][2];
      }
      /* first derivative */
      dp1dx = (vr1 - v1) / hx;
      dp2dx = (vr2 - v2) / hx;
      dp3dx = (vr3 - v3) / hx;
      dp1dy = (vt1 - v1) / hy;
      dp2dy = (vt2 - v2) / hy;
      dp3dy = (vt3 - v3) / hy;

      pnorm = v1 * v1 + v2 * v2 + v3 * v3;
      f12   = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12   -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;

      PetscReal ddeddp1dx,ddeddp1dy,ddeddp2dx,ddeddp2dy,ddeddp3dx,ddeddp3dy;
      PetscReal dedp1,dedp2,dedp3;
      ddeddp1dx = (user->param_c4 * f12 * (v3 * dp2dy - v2 * dp3dy) ) / hx;
      ddeddp1dy = (user->param_c4 * f12 * (v2 * dp3dx - v3 * dp2dx) ) / hy;
      ddeddp2dx = (user->param_c4 * f12 * (v1 * dp3dy - v3 * dp1dy) ) / hx;
      ddeddp2dy = (user->param_c4 * f12 * (v3 * dp1dx - v1 * dp3dx) ) / hy;
      ddeddp3dx = (user->param_c4 * f12 * (v2 * dp1dy - v1 * dp2dy) ) / hx;
      ddeddp3dy = (user->param_c4 * f12 * (v1 * dp2dx - v2 * dp1dx) ) / hy;
      dedp1     =  user->param_c4 * f12 * (dp2dx * dp3dy - dp3dx * dp2dy);
      dedp2     =  user->param_c4 * f12 * (dp3dx * dp1dy - dp1dx * dp3dy);
      dedp3     =  user->param_c4 * f12 * (dp1dx * dp2dy - dp2dx * dp1dy);

      if(i != -1 && j != -1){
        g[j][i][0] += user->param_c2 * ( - dp1dx / hx - dp1dy / hy - ddeddp1dx - ddeddp1dy) + (dedp1 + 4 * lagmul * (pnorm - 1) * v1 + DerivPotp1(v1, user)) / 3.0;
        g[j][i][1] += user->param_c2 * ( - dp2dx / hx - dp2dy / hy - ddeddp2dx - ddeddp2dy) + (dedp2 + 4 * lagmul * (pnorm - 1) * v2) / 3.0;
        g[j][i][2] += user->param_c2 * ( - dp3dx / hx - dp3dy / hy - ddeddp3dx - ddeddp3dy) + (dedp3 + 4 * lagmul * (pnorm - 1) * v3 + DerivPot(v3, user)) / 3.0;
      }
      if(i != mx - 1 && j != -1){// TODO change index 
        g[j][i + 1][0] += user->param_c2 * ( dp1dx / hx + ddeddp1dx ) + (dedp1 + 4 * lagmul * (pnorm - 1) * v1 + DerivPotp1(v1, user)) / 3.0;
        g[j][i + 1][1] += user->param_c2 * ( dp2dx / hx + ddeddp2dx ) + (dedp2 + 4 * lagmul * (pnorm - 1) * v2) / 3.0;
        g[j][i + 1][2] += user->param_c2 * ( dp3dx / hx + ddeddp3dx ) + (dedp3 + 4 * lagmul * (pnorm - 1) * v3 + DerivPot(v3, user)) / 3.0;
      }
      if(i != -1 && j != my - 1){
        g[j + 1][i][0] += user->param_c2 * ( dp1dy / hy + ddeddp1dy ) + (dedp1 + 4 * lagmul * (pnorm - 1) * v1 + DerivPotp1(v1, user)) / 3.0;
        g[j + 1][i][1] += user->param_c2 * ( dp2dy / hy + ddeddp2dy ) + (dedp2 + 4 * lagmul * (pnorm - 1) * v2) / 3.0;
        g[j + 1][i][2] += user->param_c2 * ( dp3dy / hy + ddeddp3dy ) + (dedp3 + 4 * lagmul * (pnorm - 1) * v3 + DerivPot(v3, user)) / 3.0;
      }
      /* function evaluation */
      fquad += dp1dx * dp1dx + dp1dy * dp1dy + dp2dx * dp2dx + dp2dy * dp2dy + dp3dx * dp3dx + dp3dy * dp3dy;
      flag  += (pnorm - 1) * (pnorm - 1);
      fskyrme += f12 * f12;
      chargelocal += f12;
      fpot  += PotentialTerm(user, v3) + Potp1(v1, user);
    }
  }
  /* upper triangular elements */
  for(j = ys; j < yep; j++){
    for(i = xs; i < xep; i++){
      //k = (j - gys) * gxm + i - gxs;
      v1  = boundary(i, j, 0, localX, user);
      vl1 = boundary(i, j, 0, localX, user);
      vb1 = boundary(i, j, 0, localX, user);
      v2  = boundary(i, j, 1, localX, user);
      vl2 = boundary(i, j, 1, localX, user);
      vb2 = boundary(i, j, 1, localX, user);
      v3  = boundary(i, j, 2, localX, user);
      vl3 = boundary(i, j, 2, localX, user);
      vb3 = boundary(i, j, 2, localX, user);
      if (i < mx && j > 0){
        vb1 = x[j - 1][i][0];
        vb2 = x[j - 1][i][1];
        vb3 = x[j - 1][i][2];
      }
      if(i > 0 && j < my){
        vl1 = x[j][i - 1][0];
        vl2 = x[j][i - 1][1];
        vl3 = x[j][i - 1][2];
      }
      if(i < mx && j < my){
        v1 = x[j][i][0];
        v2 = x[j][i][1];
        v3 = x[j][i][2];
      }
      /* first derivative */
      dp1dx = (v1 - vl1) / hx;
      dp2dx = (v2 - vl2) / hx;
      dp3dx = (v3 - vl3) / hx;
      dp1dy = (v1 - vb1) / hy;
      dp2dy = (v2 - vb2) / hy;
      dp3dy = (v3 - vb3) / hy;

      pnorm = v1 * v1 + v2 * v2 + v3 * v3;
      f12   = v1*dp2dx*dp3dy + v3*dp1dx*dp2dy + v2*dp3dx*dp1dy;
      f12   -= v2*dp1dx*dp3dy + v3*dp2dx*dp1dy + v1*dp3dx*dp2dy;

      PetscReal ddeddp1dx,ddeddp1dy,ddeddp2dx,ddeddp2dy,ddeddp3dx,ddeddp3dy;
      PetscReal dedp1,dedp2,dedp3;
      ddeddp1dx = (user->param_c4 * f12 * (v3 * dp2dy - v2 * dp3dy) ) / hx;
      ddeddp1dy = (user->param_c4 * f12 * (v2 * dp3dx - v3 * dp2dx) ) / hy;
      ddeddp2dx = (user->param_c4 * f12 * (v1 * dp3dy - v3 * dp1dy) ) / hx;
      ddeddp2dy = (user->param_c4 * f12 * (v3 * dp1dx - v1 * dp3dx) ) / hy;
      ddeddp3dx = (user->param_c4 * f12 * (v2 * dp1dy - v1 * dp2dy) ) / hx;
      ddeddp3dy = (user->param_c4 * f12 * (v1 * dp2dx - v2 * dp1dx) ) / hy;
      dedp1     =  user->param_c4 * f12 * (dp2dx * dp3dy - dp3dx * dp2dy);
      dedp2     =  user->param_c4 * f12 * (dp3dx * dp1dy - dp1dx * dp3dy);
      dedp3     =  user->param_c4 * f12 * (dp1dx * dp2dy - dp2dx * dp1dy);

      if(i != mx && j != 0){
        g[j - 1][i][0] += user->param_c2 * ( - dp1dy / hy - ddeddp1dy) + (dedp1 + 4 * lagmul * (pnorm - 1) * v1 + DerivPotp1(v1, user)) / 3.0;
        g[j - 1][i][1] += user->param_c2 * ( - dp2dy / hy - ddeddp2dy) + (dedp2 + 4 * lagmul * (pnorm - 1) * v2) / 3.0;
        g[j - 1][i][2] += user->param_c2 * ( - dp3dy / hy - ddeddp3dy) + (dedp3 + 4 * lagmul * (pnorm - 1) * v3 + DerivPot(v3, user)) / 3.0;
      }
      if(i != 0 && j != my){// TODO change index 
        g[j][i - 1][0] += user->param_c2 * ( - dp1dx / hx - ddeddp1dx ) + (dedp1 + 4 * lagmul * (pnorm - 1) * v1 + DerivPotp1(v1, user)) / 3.0;
        g[j][i - 1][1] += user->param_c2 * ( - dp2dx / hx - ddeddp2dx ) + (dedp2 + 4 * lagmul * (pnorm - 1) * v2) / 3.0;
        g[j][i - 1][2] += user->param_c2 * ( - dp3dx / hx - ddeddp3dx ) + (dedp3 + 4 * lagmul * (pnorm - 1) * v3 + DerivPot(v3, user)) / 3.0;
      }
      if(i != mx && j != my){
        g[j][i][0] += user->param_c2 * ( dp1dx / hx + dp1dy / hy + ddeddp1dx + ddeddp1dy ) + (dedp1 + 4 * lagmul * (pnorm - 1) * v1 + DerivPotp1(v1, user)) / 3.0;
        g[j][i][1] += user->param_c2 * ( dp2dx / hx + dp2dy / hy + ddeddp2dx + ddeddp2dy ) + (dedp2 + 4 * lagmul * (pnorm - 1) * v2) / 3.0;
        g[j][i][2] += user->param_c2 * ( dp3dx / hx + dp3dy / hy + ddeddp3dx + ddeddp3dy ) + (dedp3 + 4 * lagmul * (pnorm - 1) * v3 + DerivPot(v3, user)) / 3.0;
      }
      /* function evaluation */
      fquad += dp1dx * dp1dx + dp1dy * dp1dy + dp2dx * dp2dx + dp2dy * dp2dy + dp3dx * dp3dx + dp3dy * dp3dy;
      flag  += (pnorm - 1) * (pnorm - 1);
      fskyrme += f12 * f12;
      chargelocal += f12;
      fpot  += PotentialTerm(user, v3) + Potp1(v1, user);
    }
  }
  area = p5 * hx * hy;
  chargelocal = area * chargelocal / (4.0 * PI);
  flocal   = area * (fquad + fskyrme + fpot + flag);
  /* Sum function contributions from all processes */ /* TODO: Change to PetscCallMPI() */
  PetscCall((PetscErrorCode)MPI_Allreduce((void *)&flocal, (void *)f, 1, MPIU_REAL, MPIU_SUM, MPI_COMM_WORLD));
  PetscCall((PetscErrorCode)MPI_Allreduce((void *)&chargelocal, (void *)&chargeCheck, 1, MPIU_REAL, MPIU_SUM, MPI_COMM_WORLD));

  /* viewer */
  PetscReal derrickCHK;
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"----------------------------\n"));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"fpot(E0)           : %2.3e\n", (double)fpot * (double)area));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"fskyrme(E4)        : %2.3e\n", (double)fskyrme * (double)area));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"fskyrme - fpot : %2.3e\n", (double)derrickCHK));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"flag           : %2.3e\n", (double)flag * (double)area));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"fquad          : %2.3e\n", (double)fquad * (double)area));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"charge         : %2.3e\n", (double)chargelocal));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"funcvalue      : %2.3e\n", (double)*f));
  //PetscCall(PetscPrintf(MPI_COMM_WORLD,"pnorm : %2.3e\n", (double)pnorm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,"----------------------------\n"));
  /* finalize */
  PetscCall(DMDAVecRestoreArrayDOF(user->dm, localX, &x));
  PetscCall(DMDAVecRestoreArrayDOF(user->dm, G, &g));
  PetscCall(VecScale(G, area));
  PetscFunctionReturn(PETSC_SUCCESS);
}

