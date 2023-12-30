/*
iLQR applied to a planar bimanual robot for a tracking problem involving
the center of mass (CoM) and the end-effector (batch formulation)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Adi Niederberger <aniederberger@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
*/

#include <iostream>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <GL/glut.h>
#include <math.h>

using namespace Eigen;


#define DoF 5

// Parameters
// ===============================
struct Param {
  double dt, r;
  VectorXd Mu, MuCoM, l;
  unsigned int nbIter, nbPoints, nbData, nbVarX, nbVarU, nbVarF;
  MatrixXd Q, Qc, R;
  Param() {
    r = 1e-5;
    dt = 1e0;
    nbIter = 100;
    nbPoints = 1;
    nbData = 30;
    nbVarX = DoF;
    nbVarU = DoF;
    nbVarF = 4;
    Mu = VectorXd::Zero(nbVarF);
    Mu.head(nbVarF) << -1.0, -1.5, 4.0, 2.0;
    MuCoM = VectorXd::Zero(2);
    MuCoM.head(2) << 0., 1.4;
    l  = VectorXd::Zero(DoF); 
    l.array() = 2.0; 
    Qc = MatrixXd::Identity(nbData*2, nbData*2);
    Q =  MatrixXd::Identity(nbVarF, nbVarF);
    Q.bottomRows(2) *=0;
    Q = KroneckerProduct(MatrixXd::Identity(nbPoints, nbPoints), Q);
    R = MatrixXd::Identity(nbVarU * (nbData-1), nbVarU * (nbData-1)) * r;
  }
};

// Helper function for extracting elements by indexing
// ====================================================



// Kinematics functions
// ===============================
MatrixXd fkin(const MatrixXd &x)
{
  Param param;
  MatrixXd G = MatrixXd::Ones(3,3).triangularView<UnitLower>();
  MatrixXd f = MatrixXd::Zero(param.nbVarF, x.rows() );
  Array3i ind_vec1(0,1,2);
  Array3i ind_vec2(0,3,4);
  MatrixXd xtemp2(x.rows(),3 );
  xtemp2 << x.col(0), x.col(3) ,x.col(4);
  f.row(0) = ind_vec1.unaryExpr(param.l).matrix().transpose() * (G * (x.leftCols(3).transpose())).array().cos().matrix();
  f.row(1) = ind_vec1.unaryExpr(param.l).matrix().transpose() * (G * (x.leftCols(3).transpose())).array().sin().matrix();
  f.row(2) = ind_vec2.unaryExpr(param.l).matrix().transpose() * (G * (xtemp2.transpose())).array().cos().matrix();
  f.row(3) = ind_vec2.unaryExpr(param.l).matrix().transpose() * (G * (xtemp2.transpose())).array().sin().matrix();

  return f;
}

MatrixXd fkin0(const MatrixXd &x)
{
  Param param;
  MatrixXd G = MatrixXd::Ones(3,3).triangularView<UnitLower>();
  MatrixXd fl = MatrixXd::Zero(2, static_cast<int>((param.nbVarX+1)/2) );
  MatrixXd fr = MatrixXd::Zero(2, static_cast<int>((param.nbVarX+1)/2) );
  MatrixXd f(2,param.nbVarX+2);
  MatrixXd xl = x.leftCols(3);
  MatrixXd xr(x.rows(), 3);
  xr << x.col(0), x.col(3) ,x.col(4);
  Array3i ind_vec(0,3,4);
  fl.row(0) = (G * param.l.segment(0,3).asDiagonal() * cos((G * xl.transpose()).array()).matrix()).transpose();
  fl.row(1) = (G * param.l.segment(0,3).asDiagonal() * sin((G * xl.transpose()).array()).matrix()).transpose();
  fr.row(0) = (G * ind_vec.unaryExpr(param.l).matrix().asDiagonal() * cos((G * xr.transpose()).array()).matrix()).transpose();
  fr.row(1) = (G * ind_vec.unaryExpr(param.l).matrix().asDiagonal() * sin((G * xr.transpose()).array()).matrix()).transpose();
  MatrixXd fm = MatrixXd::Zero(2,1);
  f << fl.rowwise().reverse(), fm, fr;
  return f;
}

MatrixXd Jkin(const MatrixXd &x)
{
  Param param;
  MatrixXd G = MatrixXd::Ones(3,3).triangularView<UnitLower>();
  MatrixXd J = MatrixXd::Zero(param.nbVarF, param.nbVarX);
  MatrixXd Ju = MatrixXd::Zero(2,3);
  MatrixXd Jl = Ju;
  MatrixXd xu = x.leftCols(3);
  MatrixXd xl(x.rows(), 3);
  xl << x.col(0), x.col(3) ,x.col(4);
  Array3i ind_vec(0,3,4);
  MatrixXd t1 = - (G * xu.transpose()).array().sin().matrix();
  MatrixXd t2 = (param.l.segment(0,3).matrix().asDiagonal() * G);
  MatrixXd t3 = t1.transpose() * t2;
  Ju.row(0) = (-G * xu.transpose()).array().sin().matrix().transpose() * (param.l.segment(0,3).matrix().asDiagonal() * G);
  Ju.row(1) = (G * xu.transpose()).array().cos().matrix().transpose() * (param.l.segment(0,3).matrix().asDiagonal() * G);
  Jl.row(0) = (-G * xl.transpose()).array().sin().matrix().transpose() * (ind_vec.unaryExpr(param.l).matrix().asDiagonal() * G);
  Jl.row(1) = (G * xl.transpose()).array().cos().matrix().transpose() * (ind_vec.unaryExpr(param.l).matrix().asDiagonal() * G);
  J.block(0, 0, Ju.rows(), Ju.cols()) = Ju;
  J.block(2, 0, J.rows()-2, 1) = Jl.leftCols(1);
  J.block(2, 3, J.rows()-2, 2) = Jl.rightCols(2);
  return J;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> f_reach(const MatrixXd  &x)
{
  Param param;
  MatrixXd f = fkin(x).transpose().rowwise() - param.Mu.transpose();
  f = Map<Eigen::MatrixXd>(f.data(), f.size(), 1);
  MatrixXd J = MatrixXd::Zero(param.nbVarF* x.rows(), param.nbVarX * x.rows());
  for (unsigned t = 0; t < x.rows(); ++t)
  {
    J.block(t*param.nbVarF, t*param.nbVarX, param.nbVarF, param.nbVarX ) = Jkin(x.row(t));
  }
  return std::make_pair(f, J);
}

MatrixXd fkin_CoM(const MatrixXd  &x)
{
  Param param;
  MatrixXd G = MatrixXd::Ones(3,3).triangularView<UnitLower>();
  MatrixXd f = MatrixXd::Zero(2, x.rows() );
  MatrixXd xtemp2(x.rows(), 3);
  xtemp2 << x.col(0), x.col(3) ,x.col(4);
  Array3i ind_vec(0,3,4);
  MatrixXd deb2 =  (ind_vec.unaryExpr(param.l).matrix().transpose() * G) * (G * xtemp2.transpose() ).array().cos().matrix();
  f.row(0) = (param.l.segment(0,3).transpose() * G ) * (G * x.leftCols(3).transpose()).array().cos().matrix(); 
  f.row(0) += (ind_vec.unaryExpr(param.l).matrix().transpose() * G) * (G * xtemp2.transpose() ).array().cos().matrix() ;  
  f.row(1) = (param.l.segment(0,3).transpose() * G ) * (G * x.leftCols(3).transpose()).array().sin().matrix(); 
  f.row(1) += (ind_vec.unaryExpr(param.l).matrix().transpose() * G) * (G * xtemp2.transpose() ).array().sin().matrix() ;  
  f /= (param.nbVarX+1);
  return f;
}

MatrixXd Jkin_CoM(const MatrixXd  &x)
{
  Param param;
  MatrixXd G = MatrixXd::Ones(3,3).triangularView<UnitLower>();
  MatrixXd J(2, param.nbVarX);
  MatrixXd Jl = MatrixXd::Zero(2,3);
  MatrixXd Jr = Jl;
  MatrixXd xl = x.leftCols(3);
  MatrixXd xr(x.rows(), 3);
  xr << x.col(0), x.col(3) ,x.col(4);
  Array3i ind_vec(0,3,4);
  Jl.row(0) = (-G * xl.transpose()).array().sin().matrix().transpose() * G * (param.l.segment(0,3).matrix().transpose() * G).asDiagonal();
  Jl.row(1) = (G * xl.transpose()).array().cos().matrix().transpose() * G * (param.l.segment(0,3).matrix().transpose() * G).asDiagonal();
  Jr.row(0) = (-G * xr.transpose()).array().sin().matrix().transpose() * G * (ind_vec.unaryExpr(param.l).matrix().transpose() * G ).asDiagonal();
  Jr.row(1) = (G * xr.transpose()).array().cos().matrix().transpose() * G * (ind_vec.unaryExpr(param.l).matrix().transpose() * G).asDiagonal();
  Jl /= (param.nbVarX+1);
  Jr /= (param.nbVarX+1);
  
  J << Jl.leftCols(1) + Jr.leftCols(1), Jl.rightCols(Jl.cols()-1), Jr.rightCols(Jr.cols()-1);
  return J;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>f_reach_CoM(const MatrixXd  &x)
{
  Param param;
  MatrixXd f = fkin_CoM(x).colwise() - param.MuCoM;
  f = Map<Eigen::MatrixXd>(f.data(), f.size(), 1);
  MatrixXd J = MatrixXd::Zero(2* x.rows(), param.nbVarX * x.rows());
  for (unsigned t = 0; t < x.rows(); ++t)
  {
    J.block(t*2, t*param.nbVarX, 2, param.nbVarX ) = Jkin_CoM(x.row(t));
  }
  return std::make_pair(f, J);
}

MatrixXd get_rows(const MatrixXd &mat, const MatrixXi &rowix )
{  
  MatrixXd x_out = MatrixXd::Zero(rowix.size(), mat.cols() );
  for(unsigned int i = 0; i < rowix.size(); i++)
  {
    x_out.row(i) = mat.row(rowix(i));
  }
  return x_out;
}

MatrixXd to_vec(const MatrixXd &mat, bool rowbyrow = false)
{  
  if(rowbyrow)
  {
    mat.transpose();
  }
  VectorXd vec = Eigen::Map<const Eigen::VectorXd>(mat.data(), mat.cols()*mat.rows());

  return vec;
}

auto to_matrix = [](const VectorXd & vec) {return Eigen::Map<const MatrixXd>(vec.data(), vec.size(), 1);};

// Optimal Control
// ===============================
MatrixXd iLQR()
{
  Param param;
  MatrixXd x;
  MatrixXd Su0(param.nbData * param.nbVarX, (param.nbData - 1) * param.nbVarX  );
  MatrixXd SuA  = MatrixXd::Zero(param.nbVarX,(param.nbData - 1) * param.nbVarX );
  MatrixXd SuB = (kroneckerProduct(MatrixXd::Ones((param.nbData - 1) , (param.nbData - 1) ),  //
                                  param.dt * MatrixXd::Identity(param.nbVarX, param.nbVarX)) //
                                  ).triangularView<UnitLower>();
  // stack them vertically
  Su0 << SuA, SuB; 
  MatrixXd Sx0 = kroneckerProduct(MatrixXd::Ones(param.nbData , 1 ),  //
                                  MatrixXd::Identity(param.nbVarX, param.nbVarX));
  VectorXd u = VectorXd::Zero( param.nbVarU * (param.nbData - 1));
  VectorXd x0 = VectorXd::Zero(param.nbVarX);
  x0.head(param.nbVarX) << M_PI/2, M_PI/2, M_PI/3,-M_PI/2, -M_PI/3;

  // Set up index of viapoints for system matrices and state matrix
  VectorXi tl = VectorXd::LinSpaced(param.nbPoints+1, 0, param.nbData -1).unaryExpr([](auto x){return  static_cast<int>(std::round(x));} ).segment(1,param.nbPoints);
  MatrixXi idx = MatrixXi::Zero(tl.size(), param.nbVarU);
  std::vector<int> ivec(param.nbVarU);
  std::for_each(ivec.begin(), ivec.end(), [i=0] (int& x) mutable {x = i++;});
  Eigen::VectorXi seq1 = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(ivec.data(), ivec.size());
  idx = (idx.colwise() + tl * param.nbVarX).rowwise() + seq1.transpose();
  /* idx --> n-th row represent the n-th number of points, columns the joints (VarX) */

  MatrixXd Su = MatrixXd::Zero(idx.size(), (param.nbData - 1) * param.nbVarX  );
  idx = Eigen::Map<Eigen::VectorXi>(idx.data(), idx.size());
  
  for(unsigned int i = 0; i < idx.size(); i++)
  {
    Su.row(i) = Su0.row(idx(i));
  }
  // start iLQR iteration
  for(unsigned int k = 0; k < param.nbIter; k++) 
  {
    x = Su0 * u + Sx0 * x0;
    x = MatrixXd(Eigen::Map<Eigen::MatrixXd>(x.data(), param.nbVarX,param.nbData )).matrix().transpose();

    MatrixXd x_f1 = MatrixXd::Zero(tl.size(),param.nbVarX );
    for(unsigned int i = 0; i < tl.size(); i++)
    {
      x_f1.row(i) = x.row(tl(i));
    }

    auto [f, J] = f_reach(x_f1);
    auto [fc, Jc] = f_reach_CoM(x);

    MatrixXd du = (Su.transpose() * J.transpose() * param.Q * J * Su  + //
                  Su0.transpose() * Jc.transpose() * param.Qc * Jc * Su0 + param.R ).inverse() * //
                  (- Su.transpose() * J.transpose() * param.Q * f - //
                    Su0.transpose() * Jc.transpose() * param.Qc * fc - u * param.r);

    double alpha = 1.0;
    auto cost0= (f.transpose() * param.Q * f).value() + (fc.transpose() * param.Qc * fc).value() + u.squaredNorm() * param.r; //   array().matrix().squaredNorm() * param.r;
  
    while (true)
    {
      MatrixXd utmp = u + du * alpha;
      MatrixXd xtmp = Su0 * utmp + Sx0 * x0;
      xtmp = MatrixXd(Eigen::Map<Eigen::MatrixXd>(xtmp.data(), param.nbVarX,param.nbData )).matrix().transpose();
      MatrixXd f_val = fkin_CoM(xtmp);
      MatrixXd ftmp = std::get<0>(f_reach(get_rows(xtmp, tl)));
      MatrixXd fctmp = std::get<0>(f_reach_CoM(xtmp));

      double cost = (ftmp.transpose() * param.Q * ftmp).value() + (fctmp.transpose() * param.Qc * fctmp).value() + u.squaredNorm() * param.r; 
      if (cost < cost0  || alpha < 1e-3)
      {
        std::cout << "\t Iteration: " << k+1 << " Cost: " << cost << " alpha: " << alpha << std::endl;
        break;
      }
      alpha *= .5;
    }
    u = u + du * alpha;
    // stop optimizing if the gradient update becomes too small
    if ((du * alpha).norm() < 1e-2)
    {
      break;
    }
  }
  return x;
}

// Plot functions
// ===============================
void plot_robot(const MatrixXd& x, const double c=0.0)
{
  glColor3d(c, c, c);
  glLineWidth(8.0);
  glBegin(GL_LINE_STRIP);
  glVertex2d(x(0, 0), x(1, 0));
  for (int i = 1; i < x.cols(); i++)
  {
      glVertex2d(x(0, i), x(1, i));
  }
  glEnd();
}

void plot_ee_traj(const MatrixXd& x, const GLfloat c=0.0, const int ix = 0)
{
  // Plot end-effector trajectory
  glColor4f(c,c,c, .5f);
  //glColor3d(c, c, c);
  glLineWidth(4.0);
	glBegin(GL_LINE_STRIP);
  glVertex2d(x(ix, 0), x(ix+1, 0));
	for (int i = 1; i < x.cols(); i++)
  {
      glVertex2d(x(ix, i), x(ix+1, i));
  }
  glEnd();
}

void drawHollowCircle(double x, double y, double radius, std::tuple<double, double, double, double> color){
	int i;
	int lineAmount = 100; //# of triangles used to draw circle
  // set up alpha blending for transparenty
  glEnable (GL_BLEND); 
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	//GLfloat radius = 0.8f; //radius
	double twicePi = 2.0 * M_PI;
  glColor4f(static_cast<GLfloat>(std::get<0>(color)), static_cast<GLfloat>(std::get<1>(color)), //
            static_cast<GLfloat>(std::get<2>(color)), static_cast<GLfloat>(std::get<3>(color)));
	glLineWidth(3.0f);
	glBegin(GL_LINE_LOOP);
		for(i = 0; i <= lineAmount;i++) { 
			glVertex2f(
			    (GLfloat)(x + (radius * cos(i *  twicePi / lineAmount))), 
			    (GLfloat)(y + (radius* sin(i * twicePi / lineAmount)))
			);
		}
	glEnd();
}


void render(){
  Param param;

  MatrixXd x = iLQR();
  MatrixXd ftmp0 = fkin0(x.row(0));
  MatrixXd ftmpT = fkin0(x.row(x.rows()-1));
  MatrixXd ftmp_ee = fkin(x);
  MatrixXd fc = fkin_CoM(x); // Forward kinematics for center of mass

	glLoadIdentity();
	double d = (double)param.l.size() * .9;
	glOrtho(-d - d/4, d + d/4, -d/2 , d/2 +d/4, -1.0, 1.0);
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

  // Plot start and end configuration
  plot_robot(ftmp0, 0.8);
  plot_robot(ftmpT, 0.4);
	
	// Plot end-effector trajectory
  plot_ee_traj(ftmp_ee, 0.,0);
  plot_ee_traj(ftmp_ee,0.,2);

  // plot CoM
  drawHollowCircle(fc(0,0), fc(1,0), .09, {0,0,0,.6} );
  drawHollowCircle(fc(0,fc.cols()-1), fc(1,fc.cols()-1), .09, {0,0,0,.0} );
  drawHollowCircle(param.MuCoM(0), param.MuCoM(1), .09, {1,0,0,1} );

  // plot ee target
  glColor3d(1.0, 0, 0);
  glTranslatef((GLfloat)param.Mu(0), (GLfloat)param.Mu(1), 0.0f);
  glutSolidSphere(.08, 20.0, 20.0);

	//Render
	glutSwapBuffers();
}

Param param;

int main(int argc, char** argv) 
{
  glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(20,20);
	glutInitWindowSize(1200,600);
	glutCreateWindow("iLQR_bimanual");
	glutDisplayFunc(render);
	glutMainLoop();
  return 0;
}
	
