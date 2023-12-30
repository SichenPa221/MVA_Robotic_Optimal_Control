/*
iLQR applied to a 2D point-mass system reaching a target while avoiding 
obstacles represented as Gaussian process implicit surfaces (GPIS)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Léane Donzé <leane.donze@epfl.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
*/

#include <Eigen/Core>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

#include <iostream>
#include <vector>

namespace
{
    // define the parameters that influence the behaviour of the algorithm
    struct Parameters
    {
        int num_iterations = 300;  // maximum umber of iterations for iLQR
        double dt = 1e-2;        // time step size
        int num_timesteps = 101;  // number of datapoints

        // definition of the viapoints, size <= num_timesteps
        std::vector<Eigen::Vector3d> viapoints = {
            { 0.9, 0.9, M_PI / 6 }  //
        };

        // Control space dimension
        int nbVarU = 2; //dx1, dx2

        // State space diimension
        int nbVarX = 2; //x1, x2

        // initial configuration of the robot
        std::vector<double> initial_state = { 0.3, 0.05};

        // GPIS representation of obstacles
        std::vector<double> p = {1.4, 1e-5}; //Thin-plate covariance function parameters
        std::vector<std::vector<double>> x = { {0.2, 0.4, 0.6, -0.4, 0.6, 0.9},
                                          {0.5, 0.5, 0.5, 0.8, 0.1, 0.6} };
        std::vector<double> y = {-1, 0, 1, -1, -1, -1};

        // Disc as gemoetric prior
        double rc = 4e-1;                       // Radius of the disc
        std::vector<double> xc = {0.55, 0.55};  // Location of the disc


        double tracking_weight = 1e2;  // tracking weight term
        double obstacle_weight = 1e0;  // obstacle weight term
        double control_weight = 1e-3;  // control weight term


    };

    struct Model
    {
      public:
        /**
         * initialize the model with the given parameter
         * this calculates the matrices Su and Sx
         */
        Model(const Parameters &parameters);

        /**
         * implementation of the iLQR algorithm
         * this function calls the other functions as needed
         */
        Eigen::MatrixXd ilqr() const;

        /**
         * perform a trajectory rollout for the given initial joint angles and control commands
         * return a joint angle trajectory
         */
        Eigen::MatrixXd rollout(const Eigen::VectorXd &initial_state, const Eigen::VectorXd &control_commands) const;

        /**
         * reaching function, called at each iLQR iteration
         * calculates the error to each viapoint as well as the jacobians
         */
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> reach(const Eigen::MatrixXd &x) const;

        /**
         * Residuals f and Jacobians J for obstacle avoidance with GPIS representation
         * (for all time steps)
         */

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int> > avoid(const Eigen::MatrixXd &x) const;

        /**
         * convenience function to extract the viapoints from given joint angle trajectory
         * also reshapes the viapoints to the expected matrix dimensions
         */
        Eigen::MatrixXd viapoints(const Eigen::MatrixXd &joint_angle_trajectory) const;

        /**
         * return the viapoint_timesteps_
         * only used for plotting
         */
        const Eigen::VectorXi &viatimes() const;

        /**
         * implementation of the Matlab function "pdist2" 
         * which seems to have no equivalent in Eigen library
         */
        Eigen::MatrixXd pdist2(const Eigen::MatrixXd X, const Eigen::MatrixXd Y) const;

        /**
         * Error function
         */
        Eigen::MatrixXd substr(const Eigen::VectorXd x1, const Eigen::VectorXd x2) const;

        /**
         * Covariance function in GPIS
         */
        std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd> > covFct(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2, const Eigen::VectorXd p, bool flag_noiseObs) const;

        /**
         * Residuals f and Jacobians J for obstacle avoidance
         * with GPIS representation (for a given time step) 
         */

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd > GPIS(const Eigen::MatrixXd &x) const;


      private:
        /// parameters defined at runtime
        Eigen::MatrixXd viapoints_;                     // internal viapoint representation
        Eigen::VectorXi viapoint_timesteps_;            // discrete timesteps of the viapoints, uniformly spread

        double tracking_weight_;
        double obstacle_weight_;
        double r_;  // control weight parameter

        int nbVarU_;          // Control space dimension (dx1, dx2)
        int nbVarX_;          // State space dimension (q1,q2,q3)
        int num_timesteps_;   // copy the number of timesteps for internal use from the Parameters
        int num_iterations_;  // maximum number of iterations for the optimization

        Eigen::VectorXd initial_state_;         // internal state 
        Eigen::MatrixXd control_weight_;        // R matrix, penalizes the control commands
        Eigen::MatrixXd precision_matrix_;      // Q matrix, penalizes the state error
        Eigen::MatrixXd mat_Su0_;               // matrix for propagating the control commands for a rollout
        Eigen::MatrixXd mat_Sx0_;               // matrix for propagating the initial joint angles for a rollout
        Eigen::MatrixXd mat_Su_;                // matrix for propagating the control commands for a rollout at the viapoints
     

        Eigen::VectorXd y_;                     // GPIS representation
        Eigen::VectorXd Mu2_;                   // GPIS representation
        Eigen::MatrixXd K_;                     // GPIS representation 
        Eigen::MatrixXd x_;                     // GPIS representation
        Eigen::VectorXd p_ ;                    // GPIS representation

    };

    ////////////////////////////////////////////////////////////////////////////////
    // implementation of the iLQR algorithm

    Eigen::MatrixXd Model::ilqr() const
    {
        // initial commands, currently all zero
        // can be modified if a better guess is available
        Eigen::VectorXd control_commands = Eigen::VectorXd::Zero(nbVarU_* (num_timesteps_ - 1), 1);

        int iter = 1;
        for (; iter <= num_iterations_; ++iter)  // run the optimization for the maximum number of iterations
        {
            std::cout << "iteration " << iter << ":" << std::endl;

            // trajectory rollout, i.e. compute the trajectory for the given control commands starting from the initial state
            // Sx * x0 + Su * u
            Eigen::MatrixXd x = rollout(initial_state_, control_commands);

            // Residuals f and Jacobians J (tracking objective)
            auto [state_error, jacobian_matrix] = reach(viapoints(x));

            // Residuals f and Jacobians J (avoidance objective)
            auto [obstacle_error, obstacle_jacobian, idx, idt] = avoid(x);

            Eigen::MatrixXd mat_Su2 = Eigen::MatrixXd::Zero(static_cast<int>(idx.size()), mat_Su0_.cols());
            for (unsigned i = 0; i < idx.size(); ++i)
            {
                mat_Su2.row(i) = mat_Su0_.row(idx[i]);
            }

            // Gauss-Newton update
            // (Su'*J'*Q*J*Su+Su2'*J2'*J2*Su2*param.q2 + R) \ (-Su'*J'*Q*f(:) - Su2'*J2'*f2(:) * param.q2 - u * param.r)
            Eigen::MatrixXd control_gradient = (mat_Su_.transpose() * jacobian_matrix.transpose() * tracking_weight_ * jacobian_matrix * mat_Su_        //
                                                + mat_Su2.transpose() * obstacle_jacobian.transpose() * obstacle_jacobian * mat_Su2 * obstacle_weight_  //
                                                + control_weight_)                                                                                      //
                                                 .inverse()                                                                                             //
                                               * (-mat_Su_.transpose() * jacobian_matrix.transpose() * tracking_weight_ * state_error                   //
                                                  - mat_Su2.transpose() * obstacle_jacobian.transpose() * obstacle_error * obstacle_weight_             //
                                                  - r_ * control_commands);

            // calculate the cost of the current state
            double current_cost = state_error.squaredNorm() * tracking_weight_ + obstacle_error.squaredNorm() * obstacle_weight_ + r_ * control_commands.squaredNorm();

            std::cout << "\t cost = " << current_cost << std::endl;

            // initial step size for the line search
            double step_size = 1.0;
            // line search, i.e. find the best step size for updating the control commands with the gradient
            while (true)
            {
                //  calculate the new control commands for the current step size
                Eigen::MatrixXd tmp_control_commands = control_commands + control_gradient * step_size;

                // calculate a trajectory rollout for the current step size
                Eigen::MatrixXd tmp_x = rollout(initial_state_, tmp_control_commands);

                // try reaching the viapoints with the current state
                // we only need the state error here and can disregard the Jacobian, because we are only interested in the cost of the trajectory
                Eigen::MatrixXd tmp_state_error = reach(viapoints(tmp_x)).first;

                Eigen::MatrixXd tmp_obstacle_error = std::get<0>(avoid(tmp_x));

                // resulting cost when updating the control commands with the current step size
                double cost =
                  tmp_state_error.squaredNorm() * tracking_weight_ + tmp_obstacle_error.squaredNorm() * obstacle_weight_ + r_ * tmp_control_commands.squaredNorm();

                // end the line search if the current steps size reduces the cost or becomes too small
                if (cost < current_cost || step_size < 1e-3)
                {
                    control_commands = tmp_control_commands;

                    break;
                }

                // reduce the step size for the next iteration
                step_size *= 0.5;
            }

            std::cout << "\t step_size = " << step_size << std::endl;

            // stop optimizing if the gradient update becomes too small
            if ((control_gradient * step_size).norm() < 1e-2)
            {
                break;
            }
        }

        std::cout << "iLQR converged in " << iter << " iterations" << std::endl;

        return rollout(initial_state_, control_commands);
    }
    ////////////////////////////////////////////////////////////////////////////////
    // implementation of all functions used in the iLQR algorithm

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Model::reach(const Eigen::MatrixXd &x) const
    {
        Eigen::MatrixXd state_error = x - viapoints_.topRows(2); 

        Eigen::MatrixXd jacobian_matrix = Eigen::MatrixXd::Identity(nbVarX_ * x.cols(), nbVarX_ * x.cols());

        state_error = Eigen::Map<Eigen::MatrixXd>(state_error.data(), state_error.size(), 1);

        return std::make_pair(state_error, jacobian_matrix);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int> > Model::avoid(const Eigen::MatrixXd &x) const
    {
        auto [ftmp, Jtmp] = GPIS(x);

        std::vector<double> fs;
        std::vector<Eigen::MatrixXd> js;
        std::vector<int> idx;
        std::vector<int> idt;

        int size = 0;

        for (unsigned t = 0; t < x.cols(); ++ t)
        {
            if (ftmp(0,t) > 0)
            {
                fs.push_back(ftmp(0,t));
                js.push_back(Jtmp.col(t).transpose());

                size += 1;

                for (unsigned j = 0; j < x.rows(); ++j)
                {
                    idx.push_back(static_cast<int>(t * x.rows() + j));
                }
                idt.push_back(t);
            }
        }


        Eigen::MatrixXd f = Eigen::Map<Eigen::MatrixXd>(fs.data(), static_cast<int>(fs.size()), 1);
        Eigen::MatrixXd j = Eigen::MatrixXd::Zero(size, 2*size);

        int r = 0, c = 0;
        for (const Eigen::MatrixXd &ji : js)
        {
            j.block(r, c, ji.rows(), ji.cols()) = ji;
            r += static_cast<int>(ji.rows());
            c += static_cast<int>(ji.cols());
        }

        return std::make_tuple(f,j,idx,idt);
        
    }

    Eigen::MatrixXd Model::rollout(const Eigen::VectorXd &initial_state, const Eigen::VectorXd &control_commands) const
    {
        Eigen::MatrixXd x = mat_Su0_ * control_commands + mat_Sx0_ * initial_state;
        x = Eigen::MatrixXd(Eigen::Map<Eigen::MatrixXd>(x.data(), nbVarX_, num_timesteps_));

        return x;
    }

    Eigen::MatrixXd Model::viapoints(const Eigen::MatrixXd &joint_angle_trajectory) const
    {
        Eigen::MatrixXd via_joint_angles = Eigen::MatrixXd::Zero(joint_angle_trajectory.rows(), viapoint_timesteps_.size());

        for (unsigned t = 0; t < viapoint_timesteps_.size(); ++t)
        {
            via_joint_angles.col(t) = joint_angle_trajectory.col(viapoint_timesteps_(t));
        }

        return via_joint_angles;
    }

    const Eigen::VectorXi &Model::viatimes() const
    {
        return viapoint_timesteps_;
    }


    Eigen::MatrixXd Model::pdist2(const Eigen::MatrixXd X, const Eigen::MatrixXd Y)  const
    {   
        Eigen::MatrixXd result(X.rows(), Y.rows());
        for (unsigned i = 0; i < X.rows(); ++i)
        {
            for (unsigned j = 0; j < Y.rows(); ++j){
                Eigen::VectorXd v1 = X.row(i);
                Eigen::VectorXd v2 = Y.row(j);
                result(i,j) = (v1-v2).norm();
            }
        }
        return result;
    }

    Eigen::MatrixXd Model::substr(const Eigen::VectorXd x1, const Eigen::VectorXd x2) const
    {
        return x1.replicate(1,x2.size()) - x2.transpose().replicate(x1.size(), 1);
    }

    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd> > Model::covFct(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2, const Eigen::VectorXd p, bool flag_noiseObs) const
    {
        // Thin plate covariance function (for 3D implicit shape)
        Eigen::MatrixXd K = pow(12,-1)*(2*pdist2(x1.transpose(),x2.transpose()).array().pow(3) - 3*p(0)*pdist2(x1.transpose(), x2.transpose()).array().pow(2) + pow(p(0),3)); // Kernel
        Eigen::MatrixXd dK1 = pow(12,-1)*(6*pdist2(x1.transpose(),x2.transpose()).cwiseProduct(substr(x1.row(0).transpose(), x2.row(0))) - 6*p(0)*substr(x1.row(0).transpose(), x2.row(0))); // Derivative along x1
        Eigen::MatrixXd dK2 = pow(12,-1)*(6*pdist2(x1.transpose(),x2.transpose()).cwiseProduct(substr(x1.row(1).transpose(), x2.row(1))) - 6*p(0)*substr(x1.row(1).transpose(), x2.row(1))); // Derivative along x2

        std::vector<Eigen::MatrixXd> dK;
        dK.push_back(dK1);
        dK.push_back(dK2);

        if (flag_noiseObs == true)
        {
            K = K + p(1)*Eigen::MatrixXd::Identity(x1.cols(), x2.cols()); //Consideration of noisy observation y
        }
        return std::make_pair(K, dK);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd > Model::GPIS(const Eigen::MatrixXd &x) const
    {
        auto [K, dK] = covFct(x, x_, p_, false);
        Eigen::MatrixXd f = (K*K_.inverse()*y_).transpose();    //GPR with Mu=0
        Eigen::MatrixXd J(x.rows(), x.cols());
        J.row(0) = (dK[0]*K_.inverse() * (y_-Mu2_)).transpose();
        J.row(1) = (dK[1]*K_.inverse() * (y_-Mu2_)).transpose();

        // Reshape gradients
        Eigen::MatrixXd a = f.cwiseMax(0);  //Amplitude
        J = 1e2*(a.array().tanh().replicate(2,1).cwiseProduct(J.array()).cwiseQuotient(J.array().square().colwise().sum().sqrt().replicate(2,1))); // Vector moving away from interior of shape

        return std::make_tuple(f, J);

    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    // precalculate matrices used in the iLQR algorithm

    Model::Model(const Parameters &parameters)
    {
        int num_viapoints = static_cast<int>(parameters.viapoints.size());

        nbVarU_ = parameters.nbVarU;
        nbVarX_= parameters.nbVarX;

        r_ = parameters.control_weight;
        tracking_weight_ = parameters.tracking_weight;
        obstacle_weight_ = parameters.obstacle_weight;

        num_timesteps_ = parameters.num_timesteps;
        num_iterations_ = parameters.num_iterations;

        viapoints_ = Eigen::MatrixXd::Zero(3, num_viapoints);
        for (unsigned t = 0; t < parameters.viapoints.size(); ++t)
        {
            viapoints_.col(t) = parameters.viapoints[t];
        }
        
        initial_state_ = Eigen::VectorXd::Zero(nbVarX_);
        for (unsigned i = 0; i < parameters.initial_state.size(); ++i)
        {
            initial_state_(i) = parameters.initial_state[i];
        }

        viapoint_timesteps_ = Eigen::VectorXd::LinSpaced(num_viapoints + 1, 0, num_timesteps_ - 1).bottomRows(num_viapoints).array().round().cast<int>();

        control_weight_ = r_ * Eigen::MatrixXd::Identity((num_timesteps_ - 1) * nbVarU_, (num_timesteps_ - 1) * nbVarU_);
        precision_matrix_ = tracking_weight_* Eigen::MatrixXd::Identity(nbVarX_ * num_viapoints, nbVarX_ * num_viapoints);

        Eigen::MatrixXi idx = Eigen::VectorXi::LinSpaced(nbVarX_, 0, nbVarX_ - 1).replicate(1, num_viapoints);

        for (unsigned i = 0; i < idx.rows(); ++i)
        {
            idx.row(i) += Eigen::VectorXi((viapoint_timesteps_.array()) * nbVarX_).transpose();
        }

        mat_Su0_ = Eigen::MatrixXd::Zero(nbVarX_* (num_timesteps_), nbVarX_* (num_timesteps_ - 1));
        mat_Su0_.bottomRows(nbVarX_ * (num_timesteps_ - 1)) = kroneckerProduct(Eigen::MatrixXd::Ones(num_timesteps_ - 1, num_timesteps_ - 1),  //
                                                                                  parameters.dt * Eigen::MatrixXd::Identity(nbVarX_, nbVarX_))
                                                                   .eval()
                                                                   .triangularView<Eigen::Lower>();
        mat_Sx0_ = kroneckerProduct(Eigen::MatrixXd::Ones(num_timesteps_, 1),  //
                                    Eigen::MatrixXd::Identity(nbVarX_, nbVarX_))
                     .eval();

        mat_Su_ = Eigen::MatrixXd::Zero(idx.size(), nbVarX_* (num_timesteps_ - 1));
        for (unsigned i = 0; i < idx.size(); ++i)
        {
            mat_Su_.row(i) = mat_Su0_.row(idx(i));
        }

        y_ = Eigen::VectorXd::Zero(static_cast<int>(parameters.y.size()));
        for (unsigned i = 0; i < parameters.y.size(); ++i)
        {
            y_(i) = parameters.y[i];
        }

        Eigen::MatrixXd S = pow(parameters.rc, -2) * Eigen::MatrixXd::Identity(2,2);
        x_ = Eigen::MatrixXd::Zero(parameters.x.size(), parameters.x[0].size());
        for (unsigned i = 0; i < parameters.x.size(); ++i)
        {
            for (unsigned j = 0; j < parameters.x[0].size(); ++j){
                x_(i,j) = parameters.x[i][j];
            }
        }

        p_ = Eigen::VectorXd::Zero(static_cast<int>(parameters.y.size()));
        for (unsigned i = 0; i < parameters.p.size(); ++i)
        {
            p_(i) = parameters.p[i];
        }

        Eigen::VectorXd xc = Eigen::VectorXd::Zero(static_cast<int>(parameters.xc.size()));
        for (unsigned i = 0; i < parameters.xc.size(); ++i)
        {
            xc(i) = parameters.xc[i];
        }

        Mu2_ = 0.5* parameters.rc* (Eigen::MatrixXd::Ones(x_.cols(), x_.cols()) - (x_ - xc.replicate(1,x_.cols())).transpose()*S //
                                                                                *(x_ - xc.replicate(1,x_.cols()))).diagonal();
        
        K_ = std::get<0>(covFct(x_,x_,p_,true));

    }

    ////////////////////////////////////////////////////////////////////////////////
}

int main()
{
    Parameters parameters;
    Model model(parameters);  // initialize the model with the parameters that were defined at the top

    Eigen::MatrixXd trajectory = model.ilqr();  // calculate the iLQR solution

    return 0;
}
