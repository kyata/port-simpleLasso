#include <iostream>
#include <cstdint>
#include "pub_simpleLasso.h"

int main(int argc, char **argv)
{
    const int32_t n_iter = 1;
    const int32_t n_samples  = 10;
    const int32_t n_features = 100;
    const double alpha = 1.0;

    Eigen::Matrix<double, n_samples, n_features> X;
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n_features);
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n_features);
    Eigen::MatrixXd wp = Eigen::MatrixXd::Zero(n_samples, n_features);

    for(int32_t i=0; i<n_samples; i++) {
        for(int32_t j=0; j<n_features; j++) {
            X(i, j) = (i+1)*(j+1);
        }
    }
    X = X / 100;
    y = X * w;

    wp = coordinateDescent(X, y, alpha, n_iter);
}
