#include <iostream>
#include <cstdint>
#include <random>
#include "pub_simpleLasso.h"

int main(int argc, char **argv)
{
    const int32_t n_iter            = 1;
    const int32_t n_samples         = 100;
    const int32_t n_features        = 1000;
    const int32_t n_nonzero_coefs   = 20;
    const double alpha = 1.0;

#if 0
    Eigen::Matrix<double, n_samples, n_features> X;
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n_features);
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n_features);
    Eigen::MatrixXd wp = Eigen::MatrixXd::Zero(n_samples, n_features);

    auto num = 0;
    for(int32_t i=0; i<n_samples; i++) {
        for(int32_t j=0; j<n_features; j++) {
            X(i, j) = num;
            num++;
        }
    }
    X = X / 100;
    y = X * w;
#endif
    std::random_device seed;
    std::mt19937 mt( seed() );
    std::uniform_int_distribution<int32_t> random_dist(0, n_features-1);
    std::default_random_engine engine( seed() );

    std::vector<int32_t> idx_list;

    // 乱数を設定するを生成
    for(int32_t i=0; i<n_nonzero_coefs; i++) {
       idx_list.push_back( random_dist( mt ) );
    }

    // 標準正規分布で重みを生成
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    Eigen::VectorXd w = Eigen::VectorXd(n_features);
    for(auto e : idx_list) {
        w[e] = normal_dist(engine);
    }
    // PRINT_MAT(w);

    // 入力データの生成
    std::vector<double> vRand(n_samples * n_features);
    for(auto &e : vRand) {
        e = normal_dist(engine);
    }
    Eigen::MatrixXd X = Eigen::Map<Eigen::MatrixXd>(&vRand[0], n_samples, n_features);
    // printVector(vRand);
    // PRINT_MAT(X);

    // 観測情報の生成
    vRand.clear();
    vRand.resize(n_samples);
    for(auto &e : vRand) {
        e = normal_dist(engine);
    }
    Eigen::VectorXd z = Eigen::Map<Eigen::VectorXd>(&vRand[0], vRand.size());
    // PRINT_MAT(z);

    Eigen::VectorXd y = (X * w);
    y = y + z;
    // PRINT_MAT(y);

    // 座標降下法で正則化
    Eigen::MatrixXd w_pred = coordinateDescent(X, y, alpha, n_iter);

    std::cout << "Number of nonzero coefficiennts (true) : "
        << w.sum() << std::endl;

    std::cout << "Number of nonzero coefficiennts (predicted) : "
        << w_pred.sum() << std::endl;

    std::cout << "Euclidean distance between coefficients : "
        << (w - w_pred).norm() << std::endl;

    std::cout << "Euclidean distance between estimated output : "
        << (y -  (X * w_pred) ).norm() << std::endl;
}
