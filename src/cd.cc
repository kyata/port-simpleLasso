#include <cstdint>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "pub_simpleLasso.h"

Eigen::MatrixXd addMatrix(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
    // 行列形状が不一致の場合は計算不可
    if( ( X.rows() != Y.rows() ) ||
        ( X.cols() != Y.cols() ) ) {
        throw std::runtime_error("shape error!");
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(X.rows(), X.cols());

    for( auto i=0; i<X.rows(); i++ ) {
        for( auto j=0; j<X.cols(); j++ ) {
            C(i, j) = X(i, j) + Y(i, j);
        }
    }

    return C;
}

Eigen::MatrixXd subMatrix(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
    // 行列形状が不一致の場合は計算不可
    if( ( X.rows() != Y.rows() ) ||
        ( X.cols() != Y.cols() ) ) {
        throw std::runtime_error("shape unmatch!!");
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(X.rows(), X.cols());

    for(auto i=0; i<X.rows(); i++) {
        for(auto j=0; j<X.cols(); j++) {
            C(i, j) = X(i, j) - Y(i, j);
        }
    }

    return C;
}

Eigen::MatrixXd dotMatrix(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
    // 入力行列の行数と列数を取得
    auto rowX = X.rows();
    auto rowY = Y.rows();
    auto colX = X.cols();
    auto colY = Y.cols();

    // Xの列数とYの行数が合わない場合は計算不可
    if( colX != rowY ) {
        throw std::runtime_error("shape unmatch!!");
    }

    // 行列積の結果は左の行数と右の列数の要素が定義される
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(rowX, colY);
    for(auto i=0; i<rowX; i++) {

        // X[i]行の成分を取り出す
        std::vector<double> vecX = getVector(X, i, GET_VECTOR_TYPE_ROW);
        std::cout << "vec X :";
        printVector(vecX);

        for(auto j=0; j<colY; j++) {

            // Y[j]列の成分を取り出す
            std::vector<double> vecY = getVector(Y, j, GET_VECTOR_TYPE_COL);
            std::cout << "vec Y :";
            printVector(vecY);


            // 行列積[i][j]の成分はX[i]行とY[j]列のベクトル内積と等しい

             double prod = innerProduct(vecX, vecY);
             std::cout << "(i, j) :" << i << "," << j << "prod: " << prod << std::endl;
             C(i, j) = prod;
        }
    }

    return C;
}

std::vector<double> getVector(const Eigen::MatrixXd &X, int32_t idx, GetVectorType_t getType)
{
    std::vector<double> v;
    size_t rows = X.rows(), cols = X.cols();

    if( getType == GET_VECTOR_TYPE_COL ) {
        // pick column
        for(int32_t i=0; i<rows; i++) {
            v.push_back( X(i, idx) );
        }

    } else {
        // pick row
        for(int32_t i=0; i<cols; i++) {
            v.push_back( X(idx, i) );
        }
    }

    return v;
}

double innerProduct(const std::vector<double> &X, const std::vector<double> &Y)
{
    double sum = 0;
    std::cout << "x.size " << X.size() << " y.size " << Y.size() << std::endl;
    
    if( X.size() != Y.size() ) {
        throw std::runtime_error("size error!");
    }

    // std::cout << "X: ";
    // PRINT_VEC( const_cast<std::vector<double> &>(X) );

    // std::cout << "Y: ";
    // PRINT_VEC( const_cast<std::vector<double> &>(Y) );

    for( auto i=0; i<X.size(); i++ ) {
        // std::cout << "sum: " << sum << std::endl;
        // std::cout << "X: " << X[i] << "Y: " << Y[i] << std::endl;
        double mul = X[i] * Y[i];
        // std::cout << "mul: " << mul << std::endl;
        sum += mul;
    }

    return sum;
}

double softThreshold( double val, double thresh )
{
    if( std::abs(val) <= thresh ) {
        return 0.0;

    } else {
        return val - ( thresh * getSign(val) );
    }
}

Eigen::MatrixXd coordinateDescent(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, double alpha, int32_t nIterate)
{
    const int32_t sampleNum   = X.rows();     // Column : サンプル数
    const int32_t featureNum  = X.cols();     // Row    : 特徴量

    // std::cout << "X:" << X << std::endl;
    // std::cout << "Y:" << Y << std::endl;

    // 重みを初期化
    Eigen::VectorXd weight  = Eigen::VectorXd::Zero(featureNum);
    Eigen::VectorXd r_j     = Eigen::VectorXd::Zero(featureNum);
    for( auto i=0; i<nIterate; i++ ) {
        for( auto j=0; j<featureNum; j++ ) {
            weight(j) = 0.0;

            Eigen::MatrixXd dotXw = dotMatrix(X, weight);
            Eigen::MatrixXd r_j = subMatrix(Y, dotXw);

            // std::cout << "dotXw: = [" << std::endl << dotXw << "]" << std::endl;
            // std::cout << "r_j  : = [" << std::endl << r_j   << "]" << std::endl;

            std::vector<double> vecX    = getVector(X,  j, GET_VECTOR_TYPE_COL);
            std::vector<double> vecR_j  = getVector(r_j, 0, GET_VECTOR_TYPE_COL);
#if 0
            std::cout << "vecX: ";
            printVector(vecX);

            std::cout << "vecR_j: ";
            printVector(vecR_j);
#endif
            double prod     = innerProduct(vecX, vecR_j) / sampleNum;
            double retVal   = softThreshold(prod, alpha);
            weight(j)       = retVal;
        }
    }

    // std::cout << "weight = [ " << std::endl << weight << "]" << std::endl;
    return weight;
}
