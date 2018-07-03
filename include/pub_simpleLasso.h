#ifndef SIMPLELASSO_H
#define SIMPLELASSO_H

// Include headers
#include <cstdint>
#include <vector>
#include <Eigen/Core>

typedef enum {
    GET_VECTOR_TYPE_COL = 0,
    GET_VECTOR_TYPE_ROW,
} GetVectorType_t;

// Prototypes
Eigen::MatrixXd subMatrix( const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y );
Eigen::MatrixXd addMatrix( const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y );
Eigen::MatrixXd dotMatrix( const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y );
Eigen::MatrixXd coordinateDescent( Eigen::MatrixXd X, Eigen::MatrixXd Y, double alpha, int32_t nIterate );
std::vector<double> getVector(const Eigen::MatrixXd &X, int32_t idx, GetVectorType_t getType );
double innerProduct( const std::vector<double> &X, const std::vector<double> &Y );
double softThreshold( double val, double thresh );
inline int32_t getSign( double val )
{
    return (val > 0) - (val < 0);
}
inline void printVector( std::vector<double> &vec)
{
    for( auto i : vec ) {
        std::cout << i << ' ';
    }

    std::cout << std::endl;
}
#endif  /* SIMPLELASSO_H */