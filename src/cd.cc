#include <cstdint>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "pub_simpleLasso.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

np::ndarray addMatrix(const np::ndarray &X, const np::ndarray &Y)
{
    // 行列形状が不一致の場合は計算不可
    if( X.get_shape() != Y.get_shape() ) {
        throw std::runtime_error("shape error!");
    }

    const py::tuple shape = py::make_tuple( X.get_shape() );
    np::ndarray C = np::zeros( shape, np::dtype::get_builtin<double>() );

    for( auto i=0; i<X.shape(0); i++ ) {
        for( auto j=0; j<X.shape(1); j++ ) {
            C[i][j] = X[i][j] + Y[i][j];
        }
    }

    return C;
}

np::ndarray subMatrix(const np::ndarray &X, const np::ndarray &Y)
{
    // 行列形状が不一致の場合は計算不可
    if( X.get_shape() != Y.get_shape() ) {
        throw std::runtime_error("shape unmatch!!");
    }

    const py::tuple shape = py::make_tuple( X.get_shape() );
    np::ndarray C = np::zeros( shape, np::dtype::get_builtin<double>() );

    for(auto i=0; i<X.shape(0); i++) {
        for(auto j=0; j<X.shape(1); j++) {
            C[i][j] = X[i][j] - Y[i][j];
        }
    }

    return C;
}

np::ndarray dotMatrix(const np::ndarray &X, const np::ndarray &Y)
{
    if( (X.get_nd() > 2) || (Y.get_nd() > 2) ) {
        // とりあえず二次元までしか対応しない
        throw std::runtime_error("dimention error!!");
    }

    // 入力行列の行数と列数を取得
    auto rowX = X.shape(0);
    auto rowY = Y.shape(0);
    auto colX = X.shape(1);
    auto colY = Y.shape(1);

    // Xの列数とYの行数が合わない場合は計算不可
    if( colX != rowY ) {
        throw std::runtime_error("shape unmatch!!");
    }

    // 行列積の結果は左の行数と右の列数の要素が定義される
    const py::tuple shape = py::make_tuple(rowX, colY);
    np::ndarray C = np::zeros(shape, np::dtype::get_builtin<double>() );

    for(auto i=0; i<rowX; i++) {
        // X[i]行の成分を取り出す
        std::vector<double> vecX = getVectorByndarray(X, i, 0);

        for(auto j=0; j<colY; j++) {
            // Y[j]列の成分を取り出す
            std::vector<double> vecY = getVectorByndarray(Y, j, 1);

            // 行列積[i][j]の成分はX[i]行とY[j]列のベクトル内積と等しい
            C[i][j] = innerProduct(vecX, vecY);
        }
    }

    return C;
}

std::vector<double> getVectorByndarray(const np::ndarray &X, int32_t offset, int32_t getType)
{
    std::vector<double> v;

    if( getType == 0 ) {
        // pick row
        for(int32_t i=0; i<X.shape(0); i++) {
            v.push_back( py::extract<double>(X[offset][i]) );
        }

    } else {
        // pick column
        for(int32_t i=0; i<X.shape(1); i++) {
            v.push_back( py::extract<double>(X[i][offset]) );
        }
    }

    return v;
}
#if 0
np::ndarray getSignMatrix(const np::ndarray &X)
{
    const py::tuple shape = py::make_tuple( X.get_shape() );
    np::ndarray signArray = np::zeros( shape, np::dtype::get_builtin<double>() );

    for( auto i=0; i<signArray.shape(0); i++ ) {
        for( auto j=0; j<signArray.shape(1); j++ ) {
            
            auto elemX = X[i][j];
            auto &e = signArray[i][j];

            if( elemX > 0 ) {
                e = 1;

            } else if( elemX < 0 ) {
                e = -1;

            } else {
                e = 0;
            }
        }
   }

   return signArray;
}
#endif

double innerProduct(const std::vector<double> &X, const std::vector<double> &Y)
{
    double sum = 0;
    if( X.size() != Y.size() ) {
        throw std::runtime_error("size error!");
    }

    for( auto i : X ) {
        sum += X[i] * Y[i];
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

np::ndarray coordinateDescent( np::ndarray X, np::ndarray Y, double alpha, int32_t nIterate )
{
    const int32_t sampleNum   = X.shape(0);     // Column : サンプル数
    const int32_t featureNum  = X.shape(1);     // Row    : 特徴量

    // 重みを初期化
    py::tuple shape = py::make_tuple(0, featureNum);
    np::ndarray weight = np::zeros( shape, np::dtype::get_builtin<double>() );
    np::ndarray r_j    = np::zeros( shape, np::dtype::get_builtin<double>() );

    for( auto i=0; i<nIterate; i++ ) {
        for( auto j=0; j<featureNum; j++ ) {
            weight[j] = 0.0;

            np::ndarray dotXw = dotMatrix(X, weight);
            np::ndarray r_j = subMatrix(Y, dotXw);

            std::vector<double> vecX    = getVectorByndarray(X, j, 1);
            std::vector<double> vecR_j  = getVectorByndarray(r_j, 0, 0);

            double prod     = innerProduct(vecX, vecR_j) / sampleNum;
            double retVal   = softThreshold(prod, alpha);
            weight[j] = 1.0;
        }
    }
    return weight;
}

BOOST_PYTHON_MODULE(myCoodinateDescent)
{
    Py_Initialize();
    np::initialize();
    py::def("my_coordinate_descent", coordinateDescent);
}