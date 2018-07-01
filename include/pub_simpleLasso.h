#ifndef SIMPLELASSO_H
#define SIMPLELASSO_H

// Include headers
#include <cstdint>
#include <vector>

namespace boost { namespace python { namespace numpy {
    class ndarray;
} } }   // namespace boost::python::numpy

// Prototypes
boost::python::numpy::ndarray subMatrix( const boost::python::numpy::ndarray &X, const boost::python::numpy::ndarray &Y );
boost::python::numpy::ndarray addMatrix( const boost::python::numpy::ndarray &X, const boost::python::numpy::ndarray &Y );
boost::python::numpy::ndarray dotMatrix( const boost::python::numpy::ndarray &X, const boost::python::numpy::ndarray &Y );
boost::python::numpy::ndarray coordinateDescent( boost::python::numpy::ndarray X, boost::python::numpy::ndarray Y, double alpha, int32_t nIterate );
std::vector<double> getVectorByndarray( const boost::python::numpy::ndarray &X, int32_t offset, int32_t getType );
double innerProduct(const std::vector<double> &X, const std::vector<double> &Y);
double softThreshold( double val, double thresh );
inline int32_t getSign(double val){ return (val > 0) - (val < 0); }

#endif  /* SIMPLELASSO_H */