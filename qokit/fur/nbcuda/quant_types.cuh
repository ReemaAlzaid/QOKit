#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

/* Map QBITS → packed integer type holding {real,imag}.  */
template<int QBITS> struct qamp_t;
template<> struct qamp_t<32> { using type = double2; };   // fp64 legacy
template<> struct qamp_t<16> { using type = short2;  };   // int16 + scale
template<> struct qamp_t<8>  { using type = char2;   };   // int8  + scale
