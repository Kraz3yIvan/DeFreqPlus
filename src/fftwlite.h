// Lite version of fftw header for DeFreq+
// Based on fftw3.h typedefs for delayed loading
// Updated for 64-bit / AviSynth+ by modernization effort
//
#ifndef __FFTWLITE_H__
#define __FFTWLITE_H__

#include <cstddef>

typedef float fftwf_complex[2];
typedef struct fftwf_plan_s *fftwf_plan;

typedef void *(*fftwf_malloc_proc)(size_t n);
typedef void (*fftwf_free_proc)(void *p);
typedef fftwf_plan (*fftwf_plan_dft_r2c_2d_proc)(int ny, int nx, float *in, fftwf_complex *out, unsigned flags);
typedef fftwf_plan (*fftwf_plan_dft_c2r_2d_proc)(int ny, int nx, fftwf_complex *in, float *out, unsigned flags);
typedef void (*fftwf_destroy_plan_proc)(fftwf_plan p);
typedef void (*fftwf_execute_dft_r2c_proc)(fftwf_plan p, float *in, fftwf_complex *out);
typedef void (*fftwf_execute_dft_c2r_proc)(fftwf_plan p, fftwf_complex *in, float *out);

#define FFTW_MEASURE  (0U)
#define FFTW_ESTIMATE (1U << 6)

#endif // __FFTWLITE_H__
