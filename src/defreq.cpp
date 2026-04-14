/*
    DeFreq+ plugin for AviSynth+ - Interference frequency remover

    Original DeFreq Copyright (C) 2004-2006 A.G.Balakhnin aka Fizick
    http://avisynth.org.ru

    DeFreq+ port (2026):
      - 64-bit build support
      - High bit depth planar YUV (8-16 bit, 420/422/444)
      - AviSynth+ C++ API (AvisynthPluginInit3)
      - Removed legacy YUY2 code path
      - Dynamic FFTW3 loading with multiple DLL name fallbacks
      - Faithful line-by-line port of original Fizick algorithms

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation.

    Plugin uses external FFTW library version 3 (http://www.fftw.org).
*/

#include <windows.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "avisynth.h"
#include "fftwlite.h"
#include "info.h"

using std::min;
using std::max;

// ---------------------------------------------------------------------------
class DeFreq : public GenericVideoFilter {

    float fx, fy, dx, dy, sharp;
    float fx2, fy2, dx2, dy2, sharp2;
    float fx3, fy3, dx3, dy3, sharp3;
    float fx4, fy4, dx4, dy4, sharp4;
    float cutx, cuty;
    int   plane;
    int   show;
    bool  info;
    bool  measure;

    float          *in;
    fftwf_complex  *out;
    fftwf_plan      plan, plani;

    int             nx, ny;      // = actual image plane dimensions
    int             outwidth;    // nx/2 + 1
    float          *psd;
    int             naverage;

    int             bits_per_sample;
    int             max_pixel_value;
    float           neutral_value;
    float           input_scale;   // maps pixel values to 8-bit-equivalent range for FFT
    float           output_scale;  // maps FFT results back to native bit depth

    HINSTANCE hinstLib;
    fftwf_malloc_proc            fftwf_malloc;
    fftwf_free_proc              fftwf_free;
    fftwf_plan_dft_r2c_2d_proc   fftwf_plan_dft_r2c_2d;
    fftwf_plan_dft_c2r_2d_proc   fftwf_plan_dft_c2r_2d;
    fftwf_destroy_plan_proc      fftwf_destroy_plan;
    fftwf_execute_dft_r2c_proc   fftwf_execute_dft_r2c;
    fftwf_execute_dft_c2r_proc   fftwf_execute_dft_c2r;

    // Faithful ports of original Fizick helper functions
    void SearchBoxBackground(float *psd, int outwidth, int height,
        float fx, float fy, float dx, float dy, float &background);
    void SearchPeak(float *psd, int outwidth, int height,
        float fx, float fy, float dx, float dy,
        float *fxPeak, float *fyPeak, float *sharpPeak);
    void CleanWindow(fftwf_complex *out, int outwidth, int height,
        float fx, float fy, float dx, float dy,
        float fxPeak, float fyPeak, float sharpPeak);
    void CleanHigh(fftwf_complex *outp, int outwidth, int height, float cutx, float cuty);
    void GetFFT2minmax(float *psd, int outwidth, int height, float *fft2min, float *fft2max);
    void DrawSearchBox(float *psd, int outwidth, int height,
        float fx, float fy, float dx, float dy, float fftval);
    void FrequencySwitchOn(fftwf_complex *outp, int outwidth, int height,
        float fx, float fy, float setvalue);

    // Main processing function
    void DeFreqProcess(uint8_t *srcp0, int src_height, int src_width, int src_pitch,
        float *fxPeak, float *fyPeak, float *sharpPeak,
        float *fxPeak2, float *fyPeak2, float *sharpPeak2,
        float *fxPeak3, float *fyPeak3, float *sharpPeak3,
        float *fxPeak4, float *fyPeak4, float *sharpPeak4);

    // Bit-depth helpers
    inline float ReadPixel(const uint8_t *ptr, int x) const {
        if (bits_per_sample == 8) return static_cast<float>(ptr[x]);
        else return static_cast<float>(reinterpret_cast<const uint16_t*>(ptr)[x]);
    }
    inline void WritePixelClamped(uint8_t *ptr, int x, int ival) const {
        ival = max(0, min(max_pixel_value, ival));
        if (bits_per_sample == 8) ptr[x] = static_cast<uint8_t>(ival);
        else reinterpret_cast<uint16_t*>(ptr)[x] = static_cast<uint16_t>(ival);
    }
    inline int PixelBytes() const { return (bits_per_sample > 8) ? 2 : 1; }

public:
    DeFreq(PClip _child,
        float _fx, float _fy, float _dx, float _dy, float _sharp,
        float _fx2, float _fy2, float _dx2, float _dy2, float _sharp2,
        float _fx3, float _fy3, float _dx3, float _dy3, float _sharp3,
        float _fx4, float _fy4, float _dx4, float _dy4, float _sharp4,
        float _cutx, float _cuty, int _plane, int _show, bool _info, bool _measure,
        IScriptEnvironment *env);
    ~DeFreq();
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
};

// ---------------------------------------------------------------------------
static void mem_set_plane(uint8_t *dest, int value16, int height, int row_bytes, int pitch, int bps)
{
    if (bps == 8) {
        uint8_t v8 = static_cast<uint8_t>(value16);
        for (int h = 0; h < height; h++) { memset(dest, v8, row_bytes); dest += pitch; }
    } else {
        uint16_t v16 = static_cast<uint16_t>(value16);
        int wp = row_bytes / 2;
        for (int h = 0; h < height; h++) {
            uint16_t *d = reinterpret_cast<uint16_t*>(dest);
            for (int w = 0; w < wp; w++) d[w] = v16;
            dest += pitch;
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
DeFreq::DeFreq(PClip _child,
    float _fx, float _fy, float _dx, float _dy, float _sharp,
    float _fx2, float _fy2, float _dx2, float _dy2, float _sharp2,
    float _fx3, float _fy3, float _dx3, float _dy3, float _sharp3,
    float _fx4, float _fy4, float _dx4, float _dy4, float _sharp4,
    float _cutx, float _cuty, int _plane, int _show, bool _info, bool _measure,
    IScriptEnvironment *env)
    : GenericVideoFilter(_child),
      fx(_fx), fy(_fy), dx(_dx), dy(_dy), sharp(_sharp),
      fx2(_fx2), fy2(_fy2), dx2(_dx2), dy2(_dy2), sharp2(_sharp2),
      fx3(_fx3), fy3(_fy3), dx3(_dx3), dy3(_dy3), sharp3(_sharp3),
      fx4(_fx4), fy4(_fy4), dx4(_dx4), dy4(_dy4), sharp4(_sharp4),
      cutx(_cutx), cuty(_cuty), plane(_plane), show(_show), info(_info), measure(_measure),
      in(nullptr), out(nullptr), psd(nullptr), hinstLib(nullptr),
      plan(nullptr), plani(nullptr)
{
    if (fx < 0 || fx > 100 || fx2 < 0 || fx2 > 100 || fx3 < 0 || fx3 > 100 || fx4 < 0 || fx4 > 100)
        env->ThrowError("DeFreq: fx,fx2,fx3,fx4 must be from 0.0 to 100.0%%");
    if (fy < -100 || fy > 100 || fy2 < -100 || fy2 > 100 || fy3 < -100 || fy3 > 100 || fy4 < -100 || fy4 > 100)
        env->ThrowError("DeFreq: fy,fy2,fy3,fy4 must be from -100.0 to 100.0%%");
    if (cutx < 0 || cutx > 300) env->ThrowError("DeFreq: cutx must be from 0.0 to 300.0%%");
    if (cuty < 0 || cuty > 300) env->ThrowError("DeFreq: cuty must be from 0.0 to 300.0%%");
    if (dx < 0 || dx > 50 || dx2 < 0 || dx2 > 50 || dx3 < 0 || dx3 > 50 || dx4 < 0 || dx4 > 50)
        env->ThrowError("DeFreq: dx,dx2,dx3,dx4 must be from 0.0 to 50.0%%");
    if (dy < 0 || dy > 50 || dy2 < 0 || dy2 > 50 || dy3 < 0 || dy3 > 50 || dy4 < 0 || dy4 > 50)
        env->ThrowError("DeFreq: dy,dy2,dy3,dy4 must be from 0.0 to 50.0%%");
    if (plane < 0 || plane > 2)
        env->ThrowError("DeFreq: plane must be 0 (Y), 1 (U), or 2 (V)");
    if (!vi.IsPlanar() || vi.IsRGB())
        env->ThrowError("DeFreq: input must be planar YUV");

    bits_per_sample = vi.BitsPerComponent();
    if (bits_per_sample != 8 && bits_per_sample != 10 && bits_per_sample != 12 &&
        bits_per_sample != 14 && bits_per_sample != 16)
        env->ThrowError("DeFreq: only 8-16 bit integer formats supported");

    max_pixel_value = (1 << bits_per_sample) - 1;
    neutral_value   = static_cast<float>(1 << (bits_per_sample - 1));

    // Normalise all bit depths to 8-bit-equivalent range for FFT processing.
    // At 8-bit input_scale==1 so behaviour is identical to the original code.
    // At higher bit depths the FFT operates on the same magnitude range,
    // eliminating float-precision issues in CleanWindow's spectrum modification.
    input_scale  = 255.0f / static_cast<float>(max_pixel_value);
    output_scale = static_cast<float>(max_pixel_value) / 255.0f;

    // Exact image dimensions — same as original Fizick code
    if (plane == 0) {
        nx = vi.width;
        ny = vi.height;
    } else {
        nx = vi.width  >> vi.GetPlaneWidthSubsampling(PLANAR_U);
        ny = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);
    }

    // Load FFTW3 DLL
    const char *dll_names[] = {
        "libfftw3f-3.dll", "fftw3f.dll", "fftw3.dll", "libfftw3-3.dll", nullptr
    };
    hinstLib = nullptr;
    for (int i = 0; dll_names[i]; i++) {
        hinstLib = LoadLibraryA(dll_names[i]);
        if (hinstLib) break;
    }
    if (hinstLib) {
        fftwf_malloc           = (fftwf_malloc_proc)          GetProcAddress(hinstLib, "fftwf_malloc");
        fftwf_free             = (fftwf_free_proc)            GetProcAddress(hinstLib, "fftwf_free");
        fftwf_plan_dft_r2c_2d  = (fftwf_plan_dft_r2c_2d_proc)GetProcAddress(hinstLib, "fftwf_plan_dft_r2c_2d");
        fftwf_plan_dft_c2r_2d  = (fftwf_plan_dft_c2r_2d_proc)GetProcAddress(hinstLib, "fftwf_plan_dft_c2r_2d");
        fftwf_destroy_plan     = (fftwf_destroy_plan_proc)    GetProcAddress(hinstLib, "fftwf_destroy_plan");
        fftwf_execute_dft_r2c  = (fftwf_execute_dft_r2c_proc) GetProcAddress(hinstLib, "fftwf_execute_dft_r2c");
        fftwf_execute_dft_c2r  = (fftwf_execute_dft_c2r_proc) GetProcAddress(hinstLib, "fftwf_execute_dft_c2r");
    }
    if (!hinstLib || !fftwf_malloc || !fftwf_free || !fftwf_plan_dft_r2c_2d ||
        !fftwf_plan_dft_c2r_2d || !fftwf_destroy_plan || !fftwf_execute_dft_r2c || !fftwf_execute_dft_c2r) {
        if (hinstLib) FreeLibrary(hinstLib);
        env->ThrowError("DeFreq: cannot load FFTW3 DLL!");
    }

    in  = (float *)fftwf_malloc(sizeof(float) * nx * ny);
    outwidth = nx / 2 + 1;
    out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * ny * outwidth);
    psd = (float *)malloc(sizeof(float) * ny * outwidth);
    if (!in || !out || !psd)
        env->ThrowError("DeFreq: memory allocation failed");

    // Always use FFTW_ESTIMATE for safety with arbitrary dimensions
    plan  = fftwf_plan_dft_r2c_2d(ny, nx, in, out, FFTW_ESTIMATE);
    plani = fftwf_plan_dft_c2r_2d(ny, nx, out, in, FFTW_ESTIMATE);
    if (!plan || !plani)
        env->ThrowError("DeFreq: FFTW plan creation failed (%dx%d)", nx, ny);

    // Init PSD to zero — matching original
    naverage = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < outwidth; x++) {
            psd[y * outwidth + x] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
DeFreq::~DeFreq() {
    if (plan)     fftwf_destroy_plan(plan);
    if (plani)    fftwf_destroy_plan(plani);
    if (in)       fftwf_free(in);
    if (out)      fftwf_free(out);
    if (psd)      free(psd);
    if (hinstLib) FreeLibrary(hinstLib);
}

// ---------------------------------------------------------------------------
// FAITHFUL PORT of original Fizick SearchBoxBackground (v0.7)
void DeFreq::SearchBoxBackground(float *psd, int outwidth, int height,
    float fx, float fy, float dx, float dy, float &background)
{
    float fxmin = max(0.0f, fx - dx);
    float fxmax = min(100.0f, fx + dx);
    float fymin = max(-100.0f, fy - dy);
    float fymax = min(100.0f, fy + dy);

    int ixmin = int(fxmin * outwidth) / 100;
    int ixmax = int(fxmax * outwidth) / 100;
    int iymin = int(fymin * height) / 200;
    int iymax = int(fymax * height) / 200;

    int hmin = (iymin >= 0) ? iymin : iymin + height - 1;
    int hmax = (iymax >= 0) ? iymax : iymax + height - 1;
    int w, h;
    int counter = 0;
    float sum = 0;
    if (hmin <= hmax) {
        float *p = psd + outwidth * hmin;
        for (w = ixmin; w <= ixmax; w++) sum += sqrtf(p[w]);
        for (h = hmin; h < hmax; h++) {
            sum += sqrtf(p[ixmin]);
            sum += sqrtf(p[ixmax]);
            p += outwidth;
        }
        for (w = ixmin; w <= ixmax; w++) sum += sqrtf(p[w]);
        counter += (ixmax - ixmin + 1) * 2 + (hmax - hmin + 1) * 2;
    } else {
        float *p = psd;
        for (h = 0; h < hmax; h++) {
            sum += sqrtf(p[ixmin]);
            sum += sqrtf(p[ixmax]);
            p += outwidth;
        }
        for (w = ixmin; w < ixmax; w++) sum += sqrtf(p[w]);
        p += outwidth * (hmin - hmax);
        for (w = ixmin; w < ixmax; w++) sum += sqrtf(p[w]);
        for (h = hmin; h < height; h++) {
            sum += sqrtf(p[ixmin]);
            sum += sqrtf(p[ixmax]);
            p += outwidth;
        }
        counter += hmax * 2 + (ixmax - ixmin) + (height - hmin) * 2;
    }
    background = sum * sum / ((float)counter * counter + 0.0001f);
}

// ---------------------------------------------------------------------------
// FAITHFUL PORT of original Fizick SearchPeak
void DeFreq::SearchPeak(float *psd, int outwidth, int height,
    float fx, float fy, float dx, float dy,
    float *fxPeak, float *fyPeak, float *sharpPeak)
{
    float fxmin = max(0.0f, fx - dx);
    float fxmax = min(100.0f, fx + dx);
    float fymin = max(-100.0f, fy - dy);
    float fymax = min(100.0f, fy + dy);

    int ixmin = int(fxmin * outwidth) / 100;
    int ixmax = int(fxmax * outwidth) / 100;
    int iymin = int(fymin * height) / 200;
    int iymax = int(fymax * height) / 200;

    float fftmax = -1;
    int hmax = 0, wmax = 0;
    int h, w;

    float fftbackground;
    SearchBoxBackground(psd, outwidth, height, fx, fy, dx, dy, fftbackground);

    float *p = psd;
    for (h = 0; h < height / 2; h++) {
        if (h >= iymin && h <= iymax) {
            for (w = ixmin; w <= ixmax; w++) {
                float fftcur = p[w];
                if (fftcur > fftmax) { fftmax = fftcur; hmax = h; wmax = w; }
            }
        }
        p += outwidth;
    }
    for (h = height / 2; h < height; h++) {
        if (h >= height + iymin - 1 && h <= height + iymax - 1) {
            for (w = ixmin; w <= ixmax; w++) {
                float fftcur = p[w];
                if (fftcur > fftmax) { fftmax = fftcur; hmax = h; wmax = w; }
            }
        }
        p += outwidth;
    }

    *fxPeak = (wmax * 100.0f) / outwidth;
    *fyPeak = (hmax < height / 2) ? (hmax * 200.0f) / height : (hmax - height + 1) * 200.0f / height;

    if (fftbackground > 0)
        *sharpPeak = fftmax / fftbackground;
    else
        *sharpPeak = 1;
}

// ---------------------------------------------------------------------------
// FAITHFUL PORT of original Fizick CleanWindow (v0.7)
void DeFreq::CleanWindow(fftwf_complex *out, int outwidth, int height,
    float fx, float fy, float dx, float dy,
    float fxPeak, float fyPeak, float sharpPeak)
{
    fftwf_complex *outp = out;

    int w_Peak = int(fxPeak * outwidth) / 100;
    int h_Peak = (fyPeak > 0) ? int(fyPeak * height) / 200 : int(fyPeak * height) / 200 + height - 1;
    outp += outwidth * h_Peak;
    float fftmax = (outp[w_Peak][0]) * (outp[w_Peak][0]) + (outp[w_Peak][1]) * (outp[w_Peak][1]);
    float fftbackground = fftmax / sharpPeak;

    outp = out;

    float fxmin = max(0.0f, fx - dx);
    float fxmax = min(100.0f, fx + dx);
    float fymin = max(-100.0f, fy - dy);
    float fymax = min(100.0f, fy + dy);

    int ixmin = int(fxmin * outwidth) / 100;
    int ixmax = int(fxmax * outwidth) / 100;
    int iymin = int(fymin * height) / 200;
    int iymax = int(fymax * height) / 200;

    int h, w;
    for (h = 0; h < height / 2; h++) {
        if (h >= iymin && h <= iymax) {
            for (w = ixmin; w <= ixmax; w++) {
                float fftcur = (outp[w][0]) * (outp[w][0]) + (outp[w][1]) * (outp[w][1]);
                if (fftcur > fftbackground) {
                    float f = sqrtf(fftbackground / fftcur);
                    outp[w][0] *= f;
                    outp[w][1] *= f;
                }
            }
        }
        outp += outwidth;
    }
    for (h = height / 2; h < height; h++) {
        if (h >= height + iymin - 1 && h <= height + iymax - 1) {
            for (w = ixmin; w <= ixmax; w++) {
                float fftcur = (outp[w][0]) * (outp[w][0]) + (outp[w][1]) * (outp[w][1]);
                if (fftcur > fftbackground) {
                    float f = sqrtf(fftbackground / fftcur);
                    outp[w][0] *= f;
                    outp[w][1] *= f;
                }
            }
        }
        outp += outwidth;
    }
}

// ---------------------------------------------------------------------------
// FAITHFUL PORT of original Fizick CleanHigh
void DeFreq::CleanHigh(fftwf_complex *outp, int outwidth, int height, float cutx, float cuty)
{
    int w, h;
    float fh, fw, f;
    float invcutx = 100.0f / (cutx * outwidth);
    float invcuty = 200.0f / (cuty * height);
    for (h = 0; h < height / 2; h++) {
        fh = float(h) * invcuty; fh *= fh;
        for (w = 0; w < outwidth; w++) {
            fw = float(w) * invcutx; fw *= fw;
            f = 1 / (1 + fh + fw);
            outp[w][0] *= f; outp[w][1] *= f;
        }
        outp += outwidth;
    }
    for (h = height / 2; h < height; h++) {
        fh = float(height - h - 1) * invcuty; fh *= fh;
        for (w = 0; w < outwidth; w++) {
            fw = float(w) * invcutx; fw *= fw;
            f = 1 / (1 + fh + fw);
            outp[w][0] *= f; outp[w][1] *= f;
        }
        outp += outwidth;
    }
}

// ---------------------------------------------------------------------------
void DeFreq::GetFFT2minmax(float *psd, int outwidth, int height, float *fft2min, float *fft2max)
{
    float psdmin = 1.0e14f;
    float psdmax = 0;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < outwidth; w++) {
            float fft2cur = psd[w];
            if (fft2cur > psdmax) psdmax = fft2cur;
        }
        psd += outwidth;
    }
    if (psdmax == 0) psdmax = 1.0f;
    psdmin = psdmax * 1.0e-13f;
    *fft2min = psdmin;
    *fft2max = psdmax;
}

// ---------------------------------------------------------------------------
// FAITHFUL PORT of original Fizick DrawSearchBox
void DeFreq::DrawSearchBox(float *psd, int outwidth, int height,
    float fx, float fy, float dx, float dy, float fftvalue)
{
    float fxmin = max(0.0f, fx - dx);
    float fxmax = min(100.0f, fx + dx);
    float fymin = max(-100.0f, fy - dy);
    float fymax = min(100.0f, fy + dy);

    int ixmin = int(fxmin * outwidth) / 100;
    int ixmax = int(fxmax * outwidth) / 100;
    int iymin = int(fymin * height) / 200;
    int iymax = int(fymax * height) / 200;

    int hmin = (iymin >= 0) ? iymin : iymin + height - 1;
    int hmax = (iymax >= 0) ? iymax : iymax + height - 1;
    int w, h;
    if (hmin <= hmax) {
        float *p = psd + outwidth * hmin;
        for (w = ixmin; w <= ixmax; w++) p[w] = fftvalue;
        for (h = hmin; h < hmax; h++) {
            p[ixmin] = fftvalue; p[ixmax] = fftvalue; p += outwidth;
        }
        for (w = ixmin; w <= ixmax; w++) p[w] = fftvalue;
    } else {
        float *p = psd;
        for (h = 0; h < hmax; h++) { p[ixmin] = fftvalue; p[ixmax] = fftvalue; p += outwidth; }
        for (w = ixmin; w < ixmax; w++) p[w] = fftvalue;
        p += outwidth * (hmin - hmax);
        for (w = ixmin; w < ixmax; w++) p[w] = fftvalue;
        for (h = hmin; h < height; h++) { p[ixmin] = fftvalue; p[ixmax] = fftvalue; p += outwidth; }
    }
}

// ---------------------------------------------------------------------------
void DeFreq::FrequencySwitchOn(fftwf_complex *outp, int outwidth, int height,
    float fx, float fy, float setvalue)
{
    int ix = int(fx * outwidth) / 100;
    int iy = int(fy * height) / 200;
    int h = (iy >= 0) ? iy : iy + height - 1;
    outp += outwidth * h;
    outp[ix][0] = setvalue;
}

// ---------------------------------------------------------------------------
// FAITHFUL PORT of original Fizick DeFreqYV12 — adapted for high bit depth
void DeFreq::DeFreqProcess(uint8_t *srcp0, int src_height, int src_width, int src_pitch,
    float *fxPeak, float *fyPeak, float *sharpPeak,
    float *fxPeak2, float *fyPeak2, float *sharpPeak2,
    float *fxPeak3, float *fyPeak3, float *sharpPeak3,
    float *fxPeak4, float *fyPeak4, float *sharpPeak4)
{
    float *inp = in;
    fftwf_complex *outp = out;

    int w, h;
    int width_2 = src_width / 2;

    uint8_t *srcp = srcp0;
    uint8_t *dstp = srcp0;

    // pixel to float — normalised to 8-bit-equivalent range for all bit depths
    for (h = 0; h < src_height; h++) {
        for (w = 0; w < src_width; w++)
            inp[w] = ReadPixel(srcp, w) * input_scale;
        srcp += src_pitch;
        inp += nx;
    }
    inp -= nx * src_height;
    srcp -= src_pitch * src_height;

    fftwf_execute_dft_r2c(plan, inp, out); // do FFT

    if (show == 2)
        naverage += 1;
    else
        naverage = 1;
    float faverage = 1.0f / naverage;
    // PSD update — using moving pointer exactly like original
    for (h = 0; h < src_height; h++) {
        for (w = 0; w < outwidth; w++)
            psd[w] = psd[w] * (1 - faverage) + (outp[w][0] * outp[w][0] + outp[w][1] * outp[w][1]) * faverage;
        psd += outwidth;
        outp += outwidth;
    }
    psd -= outwidth * src_height;
    outp -= outwidth * src_height;

    // search Peaks
    if (fx > 0 || fy != 0)
        SearchPeak(psd, outwidth, src_height, fx, fy, dx, dy, fxPeak, fyPeak, sharpPeak);
    if (fx2 > 0 || fy2 != 0)
        SearchPeak(psd, outwidth, src_height, fx2, fy2, dx2, dy2, fxPeak2, fyPeak2, sharpPeak2);
    if (fx3 > 0 || fy3 != 0)
        SearchPeak(psd, outwidth, src_height, fx3, fy3, dx3, dy3, fxPeak3, fyPeak3, sharpPeak3);
    if (fx4 > 0 || fy4 != 0)
        SearchPeak(psd, outwidth, src_height, fx4, fy4, dx4, dy4, fxPeak4, fyPeak4, sharpPeak4);

    if (show) {
        // show mode — EXACT COPY of original Fizick pointer walk, adapted for bit depth
        float fft2min = 0, fft2max = 0;
        GetFFT2minmax(psd, outwidth, src_height, &fft2min, &fft2max);

        if (fx > 0 || fy != 0)   DrawSearchBox(psd, outwidth, src_height, fx, fy, dx, dy, fft2max);
        if (fx2 > 0 || fy2 != 0) DrawSearchBox(psd, outwidth, src_height, fx2, fy2, dx2, dy2, fft2max);
        if (fx3 > 0 || fy3 != 0) DrawSearchBox(psd, outwidth, src_height, fx3, fy3, dx3, dy3, fft2max);
        if (fx4 > 0 || fy4 != 0) DrawSearchBox(psd, outwidth, src_height, fx4, fy4, dx4, dy4, fft2max);

        float logmin = logf(fft2min);
        float logmax = logf(fft2max);
        float fac = ((float)max_pixel_value + 0.5f) / (logmax - logmin);

        // show fft log PSD on left half — EXACT pointer walk from original
        psd += (src_height / 2) * outwidth; // middle positive
        for (h = 0; h < src_height / 2; h++) {
            psd -= outwidth;
            for (w = 0; w < width_2; w++) {
                int dstcur = (int)(fac * (logf(psd[w] + 1e-15f) - logmin));
                WritePixelClamped(dstp, w, dstcur);
            }
            dstp += src_pitch;
        }
        psd += (src_height) * outwidth; // bottom negative
        for (h = src_height / 2; h < src_height; h++) {
            psd -= outwidth;
            for (w = 0; w < width_2; w++) {
                int dstcur = (int)(fac * (logf(psd[w] + 1e-15f) - logmin));
                WritePixelClamped(dstp, w, dstcur);
            }
            dstp += src_pitch;
        }
        dstp -= src_pitch * src_height;
        psd -= outwidth * (src_height - (src_height / 2));

        // test frequency stripes — exact copy of original
        for (h = 0; h < src_height; h++) {
            for (w = 0; w < outwidth; w++) {
                outp[w][0] = 0; outp[w][1] = 0;
            }
            outp += outwidth;
        }
        outp -= outwidth * src_height;

        outp[0][0] = neutral_value * input_scale; // DC = neutral grey (128 in normalised space)

        float weight = 0;
        if (fx > 0 || fy != 0)   weight += 1.0f;
        if (fx2 > 0 || fy2 != 0) weight += 1 / 1.4f;
        if (fx3 > 0 || fy3 != 0) weight += 1 / 2.0f;
        if (fx4 > 0 || fy4 != 0) weight += 1 / 2.8f;
        float setvalue = 60.0f / (weight + 0.0001f);
        // No bit-depth scaling needed: FFT operates in normalised 8-bit-equivalent space

        if (fx > 0 || fy != 0)   FrequencySwitchOn(outp, outwidth, src_height, fx, fy, setvalue);
        if (fx2 > 0 || fy2 != 0) FrequencySwitchOn(outp, outwidth, src_height, fx2, fy2, setvalue / 1.4f);
        if (fx3 > 0 || fy3 != 0) FrequencySwitchOn(outp, outwidth, src_height, fx3, fy3, setvalue / 2.0f);
        if (fx4 > 0 || fy4 != 0) FrequencySwitchOn(outp, outwidth, src_height, fx4, fy4, setvalue / 2.8f);

        fftwf_execute_dft_c2r(plani, outp, in); // iFFT

        // show stripes on right top quarter — scale back to native bit depth
        inp = in;
        dstp = srcp0;
        for (h = 0; h < src_height / 2; h++) {
            for (w = width_2; w < src_width; w++) {
                int result = (int)(inp[w] * output_scale + 0.5f);
                WritePixelClamped(dstp, w, result);
            }
            dstp += src_pitch;
            inp += nx;
        }
        dstp -= src_pitch * (src_height / 2);
        inp -= nx * (src_height / 2);

    } else {
        // work mode — exact copy of original
        bool clean = false;

        if (*sharpPeak > sharp) {
            CleanWindow(out, outwidth, src_height, fx, fy, dx, dy, *fxPeak, *fyPeak, *sharpPeak);
            clean = true;
        }
        if (*sharpPeak2 > sharp2) {
            CleanWindow(out, outwidth, src_height, fx2, fy2, dx2, dy2, *fxPeak2, *fyPeak2, *sharpPeak2);
            clean = true;
        }
        if (*sharpPeak3 > sharp3) {
            CleanWindow(out, outwidth, src_height, fx3, fy3, dx3, dy3, *fxPeak3, *fyPeak3, *sharpPeak3);
            clean = true;
        }
        if (*sharpPeak4 > sharp4) {
            CleanWindow(out, outwidth, src_height, fx4, fy4, dx4, dy4, *fxPeak4, *fyPeak4, *sharpPeak4);
            clean = true;
        }

        if (cutx > 0 && cuty > 0) {
            CleanHigh(out, outwidth, src_height, cutx, cuty);
            clean = true;
        }

        if (clean) {
            fftwf_execute_dft_c2r(plani, out, in); // iFFT

            float norm = output_scale / (nx * ny);  // combined iFFT normalisation + bit-depth restore
            inp = in;
            for (h = 0; h < src_height; h++) {
                for (w = 0; w < src_width; w++) {
                    int result = (int)(inp[w] * norm + 0.5f);
                    WritePixelClamped(dstp, w, result);
                }
                dstp += src_pitch;
                inp += nx;
            }
        }
    }
}

// ---------------------------------------------------------------------------
PVideoFrame __stdcall DeFreq::GetFrame(int n, IScriptEnvironment *env)
{
    float fxPeak = 0, fyPeak = 0, sharpPeak = 0;
    float fxPeak2 = 0, fyPeak2 = 0, sharpPeak2 = 0;
    float fxPeak3 = 0, fyPeak3 = 0, sharpPeak3 = 0;
    float fxPeak4 = 0, fyPeak4 = 0, sharpPeak4 = 0;

    PVideoFrame src = child->GetFrame(n, env);
    env->MakeWritable(&src);

    int avs_plane;
    switch (plane) {
        case 0:  avs_plane = PLANAR_Y; break;
        case 1:  avs_plane = PLANAR_U; break;
        default: avs_plane = PLANAR_V; break;
    }

    if (show) {
        // Set non-working planes to neutral
        int planes_to_clear[2]; int cc = 0;
        if (plane != 0) planes_to_clear[cc++] = PLANAR_Y;
        if (plane != 1) planes_to_clear[cc++] = PLANAR_U;
        if (plane != 2 && cc < 2) planes_to_clear[cc++] = PLANAR_V;
        for (int i = 0; i < cc; i++) {
            uint8_t *p = src->GetWritePtr(planes_to_clear[i]);
            int pitch = src->GetPitch(planes_to_clear[i]);
            int rowsize = src->GetRowSize(planes_to_clear[i]);
            int height = src->GetHeight(planes_to_clear[i]);
            mem_set_plane(p, (int)neutral_value, height, rowsize, pitch, bits_per_sample);
        }
    }

    uint8_t *srcp = src->GetWritePtr(avs_plane);
    int src_pitch = src->GetPitch(avs_plane);
    int src_width_bytes = src->GetRowSize(avs_plane);
    int src_height = src->GetHeight(avs_plane);
    int src_width_pixels = src_width_bytes / PixelBytes();

    DeFreqProcess(srcp, src_height, src_width_pixels, src_pitch,
        &fxPeak, &fyPeak, &sharpPeak, &fxPeak2, &fyPeak2, &sharpPeak2,
        &fxPeak3, &fyPeak3, &sharpPeak3, &fxPeak4, &fyPeak4, &sharpPeak4);

    if (info && bits_per_sample == 8) {
        char messagebuf[64];
        int x0 = vi.width / 20 + 1; int y0 = 0;
        DrawString(src, x0, y0++, "DeFreq+ peaks:");
        if (fx > 0 || fy != 0) {
            sprintf(messagebuf, "x=%.1f y=%.1f", fxPeak, fyPeak); DrawString(src, x0, y0++, messagebuf);
            sprintf(messagebuf, sharpPeak > sharp ? "SHARP=%.1f" : "sharp=%.1f", sharpPeak); DrawString(src, x0, y0++, messagebuf);
        }
        if (fx2 > 0 || fy2 != 0) {
            sprintf(messagebuf, "x2=%.1f y2=%.1f", fxPeak2, fyPeak2); DrawString(src, x0, y0++, messagebuf);
            sprintf(messagebuf, sharpPeak2 > sharp2 ? "SHARP2=%.1f" : "sharp2=%.1f", sharpPeak2); DrawString(src, x0, y0++, messagebuf);
        }
        if (fx3 > 0 || fy3 != 0) {
            sprintf(messagebuf, "x3=%.1f y3=%.1f", fxPeak3, fyPeak3); DrawString(src, x0, y0++, messagebuf);
            sprintf(messagebuf, sharpPeak3 > sharp3 ? "SHARP3=%.1f" : "sharp3=%.1f", sharpPeak3); DrawString(src, x0, y0++, messagebuf);
        }
        if (fx4 > 0 || fy4 != 0) {
            sprintf(messagebuf, "x4=%.1f y4=%.1f", fxPeak4, fyPeak4); DrawString(src, x0, y0++, messagebuf);
            sprintf(messagebuf, sharpPeak4 > sharp4 ? "SHARP4=%.1f" : "sharp4=%.1f", sharpPeak4); DrawString(src, x0, y0++, messagebuf);
        }
    }
    return src;
}

// ---------------------------------------------------------------------------
AVSValue __cdecl Create_DeFreq(AVSValue args, void*, IScriptEnvironment *env) {
    return new DeFreq(args[0].AsClip(),
        (float)args[1].AsFloat(10.0), (float)args[2].AsFloat(-10.0),
        (float)args[3].AsFloat(1.5), (float)args[4].AsFloat(2.0), (float)args[5].AsFloat(50.0),
        (float)args[6].AsFloat(0), (float)args[7].AsFloat(0),
        (float)args[8].AsFloat(1.5), (float)args[9].AsFloat(2.0), (float)args[10].AsFloat(50.0),
        (float)args[11].AsFloat(0), (float)args[12].AsFloat(0),
        (float)args[13].AsFloat(1.5), (float)args[14].AsFloat(2.0), (float)args[15].AsFloat(50.0),
        (float)args[16].AsFloat(0), (float)args[17].AsFloat(0),
        (float)args[18].AsFloat(1.5), (float)args[19].AsFloat(2.0), (float)args[20].AsFloat(50.0),
        (float)args[21].AsFloat(0), (float)args[22].AsFloat(0),
        args[23].AsInt(0), args[24].AsInt(0), args[25].AsBool(false), args[26].AsBool(true), env);
}

// ---------------------------------------------------------------------------
const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char * __stdcall AvisynthPluginInit3(
    IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("DeFreq",
        "c[FX]f[FY]f[DX]f[DY]f[SHARP]f"
        "[FX2]f[FY2]f[DX2]f[DY2]f[SHARP2]f"
        "[FX3]f[FY3]f[DX3]f[DY3]f[SHARP3]f"
        "[FX4]f[FY4]f[DX4]f[DY4]f[SHARP4]f"
        "[CUTX]f[CUTY]f"
        "[PLANE]i[SHOW]i[INFO]b[MEASURE]b",
        Create_DeFreq, 0);
    return "`DeFreq' DeFreq+ plugin - FFT interference frequency remover (64-bit, high bit depth)";
}
