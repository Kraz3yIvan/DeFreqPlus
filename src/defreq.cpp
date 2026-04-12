/*
    DeFreq+ plugin for AviSynth+ - Interference frequency remover

    Original DeFreq Copyright (C) 2004-2006 A.G.Balakhnin aka Fizick
    http://avisynth.org.ru

    DeFreq+ modernization (2026):
      - 64-bit build support
      - High bit depth planar YUV (8-16 bit, 420/422/444)
      - YUV444P16 and all AviSynth+ planar formats
      - AviSynth+ C++ API
      - Removed legacy YUY2 code path
      - Dynamic FFTW3 loading with multiple DLL name fallbacks
      - Internal FFT dimension padding for robustness with all frame sizes

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation.

    Plugin uses external FFTW library version 3 (http://www.fftw.org).
    For 64-bit: place libfftw3f-3.dll (or fftw3.dll) in your PATH or
    AviSynth+ plugins64 folder.
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
// Find the next "FFT-friendly" size >= n.
// FFTW is fastest for sizes that are products of small primes (2, 3, 5, 7).
// We also enforce even sizes for the r2c transform symmetry.
static int NextFFTSize(int n)
{
    if (n <= 0) return 2;
    int candidate = (n % 2 == 0) ? n : n + 1;
    while (true) {
        int tmp = candidate;
        while (tmp % 2 == 0) tmp /= 2;
        while (tmp % 3 == 0) tmp /= 3;
        while (tmp % 5 == 0) tmp /= 5;
        while (tmp % 7 == 0) tmp /= 7;
        if (tmp == 1) return candidate;
        candidate += 2;
    }
}

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

    float          *fft_in;
    fftwf_complex  *fft_out;
    fftwf_plan      plan_fwd, plan_inv;

    int             img_width, img_height;  // actual image plane dimensions
    int             nx, ny;                 // padded FFT dimensions
    int             outwidth;               // nx/2 + 1
    float          *psd;
    int             naverage;

    int             bits_per_sample;
    int             max_pixel_value;
    float           neutral_value;

    HINSTANCE hinstLib;
    fftwf_malloc_proc            fftwf_malloc;
    fftwf_free_proc              fftwf_free;
    fftwf_plan_dft_r2c_2d_proc   fftwf_plan_dft_r2c_2d;
    fftwf_plan_dft_c2r_2d_proc   fftwf_plan_dft_c2r_2d;
    fftwf_destroy_plan_proc      fftwf_destroy_plan;
    fftwf_execute_dft_r2c_proc   fftwf_execute_dft_r2c;
    fftwf_execute_dft_c2r_proc   fftwf_execute_dft_c2r;

    void ProcessPlanar(uint8_t *srcp, int height, int width, int pitch,
        float *fxPeak, float *fyPeak, float *sharpPeak,
        float *fxPeak2, float *fyPeak2, float *sharpPeak2,
        float *fxPeak3, float *fyPeak3, float *sharpPeak3,
        float *fxPeak4, float *fyPeak4, float *sharpPeak4);
    void SearchPeak(float *psd, int outwidth, int height,
        float fx, float fy, float dx, float dy,
        float *fxPeak, float *fyPeak, float *sharpPeak);
    void SearchBoxBackground(float *psd, int outwidth, int height,
        float fx, float fy, float dx, float dy, float &background);
    void CleanWindow(fftwf_complex *out, int outwidth, int height,
        float fx, float fy, float dx, float dy,
        float fxPeak, float fyPeak, float sharpPeak);
    void CleanHigh(fftwf_complex *outp, int outwidth, int height, float cutx, float cuty);
    void GetFFT2minmax(float *psd, int outwidth, int height, float *fft2min, float *fft2max);
    void DrawSearchBox(float *psd, int outwidth, int height,
        float fx, float fy, float dx, float dy, float fftval);
    void FrequencySwitchOn(fftwf_complex *outp, int outwidth, int height,
        float fx, float fy, float setvalue);

    inline float ReadPixel(const uint8_t *ptr, int x) const {
        if (bits_per_sample == 8) return static_cast<float>(ptr[x]);
        else return static_cast<float>(reinterpret_cast<const uint16_t*>(ptr)[x]);
    }
    inline void WritePixelInt(uint8_t *ptr, int x, int ival) const {
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
        int width_pixels = row_bytes / 2;
        for (int h = 0; h < height; h++) {
            uint16_t *d = reinterpret_cast<uint16_t*>(dest);
            for (int w = 0; w < width_pixels; w++) d[w] = v16;
            dest += pitch;
        }
    }
}

// ---------------------------------------------------------------------------
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
      fft_in(nullptr), fft_out(nullptr), psd(nullptr), hinstLib(nullptr),
      plan_fwd(nullptr), plan_inv(nullptr)
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
        env->ThrowError("DeFreq: input must be planar YUV (YV12, YV24, YUV444P16, etc.)");

    bits_per_sample = vi.BitsPerComponent();
    if (bits_per_sample != 8 && bits_per_sample != 10 && bits_per_sample != 12 &&
        bits_per_sample != 14 && bits_per_sample != 16)
        env->ThrowError("DeFreq: only 8-16 bit integer formats supported");

    max_pixel_value = (1 << bits_per_sample) - 1;
    neutral_value   = static_cast<float>(1 << (bits_per_sample - 1));

    if (plane == 0) {
        img_width  = vi.width;
        img_height = vi.height;
    } else {
        img_width  = vi.width  >> vi.GetPlaneWidthSubsampling(PLANAR_U);
        img_height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);
    }

    // Pad to FFT-friendly dimensions (products of 2,3,5,7 only)
    nx = NextFFTSize(img_width);
    ny = NextFFTSize(img_height);

    const char *dll_names[] = {
        "libfftw3f-3.dll", "fftw3f.dll", "fftw3.dll", "libfftw3-3.dll", nullptr
    };
    hinstLib = nullptr;
    for (int i = 0; dll_names[i] != nullptr; i++) {
        hinstLib = LoadLibraryA(dll_names[i]);
        if (hinstLib != nullptr) break;
    }
    if (hinstLib != nullptr) {
        fftwf_malloc           = (fftwf_malloc_proc)          GetProcAddress(hinstLib, "fftwf_malloc");
        fftwf_free             = (fftwf_free_proc)            GetProcAddress(hinstLib, "fftwf_free");
        fftwf_plan_dft_r2c_2d  = (fftwf_plan_dft_r2c_2d_proc)GetProcAddress(hinstLib, "fftwf_plan_dft_r2c_2d");
        fftwf_plan_dft_c2r_2d  = (fftwf_plan_dft_c2r_2d_proc)GetProcAddress(hinstLib, "fftwf_plan_dft_c2r_2d");
        fftwf_destroy_plan     = (fftwf_destroy_plan_proc)    GetProcAddress(hinstLib, "fftwf_destroy_plan");
        fftwf_execute_dft_r2c  = (fftwf_execute_dft_r2c_proc) GetProcAddress(hinstLib, "fftwf_execute_dft_r2c");
        fftwf_execute_dft_c2r  = (fftwf_execute_dft_c2r_proc) GetProcAddress(hinstLib, "fftwf_execute_dft_c2r");
    }
    if (hinstLib == nullptr || fftwf_malloc == nullptr || fftwf_free == nullptr ||
        fftwf_plan_dft_r2c_2d == nullptr || fftwf_plan_dft_c2r_2d == nullptr ||
        fftwf_destroy_plan == nullptr || fftwf_execute_dft_r2c == nullptr ||
        fftwf_execute_dft_c2r == nullptr) {
        if (hinstLib) FreeLibrary(hinstLib);
        env->ThrowError("DeFreq: cannot load FFTW3 DLL!");
    }

    fft_in  = (float *)fftwf_malloc(sizeof(float) * nx * ny);
    outwidth = nx / 2 + 1;
    fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * ny * outwidth);
    psd     = (float *)calloc(ny * outwidth, sizeof(float));
    if (!fft_in || !fft_out || !psd)
        env->ThrowError("DeFreq: memory allocation failed");

    memset(fft_in, 0, sizeof(float) * nx * ny);

    unsigned planFlags = (measure && !info && !show) ? FFTW_MEASURE : FFTW_ESTIMATE;
    plan_fwd = fftwf_plan_dft_r2c_2d(ny, nx, fft_in, fft_out, planFlags);
    plan_inv = fftwf_plan_dft_c2r_2d(ny, nx, fft_out, fft_in, planFlags);
    if (!plan_fwd || !plan_inv)
        env->ThrowError("DeFreq: FFTW plan creation failed (%dx%d padded to %dx%d)",
            img_width, img_height, nx, ny);

    naverage = 0;
}

// ---------------------------------------------------------------------------
DeFreq::~DeFreq() {
    if (plan_fwd)   fftwf_destroy_plan(plan_fwd);
    if (plan_inv)   fftwf_destroy_plan(plan_inv);
    if (fft_in)     fftwf_free(fft_in);
    if (fft_out)    fftwf_free(fft_out);
    if (psd)        free(psd);
    if (hinstLib)   FreeLibrary(hinstLib);
}

// ---------------------------------------------------------------------------
void DeFreq::SearchBoxBackground(float *psd_ptr, int ow, int h,
    float fx_, float fy_, float dx_, float dy_, float &background)
{
    float fxmin = max(0.0f, fx_ - dx_);
    float fxmax = min(100.0f, fx_ + dx_);
    float fymin = max(-100.0f, fy_ - dy_);
    float fymax = min(100.0f, fy_ + dy_);
    int ixmin = max(0, (int)(fxmin * ow) / 100);
    int ixmax = min((int)(fxmax * ow) / 100, ow - 1);
    int iymin = (int)(fymin * h) / 200;
    int iymax = (int)(fymax * h) / 200;
    int hmin_idx = (iymin >= 0) ? iymin : iymin + h - 1;
    int hmax_idx = (iymax >= 0) ? iymax : iymax + h - 1;
    hmin_idx = max(0, min(hmin_idx, h - 1));
    hmax_idx = max(0, min(hmax_idx, h - 1));
    ixmin = max(0, min(ixmin, ow - 1));
    ixmax = max(0, min(ixmax, ow - 1));

    int counter = 0; float sum = 0;
    float *p = psd_ptr;
    if (hmin_idx <= hmax_idx) {
        p += ow * hmin_idx;
        for (int w = ixmin; w <= ixmax; w++) { sum += sqrtf(p[w]); counter++; }
        for (int hh = hmin_idx; hh < hmax_idx; hh++) {
            sum += sqrtf(p[ixmin]); sum += sqrtf(p[ixmax]); counter += 2; p += ow;
        }
        for (int w = ixmin; w <= ixmax; w++) { sum += sqrtf(p[w]); counter++; }
    } else {
        for (int hh = 0; hh < hmax_idx && hh < h; hh++) {
            sum += sqrtf(p[ixmin]); sum += sqrtf(p[ixmax]); counter += 2; p += ow;
        }
        for (int w = ixmin; w < ixmax; w++) { sum += sqrtf(p[w]); counter++; }
        p = psd_ptr + ow * hmin_idx;
        for (int w = ixmin; w < ixmax; w++) { sum += sqrtf(p[w]); counter++; }
        for (int hh = hmin_idx; hh < h; hh++) {
            sum += sqrtf(p[ixmin]); sum += sqrtf(p[ixmax]); counter += 2; p += ow;
        }
    }
    background = (counter > 0) ? sum * sum / ((float)counter * counter) : 0;
}

// ---------------------------------------------------------------------------
void DeFreq::SearchPeak(float *psd_ptr, int ow, int height,
    float fx_, float fy_, float dx_, float dy_,
    float *fxPeak, float *fyPeak, float *sharpPeak)
{
    float fxmin = max(0.0f, fx_ - dx_);
    float fxmax = min(100.0f, fx_ + dx_);
    float fymin = max(-100.0f, fy_ - dy_);
    float fymax = min(100.0f, fy_ + dy_);
    int ixmin = max(0, (int)(fxmin * ow) / 100);
    int ixmax = min((int)(fxmax * ow) / 100, ow - 1);
    int iymin = (int)(fymin * height) / 200;
    int iymax = (int)(fymax * height) / 200;

    float fftmax = -1.0f; int hmax = 0, wmax = 0;
    float *p = psd_ptr;
    float fftbackground;
    SearchBoxBackground(psd_ptr, ow, height, fx_, fy_, dx_, dy_, fftbackground);

    for (int h = 0; h < height / 2; h++) {
        if (h >= iymin && h <= iymax) {
            for (int w = ixmin; w <= ixmax; w++) {
                if (p[w] > fftmax) { fftmax = p[w]; hmax = h; wmax = w; }
            }
        }
        p += ow;
    }
    for (int h = height / 2; h < height; h++) {
        if (h >= height + iymin - 1 && h <= height + iymax - 1) {
            for (int w = ixmin; w <= ixmax; w++) {
                if (p[w] > fftmax) { fftmax = p[w]; hmax = h; wmax = w; }
            }
        }
        p += ow;
    }

    *fxPeak = (wmax * 100.0f) / ow;
    *fyPeak = (hmax < height / 2) ? (hmax * 200.0f) / height
                                   : (hmax - height + 1) * 200.0f / height;
    *sharpPeak = (fftbackground > 0) ? fftmax / fftbackground : 1;
}

// ---------------------------------------------------------------------------
void DeFreq::CleanWindow(fftwf_complex *out_ptr, int ow, int height,
    float fx_, float fy_, float dx_, float dy_,
    float fxPeak, float fyPeak, float sharpPeak)
{
    int w_Peak = min((int)(fxPeak * ow) / 100, ow - 1);
    int h_Peak = (fyPeak > 0) ? (int)(fyPeak * height) / 200 : (int)(fyPeak * height) / 200 + height - 1;
    h_Peak = max(0, min(h_Peak, height - 1));

    fftwf_complex *pp = out_ptr + ow * h_Peak;
    float fftmax = pp[w_Peak][0] * pp[w_Peak][0] + pp[w_Peak][1] * pp[w_Peak][1];
    float fftbackground = (sharpPeak > 0) ? fftmax / sharpPeak : 0;

    float fxmin = max(0.0f, fx_ - dx_);  float fxmax = min(100.0f, fx_ + dx_);
    float fymin = max(-100.0f, fy_ - dy_); float fymax = min(100.0f, fy_ + dy_);
    int ixmin = max(0, (int)(fxmin * ow) / 100);
    int ixmax = min((int)(fxmax * ow) / 100, ow - 1);
    int iymin = (int)(fymin * height) / 200;
    int iymax = (int)(fymax * height) / 200;

    fftwf_complex *p = out_ptr;
    for (int h = 0; h < height / 2; h++) {
        if (h >= iymin && h <= iymax) {
            for (int w = ixmin; w <= ixmax; w++) {
                float fftcur = p[w][0] * p[w][0] + p[w][1] * p[w][1];
                if (fftcur > fftbackground) {
                    float f = sqrtf(fftbackground / fftcur);
                    p[w][0] *= f; p[w][1] *= f;
                }
            }
        }
        p += ow;
    }
    for (int h = height / 2; h < height; h++) {
        if (h >= height + iymin - 1 && h <= height + iymax - 1) {
            for (int w = ixmin; w <= ixmax; w++) {
                float fftcur = p[w][0] * p[w][0] + p[w][1] * p[w][1];
                if (fftcur > fftbackground) {
                    float f = sqrtf(fftbackground / fftcur);
                    p[w][0] *= f; p[w][1] *= f;
                }
            }
        }
        p += ow;
    }
}

// ---------------------------------------------------------------------------
void DeFreq::CleanHigh(fftwf_complex *outp, int ow, int height, float cutx_, float cuty_)
{
    float invcutx = 100.0f / (cutx_ * ow);
    float invcuty = 200.0f / (cuty_ * height);
    fftwf_complex *p = outp;
    for (int h = 0; h < height / 2; h++) {
        float fh = (float)h * invcuty; fh *= fh;
        for (int w = 0; w < ow; w++) {
            float fw = (float)w * invcutx; fw *= fw;
            float f = 1.0f / (1.0f + fh + fw);
            p[w][0] *= f; p[w][1] *= f;
        }
        p += ow;
    }
    for (int h = height / 2; h < height; h++) {
        float fh = (float)(height - h - 1) * invcuty; fh *= fh;
        for (int w = 0; w < ow; w++) {
            float fw = (float)w * invcutx; fw *= fw;
            float f = 1.0f / (1.0f + fh + fw);
            p[w][0] *= f; p[w][1] *= f;
        }
        p += ow;
    }
}

// ---------------------------------------------------------------------------
void DeFreq::GetFFT2minmax(float *psd_ptr, int ow, int height, float *fft2min, float *fft2max)
{
    float psdmax = 0; float *p = psd_ptr;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < ow; w++) { if (p[w] > psdmax) psdmax = p[w]; }
        p += ow;
    }
    if (psdmax == 0) psdmax = 1.0f;
    *fft2min = psdmax * 1.0e-13f;
    *fft2max = psdmax;
}

// ---------------------------------------------------------------------------
void DeFreq::DrawSearchBox(float *psd_ptr, int ow, int height,
    float fx_, float fy_, float dx_, float dy_, float fftval)
{
    float fxmin = max(0.0f, fx_ - dx_); float fxmax = min(100.0f, fx_ + dx_);
    float fymin = max(-100.0f, fy_ - dy_); float fymax = min(100.0f, fy_ + dy_);
    int ixmin = max(0, (int)(fxmin * ow) / 100);
    int ixmax = min((int)(fxmax * ow) / 100, ow - 1);
    int hmin_idx = (int)(fymin * height) / 200;
    int hmax_idx = (int)(fymax * height) / 200;
    hmin_idx = (hmin_idx >= 0) ? hmin_idx : hmin_idx + height - 1;
    hmax_idx = (hmax_idx >= 0) ? hmax_idx : hmax_idx + height - 1;
    hmin_idx = max(0, min(hmin_idx, height - 1));
    hmax_idx = max(0, min(hmax_idx, height - 1));

    float *p = psd_ptr;
    if (hmin_idx <= hmax_idx) {
        p += ow * hmin_idx;
        for (int w = ixmin; w <= ixmax; w++) p[w] = fftval;
        for (int h = hmin_idx; h < hmax_idx; h++) { p[ixmin] = fftval; p[ixmax] = fftval; p += ow; }
        for (int w = ixmin; w <= ixmax; w++) p[w] = fftval;
    } else {
        for (int h = 0; h < hmax_idx; h++) { p[ixmin] = fftval; p[ixmax] = fftval; p += ow; }
        for (int w = ixmin; w < ixmax; w++) p[w] = fftval;
        p = psd_ptr + ow * hmin_idx;
        for (int w = ixmin; w < ixmax; w++) p[w] = fftval;
        for (int h = hmin_idx; h < height; h++) { p[ixmin] = fftval; p[ixmax] = fftval; p += ow; }
    }
}

// ---------------------------------------------------------------------------
void DeFreq::FrequencySwitchOn(fftwf_complex *outp, int ow, int height,
    float fx_, float fy_, float setvalue)
{
    int ix = min((int)(fx_ * ow) / 100, ow - 1);
    int iy = (int)(fy_ * height) / 200;
    int h = (iy >= 0) ? iy : iy + height - 1;
    h = max(0, min(h, height - 1));
    outp[ow * h + ix][0] = setvalue;
}

// ---------------------------------------------------------------------------
void DeFreq::ProcessPlanar(uint8_t *srcp0, int src_height, int src_width_pixels, int src_pitch,
    float *fxPeak, float *fyPeak, float *sharpPeak,
    float *fxPeak2, float *fyPeak2, float *sharpPeak2,
    float *fxPeak3, float *fyPeak3, float *sharpPeak3,
    float *fxPeak4, float *fyPeak4, float *sharpPeak4)
{
    float *inp = fft_in;
    fftwf_complex *outp = fft_out;
    int width_half = src_width_pixels / 2;
    uint8_t *srcp = srcp0;
    uint8_t *dstp = srcp0;

    // Zero entire padded buffer, then copy image data in
    memset(fft_in, 0, sizeof(float) * nx * ny);
    for (int h = 0; h < src_height; h++) {
        for (int w = 0; w < src_width_pixels; w++)
            inp[w] = ReadPixel(srcp, w);
        srcp += src_pitch;
        inp += nx;
    }
    inp = fft_in; srcp = srcp0;

    // Forward FFT on padded dimensions
    fftwf_execute_dft_r2c(plan_fwd, fft_in, fft_out);

    // PSD averaging
    if (show == 2) naverage += 1; else naverage = 1;
    float faverage = 1.0f / naverage;
    float *psd_ptr = psd; outp = fft_out;
    for (int h = 0; h < ny; h++) {
        for (int w = 0; w < outwidth; w++)
            psd_ptr[w] = psd_ptr[w] * (1 - faverage) + (outp[w][0] * outp[w][0] + outp[w][1] * outp[w][1]) * faverage;
        psd_ptr += outwidth; outp += outwidth;
    }
    outp = fft_out;

    // Search peaks (on padded FFT dimensions)
    if (fx > 0 || fy != 0)   SearchPeak(psd, outwidth, ny, fx, fy, dx, dy, fxPeak, fyPeak, sharpPeak);
    if (fx2 > 0 || fy2 != 0) SearchPeak(psd, outwidth, ny, fx2, fy2, dx2, dy2, fxPeak2, fyPeak2, sharpPeak2);
    if (fx3 > 0 || fy3 != 0) SearchPeak(psd, outwidth, ny, fx3, fy3, dx3, dy3, fxPeak3, fyPeak3, sharpPeak3);
    if (fx4 > 0 || fy4 != 0) SearchPeak(psd, outwidth, ny, fx4, fy4, dx4, dy4, fxPeak4, fyPeak4, sharpPeak4);

    if (show) {
        float fft2min = 0, fft2max = 0;
        GetFFT2minmax(psd, outwidth, ny, &fft2min, &fft2max);
        if (fx > 0 || fy != 0)   DrawSearchBox(psd, outwidth, ny, fx, fy, dx, dy, fft2max);
        if (fx2 > 0 || fy2 != 0) DrawSearchBox(psd, outwidth, ny, fx2, fy2, dx2, dy2, fft2max);
        if (fx3 > 0 || fy3 != 0) DrawSearchBox(psd, outwidth, ny, fx3, fy3, dx3, dy3, fft2max);
        if (fx4 > 0 || fy4 != 0) DrawSearchBox(psd, outwidth, ny, fx4, fy4, dx4, dy4, fft2max);

        float logmin = logf(fft2min), logmax = logf(fft2max);
        float fac = (float)max_pixel_value + 0.5f;
        if (logmax > logmin) fac /= (logmax - logmin);

        // Show spectrum on left half (only within image bounds)
        psd_ptr = psd + (ny / 2) * outwidth;
        dstp = srcp0;
        for (int h = 0; h < src_height / 2; h++) {
            psd_ptr -= outwidth;
            int wlimit = min(width_half, outwidth);
            for (int w = 0; w < wlimit; w++)
                WritePixelInt(dstp, w, (int)(fac * (logf(psd_ptr[w] + 1e-15f) - logmin)));
            dstp += src_pitch;
        }
        psd_ptr = psd + ny * outwidth;
        for (int h = src_height / 2; h < src_height; h++) {
            psd_ptr -= outwidth;
            int wlimit = min(width_half, outwidth);
            for (int w = 0; w < wlimit; w++)
                WritePixelInt(dstp, w, (int)(fac * (logf(psd_ptr[w] + 1e-15f) - logmin)));
            dstp += src_pitch;
        }

        // Test frequency stripes
        outp = fft_out;
        for (int h = 0; h < ny; h++) {
            for (int w = 0; w < outwidth; w++) { outp[w][0] = 0; outp[w][1] = 0; }
            outp += outwidth;
        }
        outp = fft_out;
        fft_out[0][0] = neutral_value;

        float weight = 0;
        if (fx > 0 || fy != 0)   weight += 1.0f;
        if (fx2 > 0 || fy2 != 0) weight += 1 / 1.4f;
        if (fx3 > 0 || fy3 != 0) weight += 1 / 2.0f;
        if (fx4 > 0 || fy4 != 0) weight += 1 / 2.8f;
        float setvalue = 60.0f / (weight + 0.0001f);
        if (bits_per_sample > 8) setvalue *= (float)(1 << (bits_per_sample - 8));

        if (fx > 0 || fy != 0)   FrequencySwitchOn(outp, outwidth, ny, fx, fy, setvalue);
        if (fx2 > 0 || fy2 != 0) FrequencySwitchOn(outp, outwidth, ny, fx2, fy2, setvalue / 1.4f);
        if (fx3 > 0 || fy3 != 0) FrequencySwitchOn(outp, outwidth, ny, fx3, fy3, setvalue / 2.0f);
        if (fx4 > 0 || fy4 != 0) FrequencySwitchOn(outp, outwidth, ny, fx4, fy4, setvalue / 2.8f);

        fftwf_execute_dft_c2r(plan_inv, fft_out, fft_in);

        inp = fft_in; dstp = srcp0;
        for (int h = 0; h < src_height / 2; h++) {
            for (int w = width_half; w < src_width_pixels; w++)
                WritePixelInt(dstp, w, (int)(inp[w]));
            dstp += src_pitch; inp += nx;
        }

    } else {
        // Work mode
        bool clean = false;
        if (*sharpPeak > sharp)   { CleanWindow(fft_out, outwidth, ny, fx, fy, dx, dy, *fxPeak, *fyPeak, *sharpPeak); clean = true; }
        if (*sharpPeak2 > sharp2) { CleanWindow(fft_out, outwidth, ny, fx2, fy2, dx2, dy2, *fxPeak2, *fyPeak2, *sharpPeak2); clean = true; }
        if (*sharpPeak3 > sharp3) { CleanWindow(fft_out, outwidth, ny, fx3, fy3, dx3, dy3, *fxPeak3, *fyPeak3, *sharpPeak3); clean = true; }
        if (*sharpPeak4 > sharp4) { CleanWindow(fft_out, outwidth, ny, fx4, fy4, dx4, dy4, *fxPeak4, *fyPeak4, *sharpPeak4); clean = true; }
        if (cutx > 0 && cuty > 0) { CleanHigh(fft_out, outwidth, ny, cutx, cuty); clean = true; }

        if (clean) {
            fftwf_execute_dft_c2r(plan_inv, fft_out, fft_in);
            float norm = 1.0f / ((float)nx * ny);
            inp = fft_in; dstp = srcp0;
            for (int h = 0; h < src_height; h++) {
                for (int w = 0; w < src_width_pixels; w++)
                    WritePixelInt(dstp, w, (int)(inp[w] * norm + 0.5f));
                dstp += src_pitch; inp += nx;
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
        int planes_to_clear[2]; int clear_count = 0;
        if (plane != 0) planes_to_clear[clear_count++] = PLANAR_Y;
        if (plane != 1) planes_to_clear[clear_count++] = PLANAR_U;
        if (plane != 2 && clear_count < 2) planes_to_clear[clear_count++] = PLANAR_V;
        for (int i = 0; i < clear_count; i++) {
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

    ProcessPlanar(srcp, src_height, src_width_pixels, src_pitch,
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
