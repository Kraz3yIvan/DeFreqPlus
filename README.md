# DeFreq+ — FFT Interference Frequency Remover for AviSynth+

**64-bit, high bit depth modernization** of the original DeFreq v0.7 plugin by A.G.Balakhnin (Fizick), 2004-2006.

## What's Changed from the Original

- **64-bit native** — builds as a 64-bit DLL for AviSynth+ x64
- **High bit depth support** — works with 8, 10, 12, 14, and 16-bit planar YUV (including YUV444P16)
- **All planar YUV formats** — YV12, YV16, YV24, YUV420P10, YUV422P16, YUV444P16, etc.
- **AviSynth+ API v3** — uses `AvisynthPluginInit3` and `AVS_Linkage`
- **Removed YUY2** — legacy packed format removed; convert to planar first if needed
- **Multi-DLL fallback** — tries `libfftw3f-3.dll`, `fftw3f.dll`, `fftw3.dll` automatically
- **Modern C++17** — clean types, no MSVC6 hacks
- **Same parameter interface** — drop-in replacement; existing scripts work unchanged

## Building

### Prerequisites

1. **Visual Studio 2019/2022** (or MinGW-w64 with CMake)
2. **CMake 3.15+**
3. **AviSynth+ headers** — clone from https://github.com/AviSynth/AviSynthPlus
   - You only need the headers from `avs_core/include/`

### Build Steps (MSVC)

```bat
git clone https://github.com/AviSynth/AviSynthPlus.git

mkdir build && cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DAVISYNTH_INCLUDE_DIR="C:\path\to\AviSynthPlus\avs_core\include"

cmake --build . --config Release
```

### Build Steps (MinGW-w64)

```bash
mkdir build && cd build

cmake .. -G "MinGW Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAVISYNTH_INCLUDE_DIR="/path/to/AviSynthPlus/avs_core/include"

cmake --build .
```

The output `DeFreq.dll` goes into your AviSynth+ `plugins64` folder.

### FFTW3 Runtime Dependency

Download the **64-bit float** FFTW3 binaries from https://www.fftw.org/install/windows.html

Place `libfftw3f-3.dll` in one of:
- Your AviSynth+ `plugins64` folder
- Your system PATH
- The same directory as your AviSynth script

**Important:** You need the **float (single-precision)** version — the file is typically called `libfftw3f-3.dll` (note the **f** for float). The double-precision `libfftw3-3.dll` will NOT work.

## Usage

Usage is identical to the original DeFreq. Example AviSynth+ script:

```avs
# Basic interference removal on Y plane
video = LWLibavVideoSource("my_vhs_capture.mkv")
video = ConvertToYUV444(video, bits=16)   # or keep at source format
DeFreq(video, fx=10.0, fy=-10.0, dx=1.5, dy=2.0, sharp=50.0)
```

```avs
# Show spectrum analysis mode
video = LWLibavVideoSource("my_capture.mkv")
DeFreq(video, fx=10.0, fy=-10.0, show=1)
```

```avs
# Multiple search windows
video = LWLibavVideoSource("my_capture.mkv")
DeFreq(video, \
    fx=10.0, fy=-10.0, dx=1.5, dy=2.0, sharp=50.0, \
    fx2=25.0, fy2=15.0, dx2=1.5, dy2=2.0, sharp2=50.0)
```

## Parameters

All parameters are identical to the original DeFreq v0.7:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| fx | float | 10.0 | Window center frequency X (0-100%) |
| fy | float | -10.0 | Window center frequency Y (-100 to 100%) |
| dx | float | 1.5 | Half-width of search window (%) |
| dy | float | 2.0 | Half-height of search window (%) |
| sharp | float | 50.0 | Peak-to-background threshold |
| fx2-fx4, fy2-fy4 | float | 0 (disabled) | Additional search windows |
| dx2-dx4, dy2-dy4 | float | 1.5/2.0 | Additional window sizes |
| sharp2-sharp4 | float | 50.0 | Additional thresholds |
| cutx | float | 0 (disabled) | Low-pass cutoff X (0-300%) |
| cuty | float | 0 (disabled) | Low-pass cutoff Y (0-300%) |
| plane | int | 0 | Plane to process: 0=Y, 1=U, 2=V |
| show | int | 0 | 0=process, 1=show spectrum, 2=show temporal average |
| info | bool | false | Overlay peak info text (8-bit only) |
| measure | bool | true | Use FFTW_MEASURE for optimal FFT plan |

## Notes

- **Interlaced sources**: Use `SeparateFields()` before DeFreq, then `Weave()` after.
- **Info overlay**: The text OSD (`info=true`) only works with 8-bit formats. For 16-bit, use `show` mode visually or check frame properties.
- **Frequency values**: Independent of crop but must be recalculated after resize (same as original).
- The plugin processes one plane at a time. To clean multiple planes, chain multiple DeFreq calls with different `plane` values.

## License

GNU General Public License v2 or later. See `gpl.txt`.

## Credits

- Original DeFreq: A.G.Balakhnin aka Fizick (2004-2006)
- FFTW3 library: Matteo Frigo and Steven G. Johnson (http://www.fftw.org)
- Font data (info.h): IT0051 by thejam79, YV12 mode by minamina
