cmake_minimum_required(VERSION 3.15)
project(DeFreqPlus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── AviSynth+ SDK ─────────────────────────────────────────────────────────
# Set this to your AviSynth+ header directory, e.g.:
#   cmake -DAVISYNTH_INCLUDE_DIR="C:/AviSynthPlus/avs_core/include" ..
#
# Or clone the AviSynth+ repo and point to the include directory:
#   git clone https://github.com/AviSynth/AviSynthPlus.git
#   set AVISYNTH_INCLUDE_DIR to AviSynthPlus/avs_core/include
set(AVISYNTH_INCLUDE_DIR "" CACHE PATH "Path to AviSynth+ include directory (containing avisynth.h)")

if(NOT AVISYNTH_INCLUDE_DIR OR NOT EXISTS "${AVISYNTH_INCLUDE_DIR}/avisynth.h")
    message(FATAL_ERROR
        "AVISYNTH_INCLUDE_DIR must point to a directory containing avisynth.h\n"
        "Get headers from: https://github.com/AviSynth/AviSynthPlus/tree/master/avs_core/include\n"
        "Example: cmake -DAVISYNTH_INCLUDE_DIR=C:/AviSynthPlus/avs_core/include ..")
endif()

# ─── Build the DLL ──────────────────────────────────────────────────────────
add_library(DeFreq SHARED
    src/defreq.cpp
    src/fftwlite.h
    src/info.h
)

target_include_directories(DeFreq PRIVATE
    ${AVISYNTH_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# No need to link FFTW3 at build time — we load it dynamically at runtime
# No need to link AviSynth — it's loaded by the host

set_target_properties(DeFreq PROPERTIES
    OUTPUT_NAME "DeFreq"
    SUFFIX ".dll"
    PREFIX ""
)

# ─── Install ────────────────────────────────────────────────────────────────
# Default install to a sensible location
install(TARGETS DeFreq
    RUNTIME DESTINATION plugins64
    LIBRARY DESTINATION plugins64
)
