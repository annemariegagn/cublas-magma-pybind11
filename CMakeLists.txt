

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#*************************************************
# Compiling Pybind11 + cublas + MKL + Magma
#*************************************************
# Requirements:
# Installation Cuda toolkit8.0 (including cublas), Magma2.3.0 (http://icl.cs.utk.edu/magma/software/index.html)
# IntelMKL and depedencies (https://software.intel.com/en-us/mkl), Pybind11 (https://github.com/pybind/pybind11)
# Eigen3.3.4 (http://eigen.tuxfamily.org/index.php?title=Main_Page)
#*************************************************
# Create Makefile: cmake -DMyProjectRoot=/home/local/USHERBROOKE/gaga2313/src/pybind11/include/ (ABSOLUTE PATH PYBIND HEADERS)
# If Magma not found: export MAGMA_ROOT=/usr/local/magma (PATH TO MAGMA INSTALL)
# .cu and .cpp in /src folder
# Depending on Cuda architecture, nvcc compiler options (-arch and -code) may need to be changed.
# Anne-Marie Gagnon
#*************************************************


#add_definitions(-DEIGEN_USE_MKL_ALL)
add_definitions(-DEIGEN_USE_MAGMA_ALL)
#add_definitions(-DEIGEN_USE_MAGMA_AND_MKL)

#*************************************************
# Project source directories
#*************************************************

include_directories(
    "${gpu_library_SOURCE_DIR}/src/"
)

#*************************************************
# Libraries
#*************************************************

include_directories( ${MyProjectRoot} )

# Setup submodularity project
list(APPEND CMAKE_MODULE_PATH "${gpu_library_SOURCE_DIR}/cmake")

## Eigen
include_directories($ENV{EIGEN3_INCLUDE_DIR})
message(STATUS
    "EIGEN was found:\n"
    "  EIGEN3_INCLUDE_DIRS: $ENV{EIGEN3_INCLUDE_DIR}"
)

## CUDA
find_package(CUDA)
if (CUDA_FOUND)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_CUT_INCLUDE_DIR}
        ${CUDA_CUT_LIBRARIES}
    )
endif(CUDA_FOUND)

## PYBIND11
find_package(PythonLibs 2.7 REQUIRED)

## MAGMA
#find_package(MAGMA)

if( NOT MAGMA_ROOT AND NOT $ENV{MAGMA_ROOT} STREQUAL "" )
    set( MAGMA_ROOT $ENV{MAGMA_ROOT} )
    # set library directories
    set(MAGMA_LIBRARY_DIRS ${MAGMA_ROOT}/lib)
    # set include directories
    set(MAGMA_INCLUDE_DIRS ${MAGMA_ROOT}/include)
    # set libraries
    find_library(
        MAGMA_LIBRARIES
        NAMES "magma"
        PATHS ${MAGMA_ROOT}
        PATH_SUFFIXES "lib"
        NO_DEFAULT_PATH
    )
    set(MAGMA_FOUND TRUE)
else()
    set(MAGMA_FOUND FALSE)
endif()

if (MAGMA_FOUND)
    add_definitions(-DADD_)
    add_definitions(-DMAGMA_SETAFFINITY)
    add_definitions(-DMAGMA_WITH_MKL)
    include_directories(${MAGMA_INCLUDE_DIRS})
    message(STATUS
        "MAGMA was found:\n"
        "  MAGMA_INCLUDE_DIRS: ${MAGMA_INCLUDE_DIRS}\n"
        "  MAGMA_LIBRARY_DIRS: ${MAGMA_LIBRARY_DIRS}\n"
        "  MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}"
    )
else()
    message(FATAL_ERROR "MAGMA was not found, please install it first")
endif()

if (MINGW)
  set(gtest_disable_pthreads ON CACHE INTERNAL "" FORCE)
endif()

## MKL
# CMake script to detect Intel(R) Math Kernel Library (MKL)

set(_MKL_IA32 FALSE)
set(_MKL_INTEL64 FALSE)
if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(_MKL_IA32 TRUE)
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_MKL_INTEL64 TRUE)
else()
    message(FATAL_ERROR "Unsupported 'void *' size (${SIZEOF_VOID_P})")
endif()

# Versions should be listed is decreasing order of preference
set(_MKL_TEST_VERSIONS ${MKL_ADDITIONAL_VERSIONS}
    "2011"
    # alternative form: "2011.xxx.y"
    # (y is the release-update number and xxx is the package number)
)

if (MKL_FIND_VERSION AND NOT MKL_FIND_QUIETLY)
    message(WARNING "Requesting a specific version of Intel(R) MKL is not supported")
endif()

# Use environment variables from Intel build scripts, if available
if (NOT MKL_ROOT AND NOT $ENV{MKLROOT} STREQUAL "")
  set(MKL_ROOT $ENV{MKLROOT})
endif()

if (MKL_ROOT)
  file(TO_CMAKE_PATH ${MKL_ROOT} MKL_ROOT)
endif()

if (NOT INTEL_ROOT AND NOT $ENV{INTELROOT} STREQUAL "")
  set(INTEL_ROOT $ENV{INTELROOT})
endif()

if (INTEL_ROOT)
  file(TO_CMAKE_PATH ${INTEL_ROOT} INTEL_ROOT)
endif()

if (MKL_FIND_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "_MKL_TEST_VERSIONS = ${_MKL_TEST_VERSIONS}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "MKL_ADDITIONAL_VERSIONS = ${MKL_ADDITIONAL_VERSIONS}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "MKL_USE_STATIC_LIBS = ${MKL_USE_STATIC_LIBS}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "MKL_ROOT = ${MKL_ROOT}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "INTEL_ROOT = ${INTEL_ROOT}")
endif()

# Find MKL include directory

set(_MKL_ROOT_SEARCH_DIRS
  ${MKL_ROOT}
)

foreach (_MKL_VER ${_MKL_TEST_VERSIONS})
    if (WIN32)
        list(APPEND _MKL_ROOT_SEARCH_DIRS "$ENV{ProgramFiles}/Intel/Composer XE/mkl")
    else()
        list(APPEND _MKL_ROOT_SEARCH_DIRS "/opt/intel/composerxe-${_MKL_VER}/mkl")
    endif()
endforeach()

if (MKL_FIND_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "_MKL_ROOT_SEARCH_DIRS = ${_MKL_ROOT_SEARCH_DIRS}")
endif()

find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS ${_MKL_ROOT_SEARCH_DIRS}
    PATH_SUFFIXES include
    DOC "The path to Intel(R) MKL header files"
)

if (MKL_INCLUDE_DIR)
    if (MKL_FIND_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "location of mkl.h: ${MKL_INCLUDE_DIR}/mkl.h")
    endif()
else()
    if (MKL_FIND_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "unable to find Intel(R) MKL header files. Please set MKLROOT"
                       " to the root directory containing MKL.")
    endif()
endif()

# Find MKL library directory

set(_INTEL_LIBRARY_DIR_SUFFIXES "lib")
if (_MKL_IA32)
    list(APPEND _INTEL_LIBRARY_DIR_SUFFIXES "lib/ia32")
elseif (_MKL_INTEL64)
    list(APPEND _INTEL_LIBRARY_DIR_SUFFIXES "lib/intel64")
else()
    message(FATAL_ERROR "unreachable")
endif()

set(_MKL_LIBRARY_SEARCH_DIRS ${_MKL_ROOT_SEARCH_DIRS})
if (MKL_INCLUDE_DIR)
    list(APPEND _MKL_LIBRARY_SEARCH_DIRS "${MKL_INCLUDE_DIR}/..")
endif()

if (MKL_FIND_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "_INTEL_LIBRARY_DIR_SUFFIXES = ${_INTEL_LIBRARY_DIR_SUFFIXES}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "_MKL_LIBRARY_SEARCH_DIRS = ${_MKL_LIBRARY_SEARCH_DIRS}")
endif()

set(MKL_LIB_PREFIX "mkl_")
if (MKL_USE_STATIC_LIBS)
    if (_MKL_IA32)
        if (WIN32)
            set(_MKL_LIBRARIES intel_c)
        else()
            set(_MKL_LIBRARIES intel)
        endif()
    elseif (_MKL_INTEL64)
        set(_MKL_LIBRARIES intel_lp64)
    else()
        message(FATAL_ERROR "unreachable")
    endif()

    list(APPEND _MKL_LIBRARIES intel_thread)
    list(APPEND _MKL_LIBRARIES core)
else()
    set(_MKL_LIBRARIES rt)
endif()

set(_MKL_MISSING_LIBRARIES "")
set(MKL_LIBRARIES "")
set(MKL_LIBRARY_DIRS "")
# Find MKL libraries
foreach (_MKL_LIB_RAW ${_MKL_LIBRARIES})
    set(_MKL_LIB ${MKL_LIB_PREFIX}${_MKL_LIB_RAW})
    string(TOUPPER ${_MKL_LIB} _MKL_LIB_UPPER)

    find_library(${_MKL_LIB_UPPER}_LIBRARY
        NAMES ${_MKL_LIB}
        PATHS ${_MKL_LIBRARY_SEARCH_DIRS}
        PATH_SUFFIXES ${_INTEL_LIBRARY_DIR_SUFFIXES}
        DOC "The path to Intel(R) MKL ${_MKL_LIB_RAW} library"
    )
    mark_as_advanced(${_MKL_LIB_UPPER}_LIBRARY)

    if (NOT ${_MKL_LIB_UPPER}_LIBRARY)
        list(APPEND _MKL_MISSING_LIBRARIES ${_MKL_LIB})
    else()
        list(APPEND MKL_LIBRARIES ${${_MKL_LIB_UPPER}_LIBRARY})
        if (MKL_FIND_DEBUG)
            message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                           "Found ${_MKL_LIB}: ${${_MKL_LIB_UPPER}_LIBRARY}")
        endif()

        get_filename_component(_MKL_LIB_PATH "${${_MKL_LIB_UPPER}_LIBRARY}" PATH)
        list(APPEND MKL_LIBRARY_DIRS ${_MKL_LIB_PATH})
    endif()
endforeach()

## Find OpenMP, pthread and math libraries

set(_INTEL_LIBRARY_SEARCH_DIRS
  ${INTEL_ROOT}
  ${INTEL_ROOT}/compiler
)

foreach(_MKL_DIR ${_MKL_ROOT_SEARCH_DIRS})
    list(APPEND _INTEL_LIBRARY_SEARCH_DIRS "${_MKL_DIR}/..")
    list(APPEND _INTEL_LIBRARY_SEARCH_DIRS "${_MKL_DIR}/../compiler")
endforeach()

if (MKL_FIND_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "_INTEL_LIBRARY_SEARCH_DIRS = ${_INTEL_LIBRARY_SEARCH_DIRS}")
endif()

if (NOT WIN32)
    find_library(PTHREAD_LIBRARY pthread DOC "Path to POSIX threads library")
endif()

set(_IOMP5_LIB iomp5)
if (WIN32)
  if (MKL_USE_STATIC_LIBS)
      list(APPEND _IOMP5_LIB libiomp5mt.lib)
  else()
      list(APPEND _IOMP5_LIB libiomp5md.lib)
  endif()
endif()

find_library(IOMP5_LIBRARY
    NAMES ${_IOMP5_LIB}
    PATHS ${_INTEL_LIBRARY_SEARCH_DIRS}
    PATH_SUFFIXES ${_INTEL_LIBRARY_DIR_SUFFIXES}
    DOC "Path to OpenMP runtime library"
)

if (NOT IOMP5_LIBRARY)
    # we could instead fallback to default library (via FindOpenMP.cmake)
    list(APPEND _MKL_MISSING_LIBRARIES IOMP5)
else()
    list(APPEND MKL_LIBRARIES ${IOMP5_LIBRARY})
    if (MKL_FIND_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "Found IOMP5_LIBRARY: ${IOMP5_LIBRARY}")
    endif()

    get_filename_component(_MKL_LIB_PATH "${IOMP5_LIBRARY}" PATH)
    list(APPEND MKL_LIBRARY_DIRS ${_MKL_LIB_PATH})
endif()

# Optimized math library (optional)
set(_MATH_LIB imf)  # linked by default with Intel compiler
if (WIN32)
  if (MKL_USE_STATIC_LIBS)
    list(APPEND _MATH_LIB libmmds.lib)  # assumes (/MD) otherwise libmmt.lib (for /MT)
  else()
      list(APPEND _MATH_LIB libmmd.lib)
  endif()
endif()

find_library(MATH_LIBRARY
    NAMES ${_MATH_LIB}
    PATHS ${_INTEL_LIBRARY_SEARCH_DIRS}
    PATH_SUFFIXES ${_INTEL_LIBRARY_DIR_SUFFIXES}
    DOC "Path to optimized math library"
)

if (NOT MATH_LIBRARY)
    # we could instead fallback to default library (via FindOpenMP.cmake)
    list(APPEND _MKL_MISSING_LIBRARIES MATH)
else()
    list(APPEND MKL_LIBRARIES ${MATH_LIBRARY})
    if (MKL_FIND_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "Found MATH_LIBRARY: ${MATH_LIBRARY}")
    endif()

    get_filename_component(_MKL_LIB_PATH "${MATH_LIBRARY}" PATH)
    list(APPEND MKL_LIBRARY_DIRS ${_MKL_LIB_PATH})
endif()

# Check all required libraries are available
list(REMOVE_DUPLICATES MKL_LIBRARY_DIRS)

set(MKL_INCLUDE_DIRS
    ${MKL_INCLUDE_DIR}
)

set(MKL_FOUND TRUE)
if (NOT MKL_INCLUDE_DIR)
    set(MKL_FOUND FALSE)
    if (MKL_FIND_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "MKL not found - MKL_INCLUDE_DIR was empty")
    endif()
elseif (_MKL_MISSING_LIBRARIES)
    set(MKL_FOUND FALSE)
    if (MKL_FIND_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "MKL not found - the following libraries are missing: "
                       "${_MKL_MISSING_LIBRARIES}")
    endif()
endif()

if (MKL_FOUND)
    if (NOT MKL_FIND_QUIETLY OR MKL_FIND_DEBUG)
        message(STATUS
            "Intel(R) MKL was found:\n"
            "  MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}\n"
            "  MKL_LIBRARY_DIRS: ${MKL_LIBRARY_DIRS}\n"
            "  MKL_LIBRARIES: ${MKL_LIBRARIES}"
        )
    endif()
else()
    if (MKL_FIND_REQUIRED)
        message(SEND_ERROR "Intel(R) MKL could not be found.")
    else()
        message(STATUS "Intel(R) MKL could not be found.")
    endif()
endif()

mark_as_advanced(
    MKL_INCLUDE_DIR
    MKL_INCLUDE_DIRS
    MKL_LIBRARY_DIRS
)

#**************************************************
# End MKL script
#**************************************************

## MKL
#find_package(MKL)
if (MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIRS})
else()
    message (FATAL_ERROR "MKL was not found, please install it first.")
endif()

include_directories(${PYTHON_INCLUDE_DIRS})


#*************************************************
# Targets and compilers options
#*************************************************

add_compile_options( ${CMAKE_CUDA_FLAGS} )

# nvcc compiler options

#Magma do not always support architecture >52
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --debug --device-debug -gencode arch=compute_52,code=compute_52)

# gcc options

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


cuda_add_library(gpu_library SHARED
  src/gpu_library.cpp
  src/gpu_library.cu)

target_link_libraries(gpu_library
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MKL_LIBRARIES}
    ${MAGMA_LIBRARIES}
)

set_target_properties(gpu_library PROPERTIES PREFIX "")

