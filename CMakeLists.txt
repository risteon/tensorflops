# MIT License
#
# Copyright 2018 Christoph Rist
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## CMake build file for the tensorflops repository.
##

# allow for project VERSION option
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

project(tfops VERSION 0.1 LANGUAGES CXX)
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run "
  "CMake from there. You may need to remove CMakeCache.txt. ")
endif()


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CXX_CLANG_COMPILE_OPTIONS
  "-march=native"
  "-Weverything"
  "-Wno-c++98-compat"
  "-Wno-c++98-c++11-c++14-compat"
  )
set(CXX_GCC_COMPILE_OPTIONS
  "-march=native"
  "-Wall"
  "-Wno-unknown-pragmas"
  "-D_GLIBCXX_USE_CXX11_ABI=1"
  )
set(CUDA_CXX_COMPILE_OPTIONS
  "-march=native"
  "-std=c++11"
  )

# -------- FIND DEPENDENCIES WITH CMAKE CONVENTIONS --------
# -------- --------- REQUIRED DEPENDENCIES -------- --------

# TENSORFLOW
set(TENSORFLOW_LOCATION "" CACHE FILEPATH "Install location of tensorflow (python site-packages)")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(TensorFlow REQUIRED)


# ------------------ OPTIONAL DEPENDENCIES -----------------
# find_package(Eigen3 3.3 NO_MODULE)


# ------------- tensorflops LIBRARY TARGETS ----------------
add_subdirectory(src/k_nearest_neighbor)


# ------------------------- TEST ---------------------------
#enable_testing()
#add_subdirectory(ext/gtest)
