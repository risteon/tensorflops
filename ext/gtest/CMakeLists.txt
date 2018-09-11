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

cmake_minimum_required(VERSION 2.8.8)

find_package(Threads REQUIRED)

include(ExternalProject)

set(GTEST_FORCE_SHARED_CRT ON)
set(GTEST_DISABLE_PTHREADS OFF)

if(MINGW)
  set(GTEST_DISABLE_PTHREADS ON)
endif()

ExternalProject_Add(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
  CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
  -DCMAKE_CXX_FLAGS=${MSVC_COMPILER_DEFS}
  -Dgtest_force_shared_crt=${GTEST_FORCE_SHARED_CRT}
  -Dgtest_disable_pthreads=${GTEST_DISABLE_PTHREADS}
  -DBUILD_GTEST=ON
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
  # Disable install step
  INSTALL_COMMAND ""
  )

# Get GTEST source and binary directories
ExternalProject_Get_Property(googletest source_dir binary_dir)

# this workaround is necessary because of https://gitlab.kitware.com/cmake/cmake/issues/15052
file(MAKE_DIRECTORY "${source_dir}/googletest/include")
file(MAKE_DIRECTORY "${source_dir}/googlemock/include")

add_library(libgtest IMPORTED SHARED GLOBAL)
add_dependencies(libgtest googletest)
set_target_properties(libgtest PROPERTIES
  "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
  "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
  "INTERFACE_INCLUDE_DIRECTORIES" "${source_dir}/googletest/include"
  )

add_library(libgmock IMPORTED SHARED GLOBAL)
add_dependencies(libgmock googletest)
set_target_properties(libgmock PROPERTIES
  "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
  "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
  "INTERFACE_INCLUDE_DIRECTORIES" "${source_dir}/googlemock/include"
  )
