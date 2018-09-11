
# op has a GPU kernel.
find_package(CUDA 9 REQUIRED)

get_filename_component(P1 ${CUDA_INCLUDE_DIRS} PATH)
get_filename_component(P2 ${P1} PATH)

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

include_directories(SYSTEM
  "${P2}"
  "${TensorFlow_INCLUDE_DIR}"
  )
include_directories(SYSTEM "kernels")

# library target name
set(LIB_SUFFIX k_nearest_neighbor)
set(TFOPS_LIB ${PROJECT_NAME}_${LIB_SUFFIX})
set(TFOPS_LIB_CUDA ${PROJECT_NAME}_${LIB_SUFFIX}_cuda)

set(${TFOPS_LIB}_SOURCES
  k_nearest_neighbor.cc
  k_nearest_neighbor_op.cc
  k_nearest_neighbor_kernel.cc
  )
set(${TFOPS_LIB}_PRIVATE_HEADERS
  k_nearest_neighbor_op.h
  )
# to be installed
set(${TFOPS_LIB}_PUBLIC_HEADERS
  )

# CUDA LIBRARY
set(${TFOPS_LIB}_SOURCES_CUDA
  k_nearest_neighbor_kernel_gpu.cu.cc
  )
set_source_files_properties(
  ${${TFOPS_LIB}_SOURCES_CUDA} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)
cuda_add_library(${TFOPS_LIB_CUDA} SHARED
  ${${TFOPS_LIB}_SOURCES_CUDA}
  )
set_target_properties(${TFOPS_LIB_CUDA} PROPERTIES PREFIX "")

# NON-CUDA LIBRARY
add_library(${TFOPS_LIB} SHARED ${${TFOPS_LIB}_SOURCES})

set_target_properties(${TFOPS_LIB} PROPERTIES PREFIX "")
set_target_properties(${TFOPS_LIB} PROPERTIES COMPILE_FLAGS "-DGOOGLE_CUDA")
target_link_libraries(${TFOPS_LIB} LINK_PUBLIC ${TFOPS_LIB_CUDA} ${TensorFlow_LIBRARY})
# Export as title case common
set_target_properties(${TFOPS_LIB} PROPERTIES EXPORT_NAME ${TFOPS_LIB})
set_target_properties(${TFOPS_LIB} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${TFOPS_LIB} PROPERTIES SOVERSION ${PROJECT_VERSION})
set_target_properties(${TFOPS_LIB} PROPERTIES PUBLIC_HEADER "${${TFOPS_LIB}_PUBLIC_HEADERS}")

target_include_directories(${TFOPS_LIB} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_include_directories(${TFOPS_LIB} SYSTEM PUBLIC
  ${TensorFlow_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(${TFOPS_LIB} PRIVATE ${CXX_CLANG_COMPILE_OPTIONS})
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(${TFOPS_LIB} PRIVATE ${CXX_GCC_COMPILE_OPTIONS})
endif()