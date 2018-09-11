# -------------- TENSORFLOPS LIBRARY TARGET ----------------
# library target name
set(LIB_SUFFIX k-nearest-neighbor)
set(TFOPS_LIB ${PROJECT_NAME}_${LIB_SUFFIX})

set(${TFOPS_LIB}_SOURCES
  k_nearest_neighbor_op.cc
  )
set(${TFOPS_LIB}_PRIVATE_HEADERS
  k_nearest_neighbor_op.h
  )
# to be installed
set(${TFOPS_LIB}_PUBLIC_HEADERS
  )

add_library(${TFOPS_LIB} SHARED ${${TFOPS_LIB}_SOURCES} ${${TFOPS_LIB}_PRIVATE_HEADERS})
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
  )

target_link_libraries(${TFOPS_LIB}
  ${TensorFlow_LIBRARIES}
  )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(${TFOPS_LIB} PRIVATE ${CXX_CLANG_COMPILE_OPTIONS})
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(${TFOPS_LIB} PRIVATE ${CXX_GCC_COMPILE_OPTIONS})
endif()
