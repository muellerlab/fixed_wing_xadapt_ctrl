# Builds the `GeneralCode` Python module: pybind11 bindings for GeneralCode's own trirotor
# simulator and controller classes (third_party/GeneralCode submodule; only Common/ and
# Components/ are compiled). The simulation loop itself is written in Python.
cmake_minimum_required(VERSION 3.15)
project(trirotor_sim CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # static/shared libs get linked into a Python .so

set(GENERALCODE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/GeneralCode CACHE PATH "GeneralCode checkout")
get_filename_component(GENERALCODE_DIR ${GENERALCODE_DIR} ABSOLUTE)
if(NOT EXISTS ${GENERALCODE_DIR}/Components/CMakeLists.txt)
  message(FATAL_ERROR "GeneralCode not found at ${GENERALCODE_DIR} (submodule not initialised?)")
endif()

# Variables GeneralCode's own top-level CMakeLists.txt normally provides.
set(COMMON_DIR ${GENERALCODE_DIR}/Common)
set(COMPONENTS_DIR ${GENERALCODE_DIR}/Components)

find_package(Eigen3 REQUIRED)
if(EIGEN3_INCLUDE_DIR)
  include_directories(${EIGEN3_INCLUDE_DIR})
elseif(EIGEN3_INCLUDE_DIRS)
  include_directories(${EIGEN3_INCLUDE_DIRS})
else()
  include_directories(/usr/include/eigen3)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(${COMMON_DIR} ${CMAKE_BINARY_DIR}/Common)
add_subdirectory(${COMPONENTS_DIR} ${CMAKE_BINARY_DIR}/Components)

pybind11_add_module(GeneralCode bindings.cpp)
target_include_directories(GeneralCode PRIVATE ${COMMON_DIR} ${COMPONENTS_DIR})
target_link_libraries(GeneralCode PRIVATE Components Common)
set_target_properties(GeneralCode PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")  # libCommon.so is built as a shared library
