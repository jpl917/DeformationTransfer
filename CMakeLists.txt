cmake_minimum_required(VERSION 2.8)
project(deformation)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-03")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
   message("OPENMP FOUND")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


find_package(SuiteSparse QUIET NO_MODULE)  # 1st: Try to locate the *config.cmake file.
if(NOT SuiteSparse_FOUND)
        #set(SuiteSparse_VERBOSE ON)
        find_package(SuiteSparse REQUIRED) # 2nd: Use FindSuiteSparse.cmake module
        include_directories(${SuiteSparse_INCLUDE_DIRS})
else()
        message(STATUS "Find SuiteSparse : include(${USE_SuiteSparse})")
        include(${USE_SuiteSparse})
endif()
MESSAGE(STATUS "SuiteSparse_LIBS: ${SuiteSparse_LIBRARIES}")

include_directories("/usr/include/eigen3")

set(Trimesh2_DIR "/home/jpl/Programs/trimesh2")
set(Trimesh2_INCLUDE_DIRS "/home/jpl/Programs/trimesh2/include/")
set(Trimesh2_LIB_DIRS "/home/jpl/Programs/trimesh2/lib.Linux64/libtrimesh.a")
include_directories(${Trimesh2_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${Trimesh2_LIB_DIRS} ${SuiteSparse_LIBRARIES})
