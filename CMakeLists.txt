cmake_minimum_required(VERSION 3.21)
project(Tucker2EI_MPI)

set(CMAKE_CXX_STANDARD 14)

set(TUCKER_DIR "$ENV{TUCKER_DIR}")
set(TUCKER_BUILD_DIR "$ENV{TUCKER_BUILD_DIR}")
find_package(MPI REQUIRED)
find_package(Boost REQUIRED)
message("${TUCKER_BUILD_DIR}")
if (MPI_FOUND)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    set(MPI_LIBS ${MPI_CXX_LIBRARIES})
endif (MPI_FOUND)

# MKL
if (EXISTS $ENV{CRAY_LIBSCI_DIR})
    message("libsci used for blas/lapack/scalapack")
    include_directories($ENV{CRAY_LIBSCI_PREFIX_DIR}/include)
    link_directories($ENV{CRAY_LIBSCI_PREFIX_DIR}/lib)
    set(LINALG_LIBS sci_intel_mpi_mp)
elseif (EXISTS $ENV{MKLROOT})
    message("MKL Library is used")
    include_directories($ENV{MKL_INCLUDE})
    link_directories($ENV{MKL_LIB})
    set(LINALG_LIBS mkl_intel_thread mkl_core mkl_intel_lp64 iomp5 pthread)
elseif (BLAS_FOUND AND LAPACK_FOUND)
    message("blas/lapack are used")
    set(LINALG_LIBS ${LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else (EXISTS $ENV{CRAY_LIBSCI_DIR})
    message(SEND_ERROR "Cannot find MKL! Please set MKLROOT and configure again.")
endif (EXISTS $ENV{CRAY_LIBSCI_DIR})

# boost
if (Boost_FOUND)
    message("Found Boost version ${Boost_VERSION}")
    include_directories($ENV{BOOST_INC})
    link_directories($ENV{BOOST_LIB})
    set(BOOST_LIBS boost_regex boost_filesystem boost_system)
else (Boost_FOUND)
    message(SEND_ERROR "Cannot find boost! Please load module boost or set BOOST_ROOT and configure again.")
endif (Boost_FOUND)

if (TUCKER_DIR AND EXISTS ${TUCKER_BUILD_DIR}/mpi/libmpi_tucker.a AND EXISTS ${TUCKER_BUILD_DIR}/serial/libserial_tucker.a)
    include_directories(${TUCKER_DIR}/src/mpi)
    include_directories(${TUCKER_DIR}/src/serial)
    link_directories(${TUCKER_BUILD_DIR}/mpi)
    link_directories(${TUCKER_BUILD_DIR}/serial)
    set(TUCKER_LIBS mpi_tucker serial_tucker)
else (TUCKER_DIR AND EXISTS ${TUCKER_BUILD_DIR}/mpi/libmpi_tucker.a AND EXISTS ${TUCKER_BUILD_DIR}/serial/libserial_tucker.a)
    message(SEND_ERROR "Cannot find Tucker library")
endif (TUCKER_DIR AND EXISTS ${TUCKER_BUILD_DIR}/mpi/libmpi_tucker.a AND EXISTS ${TUCKER_BUILD_DIR}/serial/libserial_tucker.a)

set(LIBS ${TUCKER_LIBS} ${MPI_LIBS} ${LINALG_LIBS} ${BOOST_LIBS})

set(SRC src/TensorUtils.cpp src/DataReader.cpp src/QuadRule.cpp src/FE.cpp
        src/ConvolutionComputer.cpp src/BlasWrapper.cpp src/ExceptionHandler.h src/FE3D.cpp src/FourIndexIntegrator.cpp
        src/TwoIndexIntegrator.cpp src/TwoIndexIntegrator.h src/Integrator.cpp src/Integrator.h
        src/BasisSmoothScreener.h src/BasisSmoothScreener.cpp src/Parameters.cpp src/FeCell.cpp src/FeCell.h
        src/FeMap.cpp src/FeMap.h src/FE3D.h src/FourIndexGenerator.cpp
        )

add_executable(Tucker2EI_MPI src/main.cpp ${SRC})
target_link_libraries(Tucker2EI_MPI ${LIBS})

add_executable(convcomp src/Tucker2EIConvParamComputer.cpp ${SRC})
target_link_libraries(convcomp ${LIBS})
