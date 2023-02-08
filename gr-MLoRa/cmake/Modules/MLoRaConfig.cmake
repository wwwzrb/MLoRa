INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_MLORA MLoRa)

FIND_PATH(
    MLORA_INCLUDE_DIRS
    NAMES MLoRa/api.h
    HINTS $ENV{MLORA_DIR}/include
        ${PC_MLORA_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    MLORA_LIBRARIES
    NAMES gnuradio-MLoRa
    HINTS $ENV{MLORA_DIR}/lib
        ${PC_MLORA_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MLORA DEFAULT_MSG MLORA_LIBRARIES MLORA_INCLUDE_DIRS)
MARK_AS_ADVANCED(MLORA_LIBRARIES MLORA_INCLUDE_DIRS)

