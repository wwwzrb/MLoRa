# Install script for directory: /home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/MLoRa" TYPE FILE FILES
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python/__init__.py"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python/Stack.py"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python/lora_decode.py"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python/utility.py"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python/mlora2_receiver.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/MLoRa" TYPE FILE FILES
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/__init__.pyc"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/Stack.pyc"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/lora_decode.pyc"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/utility.pyc"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/mlora2_receiver.pyc"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/__init__.pyo"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/Stack.pyo"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/lora_decode.pyo"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/utility.pyo"
    "/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python/mlora2_receiver.pyo"
    )
endif()

