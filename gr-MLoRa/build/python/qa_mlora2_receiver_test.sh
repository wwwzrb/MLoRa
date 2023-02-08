#!/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir=/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python
export PATH=/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/python:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export PYTHONPATH=/home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/build/swig:$PYTHONPATH
/usr/bin/python2 /home/wangzhe/sjtu/USRP/MLoRa/online/gr-MLoRa/python/qa_mlora2_receiver.py 
