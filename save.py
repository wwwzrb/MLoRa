import scipy
import scipy.io as sio
import numpy as np

# path_prefix = "../dbgresult/FFT05041548/"
# path_prefix = "../dbgresult/FFT05051921/"
# path_prefix = "../dbgresult/FFT05101610/"
path_prefix = "../dbgresult/FFT05131111/"
exp_prefix = ["DIST/", "PWR/", "SF/", "PWR_LEN/"]

power = 2
sf = 7
length = 16
payload = 'inv'
params = str(power)+"dBm-"+str(length)+"byte/"+payload

exp_idx = 3
packet_idx = 0

packet = scipy.fromfile(path_prefix+exp_prefix[exp_idx]+params+"/data_"+str(packet_idx), scipy.complex64)
packet_shr = scipy.fromfile(path_prefix+exp_prefix[exp_idx]+params+"/data_shr_"+str(packet_idx), scipy.complex64)
packet_num = 100
packets = np.zeros((packet_num, len(packet)), dtype=np.complex)
packets_shr = np.zeros((packet_num, len(packet_shr)), dtype=np.complex)

for i in range(packet_num):
    packet_idx = i
    packet = scipy.fromfile(path_prefix+exp_prefix[exp_idx]+params+"/data_"+str(packet_idx), scipy.complex64)
    packet_shr = scipy.fromfile(path_prefix+exp_prefix[exp_idx]+params+"/data_shr_"+str(packet_idx), scipy.complex64)
    packets[i, :] = packet
    packets_shr[i, :] = packet_shr

file_name = path_prefix+exp_prefix[exp_idx]+str(power)+"dBm-"+str(length)+"byte/data_"\
            + payload + "_sf_" + str(sf) + "_len_" + str(length) + "_pwr_"+str(power)+".mat"

sio.savemat(file_name, {'packets': packets, 'packets_shr': packets_shr})


