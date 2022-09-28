import scipy
import numpy as np
import matplotlib.pyplot as plt

result = scipy.fromfile(open('../data/fft_result'), dtype=scipy.uint8)
snr_pow = scipy.fromfile(open('../data/fft_snr_pow'), dtype=scipy.float32)
snr_db = scipy.fromfile(open('../data/fft_snr_db'), dtype=scipy.uint8)
signal_pow = scipy.fromfile(open('../data/fft_signal_pow'), dtype=scipy.float32)
noise_pow = scipy.fromfile(open('../data/fft_noise_pow'), dtype=scipy.float32)

num_packets = 100
packet_len = 16
packet_idx = 1
packet_basic_0 = np.array([18, 52, 86, 120], dtype=np.uint8)  # 4 bytes
packet_basic_1 = np.array([33, 67, 101, 135], dtype=np.uint8)
packet_basic_2 = np.array([135, 101, 67, 33], dtype=np.uint8)  # 4 bytes
packet_len4 = np.array([4, 128, 240], dtype=np.uint8)  # 4 bytes
packet_len8 = np.array([8, 128, 144], dtype=np.uint8)  # 8 bytes
packet_len16 = np.array([16, 129, 64], dtype=np.uint8)  # 16byte
packet_len32 = np.array([32, 129, 112], dtype=np.uint8)  # 32byte

packet_basic = [packet_basic_0, packet_basic_1, packet_basic_2]
packet_header = [packet_len4, packet_len8, packet_len16, packet_len32]

packet = packet_header[np.int(np.log2(packet_len)) - 2]
for i in range(int(packet_len/4)):
    packet = np.concatenate((packet, packet_basic[packet_idx]))
error_cnt = []
header_err_cnt = []
i = 0
while i < len(result):
    length = 3 + result[i]
    comp_result = []
    header_result = []
    for j in range(i, i+3):
        header_result.append(bin(packet[j-i] ^ result[j]).count('1'))
    for j in range(i, i+length):
        if (j-i) > 2+packet_len:
            break
        comp_result.append(bin(packet[j-i] ^ result[j]).count('1'))
    print(comp_result)
    # if length-comp_result.count(0) != 0:
    #     error_cnt.append(1)
    error_cnt.append(np.sum(comp_result))
    header_err_cnt.append(np.sum(header_result))
    i += length

print("Bit Error Rate: ", np.sum(error_cnt) / 8 / len(result))
print("Packet Reception Ratio: ", (len(error_cnt)-len(np.nonzero(header_err_cnt)[0]))/num_packets)
print("Frame Reception Ratio: ", (len(error_cnt)-len(np.nonzero(error_cnt)[0]))/num_packets)
print("Average SNR dB: ", np.average(snr_db))
print("Average Signal Power: ", np.average(signal_pow))
print("Average Noise Power:", np.average(noise_pow))
