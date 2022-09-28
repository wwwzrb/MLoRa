import numpy as np
# curr_peaks = {}
# print(len(curr_peaks))
#
# while len(curr_peaks) < 3:
#     curr_peaks.update({-(len(curr_peaks)+1):-(len(curr_peaks)+1)})
#
# print(curr_peaks)

# detect_indexes = np.array([0, -1, -1])
# detect_indexes.sort()
# repeat_index = np.where(detect_indexes >= 0)[0]

# detect_indexes = [0, -1, -1]
# detect_dict = {0: 20, -1: 23, -1: 3}
# detect_dict = np.array(sorted(detect_dict.items(), key=lambda x: x[1]))
# print(detect_dict[::-1][:, 0])
# detect_indexes.sort()
# repeat_index = np.where(detect_indexes != -1)[0]
# print(detect_dict[repeat_index[0]][1])
#
# preamble_indexes = [pre_dict[0] for pre_dict in detect_dict]
# preamble_bins = [pre_dict[1] for pre_dict in detect_dict]

# chirp_offset = [13, 26]
# samp_offset = [32, 64]
#
# time_offset =np.multiply(np.array(samp_offset) , 8) + np.multiply(np.array(chirp_offset) , 1024)
# print(len(time_offset))

sfd_bins = [39, 36, 26]

sfd_bins_dict = {0: sfd_bins[0], 1: sfd_bins[1], 2: sfd_bins[2]}
sfd_bins_dict = np.array(sorted(sfd_bins_dict.items(), key=lambda x: x[1], reverse=True))
sfd_bin_order = [0, 1, 2]
for idx in range(len(sfd_bins)):
    sfd_bin_order[sfd_bins_dict[idx][0]] = idx