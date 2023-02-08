from loranode import RN2483Controller
from loraconfig import LoRaConfig
import collections
import time
import numpy as np

Test = collections.namedtuple('Test', ['payload', 'times'])


def transmit_sync(devices):
    pre_interval = 0.150
    post_interval = 1.0
    send_interval = 0.1

    # divide sending time of single packet 0.08s into sections
    section_num = 40
    least_delay = 0.04/section_num
    section_cnts = []
    intra_delays = []
    intra_intervals = []

    basic_seq_0 = "12345678"
    basic_seq_1 = "21436587"
    basic_seq_2 = "87654321"
    basic_seq = [basic_seq_0, basic_seq_1, basic_seq_2]

    seq_len = 8
    seq_idx = [0, 1]
    seq = ['', '']

    for idx in range(seq_len//4):
        for jdx in range(len(seq_idx)):
            seq[jdx] += basic_seq[jdx]

    # seq = "12345678"  # 4 Byte
    # seq = "87654321"  # 4 Byte
    # seq = "1234567812345678"  # 8 Byte
    # seq = "12345678123456781234567812345678"  # 16 Byte
    test = Test(payload=seq, times=10)

    # time.sleep(pre_delay)
    for idx in range(len(seq_idx)):
        print('Payload {} : {}'.format(idx, test.payload[idx]))

    time_stamps = []
    for idx in range(0, test.times):
        # device 0 send
        bgn_time = time.time()
        # print('Start Packet {}.{} time : {}'.format(idx, 0, bgn_time))
        devices[0].send_p2p(test.payload[0])
        end_time = time.time()
        # print('End   Packet {}.{} time : {}'.format(idx, 1, end_time))
        time_stamps.append([bgn_time, end_time, end_time-bgn_time])

        # device 1 send after intra_delay
        bgn_time = end_time
        section_cnt = np.random.randint(1, high=section_num + 1)
        # intra_delay = least_delay * section_cnt
        intra_delay = 0
        time.sleep(intra_delay)
        section_cnts.append(section_cnt)
        devices[1].send_p2p(test.payload[1])
        end_time = time.time()
        # print('Intra_delay: {}'.format(intra_delay))
        intra_delays.append(intra_delay)
        time_stamps.append([bgn_time, end_time, end_time-bgn_time])

        # wait for fixed interval
        intra_interval = send_interval - intra_delay
        time.sleep(intra_interval)
        # print('Intra_interval: {}'.format(intra_interval))
        intra_intervals.append(intra_interval)

    print('Sections_cnts  : {}'.format(section_cnts))
    print('Intra_delays   : {}'.format(intra_delays))
    print('Intra_intervals: {}'.format(intra_intervals))
    # print('Time_stamps    : {}'.format(time_stamps))

    # time.sleep(post_delay)


# Sending time of 100 packets
test_sending_wod = [8.10, 10.17]
test_sending_wd_8 = [16.11, 18.19]
test_sending_wd_10 = [18.11, 20.18]

# Initialize lora configuration
config = LoRaConfig(freq=868.1e6, sf=7, cr="4/8", bw=125e3, prlen=8, crc=False, implicit=False)
print(config.string_repr())

# Configure transmitter
# c = RN2483Controller("/dev/ttyACM1")
device_list = [RN2483Controller("/dev/ttyACM0"), RN2483Controller("/dev/ttyACM1")]
powers = [2, 0]
nodes = [0, 1]

for i in range(len(device_list)):
    try:
        c = device_list[i]
        # self.lc.set_freq(config.freq)
        print("Device {}: Node {}".format(i, nodes[i]))
        c.set_sf(config.sf)
        c.set_cr(config.cr)
        c.set_bw(int(config.bw // 1e3))
        c.set_prlen(str(config.prlen))
        c.set_crc("on" if config.crc else "off")
        # c.set_implicit("on" if config.implicit else "off")
        c.set_sync('00')
        c.set_pwr(powers[i])
        print("Pre length: " + c.get_prlen())
        print("Bandwidth : " + c.get_bw())
        print("Sync words: " + c.get_sync())
        print("Power     : " + c.get_pwr())
    except Exception as e:
        print(e)
        exit(1)

start_time = time.time()
transmit_sync(device_list)
print("--- %s seconds ---" % (time.time() - start_time))
