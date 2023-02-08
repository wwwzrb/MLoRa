from loranode import RN2483Controller
from loraconfig import LoRaConfig
import time
import numpy as np
import threading


class SendThread(threading.Thread):

    def __init__(self, thread_id, device, repeats, content, intra_delay, intra_interval):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.device = device
        self.times = repeats
        self.payload = content
        self.intra_delay = intra_delay
        self.intra_interval = intra_interval

    def run(self):
        print("开始发送: " + str(self.thread_id) + str(time.time()))
        self.send_data()
        print("结束发送: " + str(self.thread_id) + str(time.time()))

    def send_data(self):
        for idx in range(self.times):
            time.sleep(self.intra_delay[idx])
            print("Thread {} : {} : {}".format(self.thread_id, idx, str(time.time())))
            self.device.send_p2p(self.payload)
            time.sleep(self.intra_interval[idx])


def transmit_async(devices):
    pre_interval = 0.150
    post_interval = 1.0
    send_interval = 0.1

    # divide sending time of single packet 0.08s into sections
    section_num = 36
    least_delay = 0.036 / section_num
    section_cnts = []
    intra_delays = [[], []]
    intra_intervals = [[], []]

    basic_seq_0 = "12345678"
    basic_seq_1 = "21436587"
    basic_seq_2 = "87654321"
    basic_seq = [basic_seq_0, basic_seq_1, basic_seq_2]

    seq_len = 8
    seq_idx = [0, 1]
    seq = ['', '']

    for idx in range(seq_len // 4):
        for jdx in range(len(seq_idx)):
            seq[jdx] += basic_seq[seq_idx[jdx]]

    for idx in range(len(seq_idx)):
        print('Payload {} : {}'.format(idx, seq[idx]))

    times = 10
    for idx in range(times):
        intra_delays[0].append(0)
        section_cnt = np.random.randint(12, high=section_num + 1)
        section_cnts.append(section_cnt)
        intra_delays[1].append(least_delay * section_cnt)
        # intra_delays[1].append(0)

        intra_intervals[0].append(0.1)
        intra_intervals[1].append(0.1 - intra_delays[1][idx])

    print('Sections_cnts  : {}'.format(section_cnts))
    print('Intra_delays   : {}'.format(intra_delays))
    print('Intra_intervals: {}'.format(intra_intervals))

    send_thread0 = SendThread(0, devices[0], times, seq[0], intra_delays[0], intra_intervals[0])
    send_thread1 = SendThread(1, devices[1], times, seq[1], intra_delays[1], intra_intervals[1])

    send_thread0.start()
    send_thread1.start()
    send_thread0.join()
    send_thread1.join()
    print("退出主线程")


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
transmit_async(device_list)
print("--- %s seconds ---" % (time.time() - start_time))
