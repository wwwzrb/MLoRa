Intuition:

    1. Application
        Uplink: Concurrent Transmission  
        Increase Channel Utilization
        Different Nodes Use the Same Channel and Spreading Factor

        Downlink: MISO
        Utilizing LoRa Encoding to Recover Message
        Add WIFI MIMO Coding Mechanism?

        Detection Energy of different LoRa Nodes and Antennas
    
    2. Different Scenarios.
       (1) pwr offset != 0 
           a. time offset = 0  pwr difference    
           b. time offset !=0  pwr + time difference
       (2) pwr offset =0
           a. time offset = 0  encoding
           b. time offset !=0  time difference
    
</br>

TODO:

    1. Two packet stream decoding.
       How to control LoRa nodes to generate two-packet collision?(sending interval)
    2. Collecting data under difference parameters.
       TX pwr: -4~14dBm with step 1 dBm
       TXRX dist: 3~15m ?
    3. Reimplementing LoRa (de/en)coder in python.
       The encoder has been reproduced by 10, April, 2019.
       The decoder has been reproduced by 12, April, 2019.
    ４．Collecting packet with different payloads.
    ５．Synchronization of collided packets.
    ６．Testing synchronized packets.
    　　Integrating collected packets into matrix to test collision decoding int n (200) runs. 

</br>

Outline:

    1. Detect collided packets
       (1) Get the maximum (2*n-1) FFT bin.
       (2) Detect consecutive max FFT bin index of the same power level.
           a. Only one preamble w/o packet-level time offset
           b. Two preambles w/ packet-level time offset
              i.  One preamble is detected first, continues detecting in the decoding process. 
              ii. Two or preambles are detected, continues detecting in the decoding process.
    2. w/o packet-level time offset
       (1) Payload will generate two peaks
       (2) Mapping peaks to packet according to pwr offset
       (3) Abandon error packet according to FEC and CRC (too small power offset)
    3. w/ packet-level time offset
       (1) w/  chirp-level time offset
           a. Align by A's or B's chirp
           b. Each collided chirp generates three peaks
           c. Mapping peaks into packets according to time offset and power offset
       (2) w/o chirp-level time offset
           a. Two preambles will be detected
           b. Mapping peaks to packet according to pwr offset
           c. Abandon error packet according to FEC and CRC (too small power offset)


Methods Comparsion under different Sampling Rate-Bandwidth (SR-BW) pairs or Oversampling Rates (OR):

   1. Origin RPP0 LoRa receiver is not stable under OS 2 (1M-500K), stable under OS 4,8.(1M -250/125K).
   2. WZ LoRaBee receiver and FFT receiver can work well under OS 2,4,8.
   3. Both RPP0 and WZ cannot pass sync detection under OS 1 (500K-500K).<br/>
      RPP0 LoRa: Decoded chirp symbol is not accurate; Preamble correlation succeeds. (**Improve accuracy!**)<br/>
      WZ: Preabmle detection fails. (**Find reason!**)<br/>


    
    
