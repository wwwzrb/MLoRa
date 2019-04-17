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

Comparsion under different Sampling Rate-Bandwidth:
    1. Origin RPP0 LoRa receiver is not stable under oversampling rate (OS) 2 (1M-500K), stable under OS 4,8.(1M -250/125K.)
    
