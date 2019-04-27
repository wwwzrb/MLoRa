Intuition:

    1. Application
        Uplink: Concurrent Transmission  
        Increase Channel Utilization
        Different Nodes Use the Same Channel and Spreading Factor
        Resolve the pesudo-orthogonal of different spreading factor

        Downlink: MISO
        Construct packet in gateway with specific time offset to enhance throughput and channel utilization.
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

1. Two packet stream decoding. :x:

   How to control LoRa nodes to generate two-packet collision?(sending interval)

2. Collecting data under difference parameters. :x:

   TX pwr: -4~14dBm with step 1 dBm
   
   TXRX dist: 3~15m ?

3. Reimplementing LoRa (de/en)coder in python. :heavy_check_mark:

   The encoder has been reproduced by 10, April, 2019.
   
   The decoder has been reproduced by 12, April, 2019.

4. Collecting packet with different payloads.

   a. Single packet :heavy_check_mark:
   
      0x12345678 
      0x87654321
      0x1234567812345678
      0x8765432187654321
      
   b. Collided packet :x:


5. Synchronization of collided packets.

   a. Preamble Detection. :heavy_check_mark:
   
   b. Fine Sync. :x:
   
6. Testing synchronized packets.

   Integrating collected packets into matrix to test collision decoding int n (200) runs. :heavy_check_mark:

</br>

Outline:

    1. Detect collided packets
       (1) Get the maximum (2*n-1) FFT bin. 
           (For n-packet collision, there exist at most (2*n-1) peaks!)
           (For preamble, there are at most n peaks because preambles are always continuous!)
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

Exp:

1. Conducting EXP under 4 different parameters setting. 

   Params(offset) | 0 | 1 | 2 | 3
   ---- | --- | --- | --- | ---
   time |  0  |  0  |  1  |  1
   pwr  |  0  |  1  |  0  |  1

2. Intuitively control time offsets by adjusting sending interval of two devices.

3. Intuitively control power offsets by setting TX power of two devices.

4. Analyse time offset distribution of collected data.

5. Anslyse power offset distribution of collected data.
   (Estimate RX power by collision-free parts of two packets.)
   


Methods Comparsion under different Sampling Rate-Bandwidth (SR-BW) pairs or Oversampling Rates (OR):

   1. Origin RPP0 LoRa receiver is not stable under OS 2 (1M-500K), stable under OS 4,8.(1M -250/125K).
   2. WZ LoRaBee receiver and FFT receiver can work well under OS 2,4,8.
   3. Both RPP0 and WZ cannot pass sync detection under OS 1 (500K-500K).<br/>
      RPP0 LoRa: Decoded chirp symbol is not accurate; Preamble correlation succeeds. (**Improve accuracy!**)<br/>
      WZ: Preabmle detection fails. (**Find reason!**)<br/>


    
    
