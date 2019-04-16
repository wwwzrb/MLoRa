 Implementation Settings:

    1. Setting works well when sampling rate > bandwidth

    #define FFT_REQUIRED_PREAMBLE_CHIRPS   6   // Required preamble chirps.
    #define FFT_PREAMBLE_TOLERANCE         2   // Error tolerance of preamble sync.
    #define FFT_SYNC_TOLERANCE             2   // Error tolerance after preamble sync.
    #define FFT_SYNC_ATTEMPTS              4   // Attempts to detect first syncword.
    #define FFT_REQUIRED_SYNC_WORDS        2   // Required sync words in first detection.
    #define FFT_REQUIRED_SFD_CHIRPS        2   // Required chirps in fine synchronization.
    #define FFT_SFD_COARSE                 8   // To detect first syncword.
    #define FFT_SFD_FINE                   4   // To synchronize with syncword.
    #define FFT_SFD_TOTAL                 10   // Total errors tolerance in syncword.
    #define FFT_PWR_QUEUE_SIZE             8   // Power queue length.
