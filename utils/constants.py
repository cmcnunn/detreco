import numpy as np 

ADC_TO_MV = 0.222
PITCH = 0.6
NS_MM = 0.2
LG_THRESHOLD = 229
HG_THRESHOLD = 4000
VETO_THRESHOLD = -1000

# Position i in the reindexed HG array is read from raw channel X_MAPPING[i],
# so reversing the array's order (not its values) mirrors the physical X
# axis to match the silicon tracker's X direction (confirmed against tracker
# data in run 1771).
X_MAPPING = np.array([
    63,55,47,39,31,23,15,7,
    3,11,19,27,35,43,51,59,
    61,53,45,37,29,21,13,5,
    1,9,17,25,33,41,49,57,
    62,54,46,38,30,22,14,6,
    2,10,18,26,34,42,50,58,
    60,52,44,36,28,20,12,4,
    0,8,16,24,32,40,48,56
], dtype=np.int64)[::-1]

Y_MAPPING = np.array([
    7,15,23,31,39,47,55,63,
    59,51,43,35,27,19,11,3,
    5,13,21,29,37,45,53,61,
    57,49,41,33,25,17,9,1,
    6,14,22,30,38,46,54,62,
    58,50,42,34,26,18,10,2,
    4,12,20,28,36,44,52,60,
    56,48,40,32,24,16,8,0
], dtype=np.int64)

