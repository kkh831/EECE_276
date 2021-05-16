#python waveform.py
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import time,sys

f1 = open("1000iii.dat", "rb")  # reopen the file
y = np.fromfile(f1, dtype=np.uint8)
y = y - np.mean(y)
fs = 23437.5;

print('y:',np.shape(y), np.min(y), np.max(y))
plt.plot(y)
plt.show()
