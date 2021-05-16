#python waveform_wav.py 
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt
import sounddevice as sd

samplerate, data = sio.wavfile.read('domisol.wav', 'rb')
samplerate2, data2 = sio.wavfile.read('domisol_record.wav', 'rb')

times = np.arange(len(data))/float(samplerate)
times2 = np.arange(len(data2))/float(samplerate2)

plt.subplot(2,1,1)
plt.plot(times,data)
plt.xlabel('time (s)')
plt.ylabel('amplitude')

plt.subplot(2,1,2)
plt.plot(times2,data2)
plt.xlabel('time (s)')
plt.ylabel('amplitude')

plt.show()

