from scipy import signal
import numpy as np, matplotlib.pyplot as plt, sys

Fs=6e6/256

def plot_fft1(y,SAMPLE_RATE):
   fft_y = np.abs(np.fft.fft(np.hanning(len(y))*y )) / len(y); 
   fft_y = fft_y [:int(len(fft_y)/2)+1] 
   fbins = np.arange(len(fft_y)) * (SAMPLE_RATE/2 / len(fft_y)) ; 
   plt.subplot(2,1,2); plt.loglog(fbins, fft_y);

def plot_fft2(y,SAMPLE_RATE):
   fft_y = np.abs(np.fft.fft(np.hanning(len(y))*y )) / len(y); 
   fft_y = fft_y [:int(len(fft_y)/2)+1] 
   fbins = np.arange(len(fft_y)) * (SAMPLE_RATE/2 / len(fft_y)) ; 
   plt.subplot(2,1,2); plt.loglog(fbins, fft_y);
   plt.title('frequency spectrum / blue: before, orange: after Chebyshev LPF');

f= open('left_ecg_intp.txt', 'r');
data_ecg_string = f.read();
f.close;

data_ecg_string_list = data_ecg_string.split("\n")[:-1];
data_ecg_int_list = [int(float((val))) for val in data_ecg_string_list]

sos = signal.cheby1(5, 0.5, 26*2/Fs, 'low', output='sos')
filtered_out_waveform=signal.sosfilt(sos, data_ecg_int_list)

plt.figure(1); plt.subplot(2,1,1); plt.plot(data_ecg_int_list);
plt.subplot(2,1,1); plt.plot(filtered_out_waveform); plt.title('blue: before, orange: after Chebyshev LPF');

plot_fft1(data_ecg_int_list, Fs);
plot_fft2(filtered_out_waveform, Fs);

plt.show();
