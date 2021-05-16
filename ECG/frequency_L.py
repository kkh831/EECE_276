import numpy as np, matplotlib.pyplot as plt, sys

def plot_fft(y,SAMPLE_RATE):
   fft_y = np.abs(np.fft.fft(np.hanning(len(y))*y )) / len(y); 
   fft_y = fft_y [:int(len(fft_y)/2)+1] 
   fbins = np.arange(len(fft_y)) * (SAMPLE_RATE/2 / len(fft_y)) ; 
   plt.loglog(fbins, fft_y,'bo'); plt.title('frequency spectrum of L');plt.show()
   #plt.loglog()   OR   plt.semilogy()


f= open('left_ecg_intp.txt', 'r');
data_ecg_string = f.read();
f.close;

data_ecg_string_list = data_ecg_string.split("\n")[:-1];
data_ecg_int_list = [int(float((val))) for val in data_ecg_string_list]


Fs=6e6/256; # sample rate
plot_fft(data_ecg_int_list, Fs);
