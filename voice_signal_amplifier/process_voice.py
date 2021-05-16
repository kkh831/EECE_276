# python process_voice.py domisoldo.wav
import numpy as np , scipy.io as sio , scipy.io.wavfile , matplotlib.pyplot as plt
import librosa.display , librosa , soundfile , signal, sys, sounddevice as sd
from scipy import signal
SAMPLE_RATE=48000; wav_file = sys.argv[1]
voice,sr = librosa.load(wav_file, sr =SAMPLE_RATE)
sd.play(voice,sr) ; sd.wait()
time = np.linspace(0, len(voice)/sr, len(voice)); 
#f = open(sys.argv[2],'w');
#for i in range(len(voice)): out_str='+ {} {}\n'.format(time[i],voice[i]); f.write(out_str); 
#f.close()
plt.plot(time,voice,'b'); plt.xlabel('time (s)'); plt.ylabel('amplitude'); plt.show()

hop_length = 512; n_fft = 8192  
hop_length_duration = float(hop_length)/sr; n_fft_duration = float(n_fft)/sr

stft = librosa.stft(voice, n_fft=n_fft, hop_length=hop_length)

magnitude = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(magnitude)
hz_range = [0, 4000]
bin_range = (np.array(hz_range)/(sr/2) * (n_fft/2+1)).astype(int)
log_spec = log_spectrogram[bin_range[0]:bin_range[1],:]
x_axis = np.arange(0, log_spec.shape[1]*hop_length/sr, hop_length/sr)
y_axis = np.arange(hz_range[0], hz_range[1], (hz_range[1]-hz_range[0])/(bin_range[1]-bin_range[0]))
plt.pcolormesh(x_axis, y_axis, log_spec)
plt.xlabel("Time"); plt.ylabel("Frequency"); plt.title("Spectrogram (dB)"); plt.show()

