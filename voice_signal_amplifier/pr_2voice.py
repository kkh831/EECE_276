# python process_voice_1.py domisoldo.wav re_recorded.wav max_frq_spectrogram time_range_for_align
# python pr_2voice.py domisoldo.wav 2nd_recording.wav 8000
import numpy as np , scipy.io as sio , scipy.io.wavfile , matplotlib.pyplot as plt
import librosa.display , librosa , soundfile , signal, sys, sounddevice as sd
from scipy import signal
SAMPLE_RATE=48000; wav_file_1 = sys.argv[1]; wav_file_2 = sys.argv[2]
voice_1,sr=librosa.load(wav_file_1,sr=SAMPLE_RATE);#sd.play(voice_1,sr);sd.wait()
voice_2,sr=librosa.load(wav_file_2,sr=SAMPLE_RATE);##sd.play(1.5*voice_2,sr);sd.wait()
# compute power of voice_1 & voice_2
power_1 = 0; power_2 = 0;
for i in range(len(voice_1)): power_1 += voice_1[i]**2;
for i in range(len(voice_2)): power_2 += voice_2[i]**2;
print('power of original voice and 2nd-recorded voice =',power_1,power_2);
ratio_amplitude = np.sqrt(power_1/power_2);#sd.play(ratio_amplitude*voice_2,sr);sd.wait()
time_1 = np.linspace(0, len(voice_1)/sr, len(voice_1)); 
time_2 = np.linspace(0, len(voice_2)/sr, len(voice_2)); 
plt.figure(1);plt.subplot(2,1,1);
plt.plot(time_1,voice_1,'b'); plt.xlabel('time (s)'); plt.title('original waveform'); #plt.show()
plt.subplot(2,1,2);
plt.plot(time_2,ratio_amplitude*voice_2,'r'); plt.xlabel('time (s)'); plt.title('2nd recording waveform'); plt.show()

hop_length = 512; n_fft = 8192  
hop_length_duration = float(hop_length)/sr; n_fft_duration = float(n_fft)/sr

stf_1 = librosa.stft(voice_1, n_fft=n_fft, hop_length=hop_length)
stf_2 = librosa.stft(voice_2, n_fft=n_fft, hop_length=hop_length)

mag_1 = np.abs(stf_1); log_spectrogram_1 = librosa.amplitude_to_db(mag_1)
mag_2 = np.abs(stf_2); log_spectrogram_2 = librosa.amplitude_to_db(mag_2)
#hz_range = [0, 8000]
hz_range = [0, float(sys.argv[3])]
bin_range = (np.array(hz_range)/(sr/2) * (n_fft/2+1)).astype(int)
log_spec_1 = log_spectrogram_1[bin_range[0]:bin_range[1],:]
x_axis_1 = np.arange(0, log_spec_1.shape[1]*hop_length/sr, hop_length/sr)
y_axis_1 = np.arange(hz_range[0], hz_range[1], (hz_range[1]-hz_range[0])/(bin_range[1]-bin_range[0]))
plt.figure(2);plt.subplot(2,1,1);
plt.pcolormesh(x_axis_1, y_axis_1, log_spec_1)
plt.xlabel("Time"); plt.ylabel("Frequency"); plt.title("Original Spectrogram (dB)"); #plt.show()
log_spec_2 = log_spectrogram_2[bin_range[0]:bin_range[1],:]; plt.subplot(2,1,2);
x_axis_2 = np.arange(0, log_spec_2.shape[1]*hop_length/sr, hop_length/sr)
#y_axis_2 = np.arange(hz_range[0], hz_range[1], (hz_range[1]-hz_range[0])/(bin_range[1]-bin_range[0]))
plt.pcolormesh(x_axis_2, y_axis_1, log_spec_2)
plt.xlabel("Time"); plt.ylabel("Frequency"); plt.title("2nd Recording Spectrogram (dB)"); plt.show()
