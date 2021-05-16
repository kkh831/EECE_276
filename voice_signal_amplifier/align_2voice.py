# python process_voice_1.py domisoldo.wav re_recorded.wav max_frq_spectrogram start_time_align end_time_align
# python align_2voice.py domisoldo.wav 2nd_recording.wav 8000 0.115 0.145  
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
# align time space
start_index=int(sr*float(sys.argv[4])); end_index=int(sr*float(sys.argv[5]));
cross_corr_ = np.zeros(end_index-start_index, dtype='float');
#print('length of voice_1 and voive_2 = ',len(voice_1),len(voice_2));
max_index=-1; max_cross_corr=0;
for i in range(end_index-start_index):
    cross_corr_[i] = 0; 
    if(i % 100 == 0):  print('Align time in progress: ',i,'  out of ',end_index-start_index);
    for j in range(end_index-start_index,len(voice_1)-int(0.6*sr)):
        if(j+i+start_index < len(voice_2)):
            #print('index  1 & 2 =',j,j+i+start_index);
            cross_corr_[i] += voice_1[j] * voice_2[j+i+start_index];
    if( cross_corr_[i] > max_cross_corr): max_index=i; max_cross_corr=cross_corr_[i];
print('============= max index of voice_2= ',start_index+max_index,'  max_cross_corr=',max_cross_corr);
plt.figure(1);plt.plot(cross_corr_);plt.show()
