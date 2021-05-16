# Usage: python inference.py (ckpt.path)
# Example: python inference.py result_dir/ckpt/ep~.ckpt
import os, sys, queue, time, tkinter, multiprocessing, glob, pickle
from threading import Thread
from scipy.io import wavfile
import sounddevice as sd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops
from model import load_variables_from_checkpoint, split_wav, create_model, normalize_in_time_and_amplitude
np.set_printoptions(precision=2)
isRecord = False; isAlive= True; isReplay = False
allwords = ['ALEXA','BIXBY','GOOGLE','JINIYA','KLOVA']

#===================================================================================#
def create_inference_graph(setting):
    wav_data_placeholder = tf.compat.v1.placeholder(tf.string, [], name='wav_data')
    decoded_sample_data = tf.audio.decode_wav(wav_data_placeholder, desired_channels=1, desired_samples=setting['desired_samples'], name='decoded_sample_data')
    input_ = tf.reshape(decoded_sample_data.audio,[setting['desired_samples'],1])
    spectrogram = gen_audio_ops.audio_spectrogram(input_, window_size=setting['window_size_samples'], stride=setting['window_stride_samples'], magnitude_squared=True)

    L2M_mtx = tf.constant(setting['L2M'], shape = setting['L2M'].shape, dtype=tf.float32, name="L2M_mtx")
    tf_pp = tf.math.log(tf.tensordot(spectrogram, L2M_mtx, 1)+1e-6)

    #print('tf_pp:', setting['preprocess'],':',tf_pp)
    fingerprint_size = setting['fingerprint_size']
    reshaped_input = tf.reshape(tf_pp, [-1, fingerprint_size])
    logits, _ = create_model(reshaped_input, setting, setting['model_architecture'], is_training=False)
    softmax = tf.nn.softmax(logits, name='labels_softmax')
    return input_, softmax, tf_pp

def prepare_inference_engine(model_path):
    ckpt_dir =  '/'.join(os.path.split(model_path)[:-1])
    setting_path = os.path.abspath(glob.glob(os.path.join(ckpt_dir,'..','setting.bin'))[0])
    with open(setting_path,'rb') as f:
        setting = pickle.load(f)
    
    sess = tf.compat.v1.InteractiveSession()
    input_tensor, output_tensor, preprocessor_tensor = create_inference_graph(setting)
    load_variables_from_checkpoint(sess, model_path)
    inference_engine = {'sess':sess, 'tf_in':input_tensor, 'tf_out':output_tensor, 'tf_pp':preprocessor_tensor}
    return type, inference_engine

def audioVec_to_label(inference_engine, type, audio):
    smp_len = 48000
    audio = np.array(audio, dtype=np.float32).flatten()
    if len(audio) < smp_len: audio = np.pad(audio, (0, smp_len-len(audio)) ,mode='constant')
    else: audio = audio[:smp_len]   
    
    audio = normalize_in_time_and_amplitude(audio)
    
    audio = np.expand_dims(audio, axis= -1)
    label = None; logmel = None; conv1_out=None;
    [label, logmel] = inference_engine['sess'].run([inference_engine['tf_out'], inference_engine['tf_pp']], feed_dict={inference_engine['tf_in']: audio})
        
    return label, audio, logmel

def speech_command_recognition_from_WAV(model_path, wav_path):
    type, inference_engine = prepare_inference_engine(model_path)
        
    [fs, audio]= wavfile.read(wav_path)
    audio = audio / 32767.0
    assert(fs == 48000),wav_path+"'s fs = "+fs+' != 48000'
    
    label, audio_out, logmel = audioVec_to_label(inference_engine, type, audio)
    print('KEY     :    ALX   BIX   GOO   JIN   KLO')
    print('label[%]: ',np.squeeze(label*100),' --> ', allwords[np.argmax(label)])
    if sys.argv[3] == 0: sd.play(audio_out, fs); sd.wait() # sys.argv[3] == mute
    print(wav_path,' --> ', allwords[np.argmax(label)])
    return
    
def speech_command_recognition_from_MIC(model_path):
    global isRecord, isAlive, isReplay
    type, inference_engine = prepare_inference_engine(model_path)
        
    fs = 48000
    first_record_check = False
    print('KEY     :    ALX   BIX   GOO   JIN   KLO')
    while isAlive:
        time.sleep(0.1)
        if isRecord :
            # [ Record audio ]
            audio = list()
            myque = queue.Queue()
            def callback(indata, frames, time, status): myque.put(indata.copy()); return
            with sd.InputStream(samplerate = fs, channels=1, callback=callback):
                while True: 
                    if isRecord == False : break; 
                    audio.append(myque.get())
            audio = np.vstack(audio); audio = audio.flatten(); #print('audio:',audio.shape)
            if len(audio) < 96000: audio = np.pad(audio,(0,96000 - len(audio)),'constant')
            first_record_check = False
            
            # [ Split audio ]
            split_list, plot_info = split_wav(audio,fs); 
            max_idx = np.argmax(np.array([np.sum(np.array(split_).flatten()**2) for split_ in split_list])) # Find max power
            
            # [ Recognition ]
            label, audio_out, logmel = audioVec_to_label(inference_engine, type, split_list[max_idx])
            print('label[%]: ',np.squeeze(label*100),' --> ', allwords[np.argmax(label)])
            q.put([audio, plot_info[0], plot_info[1][max_idx], plot_info[2][max_idx], allwords[np.argmax(label)], logmel])
            window.after(1,updateplot,q)
            sd.play(audio_out, fs); sd.wait()
            
        if isReplay and first_record_check == False: sd.play(audio_out, fs); isReplay = False
    return
#-----------------------------------------------------------------------------------#
def updateplot(q): # Plot graph for Thread
    try:
        fs = 48000
        result=q.get_nowait()
        ax1.clear(); tidx = np.arange(len(result[0]))/fs; ax1.plot(tidx,result[0]); ax1.plot(tidx,result[1]*max(result[0])); ax1.axvspan(tidx[result[2]],tidx[result[3]-1], alpha=0.5, color='red'); fig.suptitle(result[4], fontsize="x-large")
        pos1 = result[2] - int(fs*0.1)
        if pos1 < 0: pos1 = 0
        pos2 = result[3] + int(fs*0.1)
        if pos2 > len(result[0]): pos2 = len(result[0])
        ax2.clear(); ax2.plot(tidx[pos1:pos2], result[0][pos1:pos2]); ax2.axvline(tidx[result[2]], color='r',linestyle='--');ax2.axvline(tidx[result[3]-1], color='r',linestyle='--');
        ax1.set_ylim([-1,1]); ax2.set_ylim([-1,1]); ax2.set_xlabel('time [s]');
        if type(result[5]) != type(None):
            ax3.clear(); ax3.pcolormesh(np.squeeze(result[5]).T); ax3.set_ylabel('mel'); ax3.set_xlabel('frame'); 
            fig2.suptitle('MEL SPECTROGRAM: mel vs frame'); fig2.canvas.draw(); fig2.show()
        canvas.draw()
    except Exception as ex:
        print('Exception: ',ex)
        fig.suptitle("Record Error: Try again", fontsize="x-large")

def myAction(msg):
    global isRecord, isAlive, isReplay
    if msg =="Start": isRecord = True
    elif msg=="Stop": isRecord = False
    elif msg=="Replay": isReplay = True
    elif msg=="quit":  isAlive=False; isRecord=False; window.destroy(); sys.exit();
    else: assert(0),"Error: "+msg
    return
#===================================================================================#
# [ 1.1 Window setting ]
window=tkinter.Tk()
window.title("POSTECH Speech Command Recognition")
window.geometry("500x450+100+100")
window.resizable(False, False)
#-----------------------------------------------------------------------------------#
# [ 1.2 Button setting]
btn_key1 = tkinter.Button(window, overrelief="solid", text='Start' , width=7, command=lambda: myAction('Start') , repeatdelay=1000, repeatinterval=100); btn_key1.place(x=50, y=10)
btn_key2 = tkinter.Button(window, overrelief="solid", text='Stop'  , width=7, command=lambda: myAction('Stop')  , repeatdelay=1000, repeatinterval=100); btn_key2.place(x=150, y=10)
btn_key3 = tkinter.Button(window, overrelief="solid", text='Replay', width=7, command=lambda: myAction('Replay'), repeatdelay=1000, repeatinterval=100); btn_key3.place(x=250, y=10)
btn_quit = tkinter.Button(window, overrelief="solid", text="Quit"  , width=7, command=lambda: myAction("quit")  , repeatdelay=1000, repeatinterval=100, bg='red', fg='black');btn_quit.place(x=350, y=10)
#-----------------------------------------------------------------------------------#
# Run Main:
if __name__ == '__main__': ## MUST NOT ERASE!!!!!!!
    if len(sys.argv) == 4: # WAV mode: [1]: .tflite or .ckpt, [2]: .wav, [3]: mute
        speech_command_recognition_from_WAV(sys.argv[1], sys.argv[2]);
    elif len(sys.argv) == 2: # MIC mode: [1]: .tflite or .ckpt
        q = multiprocessing.Queue()
        mythread = Thread(target = speech_command_recognition_from_MIC, args=(sys.argv[1],))
        mythread.start()
        fig = plt.figure(figsize=(8.5,4))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3])
        ax1= plt.subplot(gs[0,0])
        ax2= plt.subplot(gs[1,0])
        
        #fig2 = plt.figure(num=2, figsize=(5,5))
        fig2 = plt.figure(figsize=(5,5))
        gs2 = gridspec.GridSpec(nrows=1, ncols=1)
        ax3 = plt.subplot(gs2[0,0])
        
        canvas = FigureCanvasTkAgg(fig, master = window)
        canvas.get_tk_widget().pack(side='bottom')
        canvas.draw()
        window.mainloop()
    else: 
        print("Do nothing")
        print("WAV mode argv: [1]: .tflite or .ckpt, [2]: .wav, [3]: sound_mute")
        print("MIC mode argv: [1]: .tflite or .ckpt")
#===================================================================================#




