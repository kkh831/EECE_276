# code written by Noh Hyun-kyu POSTECH Oct 2019
# code written by Nam Hyunjoon POSTECH Sep 2020
# usage: python recorder.py name_initial+student_number
# example: python recorder.py NHJ20192643

import time, queue, os, tkinter,sys, multiprocessing, soundfile as sf
import matplotlib, matplotlib.pyplot as plt, sounddevice as sd, numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec
from threading import Thread
from split import split_wav
matplotlib.use('TkAgg')
def cleanplot():
    global ax2; ax2.clear(); canvas.draw(); return
def updateplot(q): # Plot graph for Thread
    global ax2, canvas, fs
    try :
        result=q.get_nowait() # Must : len(result[0]) > 1000 
        down_smp2 = 20; audio_down = result[0][::down_smp2]; 
        wav_start = max(0, int(result[2]/down_smp2) ); 
        wav_end = min( len(audio_down)-1, int(result[3]/down_smp2))
        tidx = np.arange(len(audio_down))/fs*down_smp2;
        pos1 = max(0, wav_start - int(fs*0.1/down_smp2)); 
        pos2 = min(wav_end + int(fs*0.1/down_smp2), len(audio_down)-1)
        ax2.clear(); ax2.plot(tidx[pos1:pos2], audio_down[pos1:pos2]); 
        ax2.axvline(tidx[wav_start], color='r',linestyle='--'); 
        ax2.axvline(tidx[wav_end-1], color='r',linestyle='--');
        canvas.draw()
    except Exception as ex: print('Error at update plot: ', ex);
    return

student_number = sys.argv[1]; fs = 48000; isAlive = True; 
isRecord = False; cur_key = None; is_plot_end=False; splitAct = -1 
# Wait: -1, Play: 0, Accept: 1, Decline: 2
def record_audio_to_wav_file(q):
    global cur_key, isRecord, isAlive, splitAct, is_plot_end
    while isAlive: 
        time.sleep(0.1)
        if isRecord :
            # [ Record audio ]
            audio = list()
            myque = queue.Queue()
            def callback(indata, frames, time, status): myque.put(indata.copy()); return
            btn_stop['state']='normal'; btn_key1['state']='disable'; btn_key2['state']='disable'; btn_key3['state']='disable'; 
            btn_key4['state']='disable'; btn_key5['state']='disable'; btn_key6['state']='disable';
            with sd.InputStream(samplerate = fs, channels=1, callback=callback):
                while True: 
                    if isRecord == False : break; 
                    audio.append(myque.get()) # myque.get(): np array, (1248, 1)
            audio = np.vstack(audio); audio = audio[int(0.2*fs):-int(0.2*fs)]; print('Shape of recorded voice: ',audio.shape)
            
            # [ Split audio ]
            try: split_list, plot_info = split_wav(audio,fs); print('Recorded voice is splitted to {} wavform'.format(len(split_list)))
            except Exception as ex: print('Error: Split_wav is failed with Recorded voice: ',ex) 
            btn_stop['state']='disable'; btn_play['state']='normal'; btn_accept['state']='normal'; btn_decline['state']='normal'
            folder = 'data_' + student_number + '/' +cur_key + '/'
            for idx in range(len(split_list)):
                splitAct = -1
                btn_play['state']='disable'; btn_accept['state']='disable'; btn_decline['state']='disable';
                filename = folder + cur_key + '_' +student_number + '_'  + str(len(os.listdir(folder))) + '.wav'
                q.put([audio, plot_info[0], plot_info[1][idx], plot_info[2][idx], filename])
                window.after(1, updateplot, q); 
                sd.play(split_list[idx], fs);  sd.wait()
                btn_play['state']='normal'; btn_accept['state']='normal'; btn_decline['state']='normal';
                while True:
                    if not isAlive    : return
                    if   splitAct == 0: splitAct = -1; sd.play(split_list[idx], fs);   sd.wait();   # Play
                    elif splitAct == 1: 
                        sf.write(filename, split_list[idx], fs,  subtype='PCM_16'); 
                        print('Save wav as ',filename,':',np.shape(split_list[idx]));split_list[idx]=0; break # Accept
                    elif splitAct == 2: break                                           # Decline
                    else              : time.sleep(0.1)                                 # Wait
            window.after(1, cleanplot);
            btn_play['state']='disable'; btn_accept['state']='disable'; btn_decline['state']='disable'; 
            btn_key1['state']='normal'; btn_key2['state']='normal'; btn_key3['state']='normal'; 
            btn_key4['state']='normal'; btn_key5['state']='normal'; btn_key6['state']='normal'; print('Waves are finished')
    print('Thread is finished')
    return
    
def myAction(cmd):
    global cur_key, isRecord, splitAct, window, q, isAlive
    if   cmd in "ALEXA,BIXBY,GOOGLE,JINIYA,KLOVA,UNKNOWN" : cur_key = cmd; isRecord = True
    elif cmd == "stop_rec": isRecord = False
    elif cmd == "replay"    : splitAct=0
    elif cmd == "accept"  : splitAct=1
    elif cmd == "decline" : splitAct=2
    elif cmd == "quit"    : isAlive=False; isRecord=False; window.destroy(); sys.exit();
    return        
#===================================================================================#
# [ 1.1 Window setting ]
window=tkinter.Tk()
window.title("POSTECH Speech Command Recognition Recoder")
window.geometry("600x400+100+100")
window.resizable(False, False)
#----------------------------------------------------------------------------------#
# [ 1.2 Button ]
os.makedirs('data_' + student_number, exist_ok=True);os.makedirs('data_' + student_number + '/ALEXA', exist_ok=True);os.makedirs('data_' + student_number + '/BIXBY', exist_ok=True);os.makedirs('data_' + student_number + '/GOOGLE', exist_ok=True);os.makedirs('data_' + student_number + '/JINIYA', exist_ok=True);os.makedirs('data_' + student_number + '/KLOVA', exist_ok=True);os.makedirs('data_' + student_number + '/UNKNOWN', exist_ok=True);
btn_key1=tkinter.Button(window,overrelief="solid",text='ALEXA',width=8,command=lambda:myAction('ALEXA'),repeatdelay=1000,repeatinterval=1000);btn_key1.place(x=20,y=10)
btn_key2=tkinter.Button(window,overrelief="solid",text='BIXBY',width=8,command=lambda:myAction('BIXBY'),repeatdelay=1000,repeatinterval=1000);btn_key2.place(x=120,y=10)
btn_key3=tkinter.Button(window,overrelief="solid",text='GOOGLE',width=8,command=lambda:myAction('GOOGLE'),repeatdelay=1000,repeatinterval=1000);btn_key3.place(x=220,y=10)
btn_key4=tkinter.Button(window,overrelief="solid",text='JINIYA',width=8,command=lambda:myAction('JINIYA'),repeatdelay=1000,repeatinterval=1000);btn_key4.place(x=320,y=10)
btn_key5=tkinter.Button(window,overrelief="solid",text='KLOVA',width=8,command=lambda:myAction('KLOVA'),repeatdelay=1000,repeatinterval=1000);btn_key5.place(x=420,y=10)
btn_key6=tkinter.Button(window,overrelief="solid",text='UNKNOWN',width=8,command=lambda:myAction('UNKNOWN'),repeatdelay=1000,repeatinterval=1000);btn_key6.place(x=520,y=10)
btn_stop=tkinter.Button(window,overrelief="solid",text="StopREC",width=8,command=lambda:myAction("stop_rec"),repeatdelay=1000,repeatinterval=1000,bg='yellow',fg='black');btn_stop.place(x=70,y=60)
btn_play=tkinter.Button(window,overrelief="solid",text="Replay",width=8,command=lambda:myAction("replay"),repeatdelay=1000,repeatinterval=1000);btn_play.place(x=170,y=60)
btn_accept=tkinter.Button(window,overrelief="solid",text="Accept",width=8,command=lambda:myAction("accept"),repeatdelay=1000,repeatinterval=1000);btn_accept.place(x=270,y=60)
btn_decline=tkinter.Button(window,overrelief="solid",text="Decline",width=8,command=lambda:myAction("decline"),repeatdelay=1000,repeatinterval=1000);btn_decline.place(x=370,y=60)
btn_quit=tkinter.Button(window,overrelief="solid",text="Quit",width=8,command=lambda:myAction("quit"),repeatdelay=1000,repeatinterval=1000,bg='red',fg='black');btn_quit.place(x=470,y=60)
btn_stop['state']='disable';btn_play['state']='disable';btn_accept['state']='disable';btn_decline['state']='disable'
#----------------------------------------------------------------------------------#
# Run Main:
if __name__ == '__main__': ## MUST NOT ERASE!!!!!!!
    q = multiprocessing.Queue()
    mythread = Thread(target = record_audio_to_wav_file, args = (q,))
    mythread.start()

    fig = plt.figure(figsize=(6,3),dpi=100)
    
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax2= plt.subplot(gs[0])
    
    canvas = FigureCanvasTkAgg(fig, master = window)
    canvas.get_tk_widget().pack(side='bottom')
    canvas.draw()
    window.mainloop()
#===================================================================================#
