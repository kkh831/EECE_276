import numpy as np
def split_wav(signal, fs):
    min_length_sec = 0.1
    max_length_sec = 1
    width=int(0.2*fs);  # 0.5
    shift=int(0.1*fs); # 0.1
    THR=0.4 # 0.15, 0.4

    out_length_min = int(min_length_sec*fs)
    out_length_max = int(max_length_sec*fs)
    utter_avg = np.mean(np.abs(signal)); 
    speech_or_silence = np.zeros(len(signal)); 
    num_steps = int((len(signal)-width)/shift)

    for fr in range(num_steps):
        begin = fr*shift
        frame=signal[begin:begin+width]
        if np.mean(np.abs(frame))>utter_avg*THR: speech_or_silence[begin:begin+width]=1

    speech_indexes = [np.where(speech_or_silence==0)[0], np.where(speech_or_silence==1)[0]]
    utterance_begin=speech_indexes[True][0]
    utterance_end=speech_indexes[False][np.where(speech_indexes[False]>utterance_begin)[0][0]]
    
    out_sample_count=0
    start_sample=utterance_begin
    end_sample=utterance_end
    out_buffer_list = list()
    wav_start_list = list()
    wav_end_list = list()
    while 1:
        assert(utterance_begin < utterance_end), 'begin(%d),end(%d)'%(utterance_begin,utterance_end)
        wav_start = utterance_begin
        out_buffer = signal[utterance_begin:utterance_end]
        out_sample_count = out_buffer.shape[0]

        if out_sample_count < out_length_min:
            while 1:
                ## Find next begin & end
                try: # Find next begin
                    utterance_begin = speech_indexes[   True][np.where(speech_indexes[   True] > utterance_end)[0][0]]
                except: # No more begin
                    break
                try: # Find next end
                    utterance_end = speech_indexes[False][np.where(speech_indexes[False]>utterance_begin)[0][0]]
                except: # No more end
                    break
            assert(utterance_begin < utterance_end), 'begin(%d),end(%d)'%(utterance_begin,utterance_end)
            out_buffer = np.concatenate([out_buffer, signal[utterance_begin:utterance_end]])
            out_sample_count = out_buffer.shape[0]
            if out_sample_count >= out_length_min: break
        #wav_end = utterance_end
        
        wav_end =  wav_start + int(max_length_sec*fs)
        if len(out_buffer) < out_length_max:
            lack_len = out_length_max - len(out_buffer); append_buffer =signal[utterance_end:utterance_end+lack_len]
            out_buffer = np.concatenate([out_buffer, append_buffer])
            '''
            # Do not allow zero padding in Recoder program.
            lack_len2 = lack_len - len(append_buffer)
            if lack_len2 == 0:
                out_buffer = np.concatenate([out_buffer, append_buffer])
            elif lack_len2 > 0:
                zero_buffer= np.zeros([lack_len2],np.int16)
                out_buffer = np.concatenate([out_buffer, append_buffer, zero_buffer])
            else: 
                assert(0)
            '''
        else: 
            out_buffer = out_buffer[:int(max_length_sec*fs)]

        out_buffer_list.append(out_buffer)
        wav_start_list.append(wav_start)
        wav_end_list.append(wav_end)
        
        ## Find next begin & end
        try: # Find next begin
            utterance_begin = speech_indexes[   True][np.where(speech_indexes[   True] > utterance_end)[0][0]]
        except: # No more begin
            break
        try: # Find next end
            utterance_end = speech_indexes[False][np.where(speech_indexes[False]>utterance_begin)[0][0]]
        except: # No more end
            break
    return out_buffer_list, [speech_or_silence, wav_start_list, wav_end_list]
