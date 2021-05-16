# python interpolate.py input_filename.txt output_filename.txt
import numpy as np, matplotlib.pyplot as plt, sys

def interpolate_discontinuity(data):
    data_intp = np.zeros(2*len(data),dtype=type(data));
    j=0;data_intp[j]=data[0];
    for i in range(1,len(data)):
        #print('i=',i);
        if(np.abs(data[i]-data[i-1]) < 4): data_intp[j]=data[i-1]; j += 1;
        else: #discontinuity
            v_start = data[i-1]; v_end = data[i]; prev_index = i-1-int(6e6/256/60);
            while(np.abs(data[i-1]-data_intp[prev_index]) > 20): prev_index -=int(6e6/256/60);
            print('difference between one 60Hz cycle=',data[i-1],data_intp[prev_index]);
            # find the number of multiples of 64 samples for missing data
            min_val=256; min_multiple=1; 
            #for k in range(1,5): 
            for k in range(2,5): 
                if(np.abs(data_intp[prev_index+k*64]-data[i]) < min_val): 
                    min_multiple = k; min_val = np.abs(data_intp[prev_index+k*64]-data[i])
            N_added = min_multiple * 64; 
            v_prev_start = data_intp[prev_index]; v_prev_end = data_intp[prev_index+N_added];
            for k in range(N_added): 
               #print('            v_prev_start, v_prev_end= ',v_prev_start,v_prev_end);
               gain = (v_start/v_prev_start*(N_added-k)+v_end/v_prev_end*k)/N_added
               data_intp[j] = gain * data_intp[prev_index+k]; j += 1;
            print('i, min_multiple=',i,min_multiple);
    data_interpolated = data_intp[0:j];
    return(data_interpolated);
            
f = open('right_ecg_intp.txt','r'); data_ecg_string_right = f.read();f.close();
data_ecg_string_list_right=data_ecg_string_right.split("\n")[:-1]; # remove last line
data_ecg_int_list_right = [int(float(val)) for val in data_ecg_string_list_right]

f = open('left_ecg_intp.txt','r'); data_ecg_string_left = f.read();f.close();
data_ecg_string_list_left=data_ecg_string_left.split("\n")[:-1]; # remove last line
data_ecg_int_list_left = [int(float(val)) for val in data_ecg_string_list_left]

data_R_L =np.zeros(len(data_ecg_int_list_right));

for i in range(len(data_R_L)):
    data_R_L[i] = data_ecg_int_list_right[i] - data_ecg_int_list_left[i];

f = open('diff_ecg_cont.txt','r'); data_ecg_diff = f.read();f.close();
data_ecg_diff_list=data_ecg_diff.split("\n")[:-1]; # remove last line
data_ecg_diff_int = [int(float(val)) for val in data_ecg_diff_list]

plt.figure(1);plt.subplot(2,1,1); plt.plot(data_R_L,'b.');plt.title('R_L');
plt.subplot(2,1,2); plt.plot(data_ecg_diff_int,'b.'); plt.title('diff');
plt.show() 
