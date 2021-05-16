# python cut_cont.py input_filename.txt index_start index_end output_filename.txt
import numpy as np, matplotlib.pyplot as plt, sys
f = open(sys.argv[1],'r'); data_ecg_string = f.read();f.close();
data_ecg_string_list=data_ecg_string.split("\n")[:-1]; # remove last line
data_ecg_int_list = [int(val) for val in data_ecg_string_list]
print(len(data_ecg_int_list));
start_index=int(sys.argv[2]); end_index=int(sys.argv[3]);
n_out=end_index-start_index;   f = open(sys.argv[4],'w');
for i in range(n_out): 
    f.write("%s\n" %data_ecg_int_list[i + start_index]);
f.close()
plt.figure(1);plt.plot(data_ecg_int_list[start_index:end_index],'b.');
plt.title('ECG measured data '+sys.argv[4]);plt.show()
