# python check_discon.py input_filename.dat output_filename.txt
import numpy as np, matplotlib.pyplot as plt, sys
f = open(sys.argv[1],'rb'); data_ecg_ = f.read();f.close(); # data_ecg_ utf8 code, needs conversion
data_ecg = np.zeros(len(data_ecg_),dtype = 'int'); 
for i in range(len(data_ecg)): data_ecg[i] = int(data_ecg_[i]);
# check discontinuity by usb missing data 
def check_discontinuity(data):
    t=np.zeros(2,dtype='float');y=np.zeros(2);y[0]=0;y[1]=255;
    for i in range(2,len(data)):
        if(np.abs(data[i]-data[i-1]) > 3): t[0]=float(i);t[1]=t[0];plt.plot(t,y,'k')
    return;
def recover_large_data(data):
    i = 1;
    while(i < len(data)): 
        if( (data[i]-data[i-1]) < -200 ): 
            while( data[i] < 50): data[i] = data[i] + 256; i +=  1
        else:  i +=  1
    return(data)
data_ecg = recover_large_data(data_ecg);
plt.figure(1);plt.plot(data_ecg,'b.');plt.title('ECG measured data: '+sys.argv[1]);
check_discontinuity(data_ecg)
plt.show()
f = open(sys.argv[2],'w'); 
for i in range(len(data_ecg)): f.write("%s\n" %data_ecg[i]);
f.close();
