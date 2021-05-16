# python plot_Z.py data_filename.txt
import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys
def read_text_file(filename):
    fp = open(filename,'r'); a_list = list()
    while True:
        a = fp.readline().split()
        if a is None or len(a)==0: break
        else: a_list.append([float(b) for b in a])
    np_array = np.array(a_list)
    return np_array

mag_phase = read_text_file(sys.argv[1]) ; f=mag_phase[:,0]
plt.figure(1);plt.subplot(1,2,1);plt.loglog(f,mag_phase[:,1],'bo');plt.title('magnitude of impedance(meas)');
plt.subplot(1,2,2);plt.semilogx(f,mag_phase[:,2],'bo');plt.title('phase of impedance(meas)'); plt.show();
