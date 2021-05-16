# python plot_Z_meas_calc.py data_filename.txt param_val.txt
import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys

def read_text_file(filename):
    fp = open(filename,'r'); a_list = list()
    while True:
        a = fp.readline().split()
        if a is None or len(a)==0: break
        else: a_list.append([float(b) for b in a])
    np_array = np.array(a_list)
    return np_array

def calcZ(s, Rs, Ls, Rp, Lp, Cp):
    Zc =  Rs + s*Ls + 1/(1/Rp + 1/s/Lp + s * Cp );
    return Zc
mag_phase = read_text_file(sys.argv[1]) ; f=mag_phase[:,0]; s=1j*2*np.pi*f; Nf=f.shape[0]
# parameter values
param_val = read_text_file(sys.argv[2])
Rs = param_val[0,0]; Ls = param_val[0,1]; 
Rp = param_val[0,2]; Lp = param_val[0,3]; Cp = param_val[0,4];
Zcalc = calcZ(s, Rs, Ls, Rp, Lp, Cp)
plt.figure(20); ph_Zcalc=np.zeros(Nf,dtype='float');
for i in range(Nf): ph_Zcalc[i]=cm.phase(Zcalc[i])*180/np.pi;
plt.figure(30);plt.subplot(1,2,1);plt.loglog(f,mag_phase[:,1],'bo'); plt.loglog(f,abs(Zcalc),'rx'); plt.title('magnitude of impedance(blue:meas red:calc)');
plt.subplot(1,2,2);plt.semilogx(f,mag_phase[:,2],'bo'); plt.semilogx(f,ph_Zcalc,'rx'); plt.title('phase of impedance(blue:meas red:calc)'); 
plt.show();
