# gradient descent optimizer of speaker equivalent circuit
# python parallel_grad_desc.py data_filename.txt Niteration initial_guess.txt optimized_value.txt 
#initial_guess.txt optimized_values.txt: Rs Ls Rp Lp Cp
import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys

def read_text_file(filename):
    fp = open(filename,'r')
    a_list = list()
    while True:
        a = fp.readline().split()
        if a is None or len(a)==0: break
        else: a_list.append([float(b) for b in a])
    np_array = np.array(a_list)
    return np_array

def calcZ(s, Rs, Ls, Rp, Lp, Cp):
    Zc =  Rs + s*Ls + 1/(1/Rp + 1/s/Lp + s * Cp );
    return Zc

def Loss(s, Rs, Ls, Rp, Lp, Cp, Zm):
    Zc = calcZ(s, Rs, Ls, Rp, Lp, Cp); L = 0.0
    for i in range(s.shape[0]):  
        L += pow((abs(Zm[i])-abs(Zc[i]))/(abs(Zm[i])+abs(Zc[i])),2)
    return L / s.shape[0]

mag_phase = read_text_file(sys.argv[1]); f = mag_phase[:,0]; Nf = f.shape[0];

Zmeas = np.zeros(Nf, dtype='complex'); p = np.pi/180;

for i in range(Nf):  
    Zmeas[i]=mag_phase[i,1]*(np.cos(mag_phase[i,2]*p)+1j*np.sin(mag_phase[i,2]*p))
    
# initial condition 
init_guess = read_text_file(sys.argv[3])
Rs = init_guess[0,0]; Ls = init_guess[0,1];
Rp = init_guess[0,2]; Lp = init_guess[0,3]; Cp = init_guess[0,4];

print('Initial guess = ',Rs,Ls,Rp,Lp,Cp)

dRs=0.01; dLs=1e-7; dRp=0.01; dLp=1e-7; dCp=1e-7; s=1j*2*np.pi*f;

Niter=int(sys.argv[2]); Losses=np.zeros(Niter,dtype='float');

for i in range(Niter):
    Loss0 = Loss(s, Rs, Ls, Rp, Lp, Cp, Zmeas); 
    Losses[i]=Loss0
    dLdRs = Loss(s, Rs+dRs, Ls, Rp, Lp, Cp, Zmeas) - Loss0
    dLdLs = Loss(s, Rs, Ls+dLs, Rp, Lp, Cp, Zmeas) - Loss0
    dLdRp = Loss(s, Rs, Ls, Rp+dRp, Lp, Cp, Zmeas) - Loss0
    dLdLp = Loss(s, Rs, Ls, Rp, Lp+dLp, Cp, Zmeas) - Loss0
    dLdCp = Loss(s, Rs, Ls, Rp, Lp, Cp+dCp, Zmeas) - Loss0
    Rs = Rs - np.sign(dLdRs)*dRs; Rs = max(Rs, 10*dRs);
    Ls = Ls - np.sign(dLdLs)*dLs; Ls = max(Ls, 10*dLs);
    Rp = Rp - np.sign(dLdRp)*dRp; Rp = max(Rp, 10*dRp);
    Lp = Lp - np.sign(dLdLp)*dLp; Lp = max(Lp, 10*dLp);
    Cp = Cp - np.sign(dLdCp)*dCp; Cp = min(Cp, 1000000*dCp);
    
    
print('iter=',i,'; Loss=',Loss0, Rs,Ls,Rp,Lp,Cp)

plt.figure(10);plt.subplot(1,3,1);plt.loglog(Losses,'bo');plt.title('Loss vs Iteration');

Zcalc = calcZ(s, Rs, Ls, Rp, Lp, Cp)

plt.subplot(1,3,2);plt.loglog(f,abs(Zmeas),'bo');plt.loglog(f,abs(Zcalc),'rx');
plt.title('magnitude of impedance: blue(meas) red(calc)');

ph_Zmeas=np.zeros(Nf,dtype='float'); ph_Zcalc=np.zeros(Nf,dtype='float');

for i in range(Nf): ph_Zmeas[i]=cm.phase(Zmeas[i])*180/np.pi; ph_Zcalc[i]=cm.phase(Zcalc[i])*180/np.pi;
plt.subplot(1,3,3);plt.semilogx(f,ph_Zmeas,'bo');plt.semilogx(f,ph_Zcalc,'rx');plt.title('phase of impedance: blue(meas) red(calc)');
plt.show();

f=open(sys.argv[4],'w'); out_str="{} {} {} {} {}".format(Rs,Ls,Rp,Lp,Cp); f.write(out_str); f.close() 
