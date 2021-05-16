import cmath as cm, numpy as np, matplotlib.pyplot as plt

f=np.logspace(-2,4,1000);

s=1j*2*np.pi*f;

G = -1222/(1+s/177)*s/(s+10.98)
H = -1222/(1+s/177)*s/(s+4.54)*s/(s+6.44)

for i in range(10):
    print(H[i]);

plt.figure(1);
plt.subplot(2,1,1); plt.loglog(f,(abs(G)),'go');
plt.loglog(f,(abs(H)));
plt.title('magnitude TF vs f   blue: Av1(s), green: Av2(s)');

ph_G=np.zeros(1000); 
ph_H=np.zeros(1000);
for i in range(1000):  
    ph_G[i]=cm.phase(G[i])*180/np.pi;
for i in range(1000):
    ph_H[i]=cm.phase(H[i])*180/np.pi;
plt.subplot(2,1,2); plt.semilogx(f,ph_G,'go');
plt.semilogx(f,ph_H);
plt.title('phase of TF vs f   blue: Av1(s), green: Av2(s)'); plt.show();
