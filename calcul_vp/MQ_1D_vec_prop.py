import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
import glob
from scipy.interpolate import spline

mpl.use('Agg')

#############################################################
#####################  fonctions  ###########################
#############################################################

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = 'img/'+i+'.png'

    return(i)

#############################################################
##############  définition des paramètres   #################
#############################################################


dt = 0.00001                    #pas de temps
tf = 8                   #temps final
t = np.arange(0, tf, dt)    #liste temps
N_t = len(t)

h = 0.1                           #pas barre
h2 = h**2
L = 20                           #longueur barre
l_part = np.arange(-L, L+h, h)    #liste barre
N = len(l_part)

hb = 1
m = 1
lim_g = 0
lim_d = 0
kx = -3
sig = 1
x0 = 0

digit = 4
pas_img = 10000

print("parameters initialization successed")

#############################################################
##############  Résolution éq chaleur   #####################
#############################################################
#V = 0.1 * l_part
#V[0 : 50] = 0

#V = 5 * l_part**2  #potentiel harmonique

V = np.zeros(N)     #effet tunnel
V[140:143] = 1000   #effet tunnel

#V = l_part * 0     #particule libre

A = np.zeros((N, N), dtype=np.complex128)
for i in range(1, N-1):
    A[i, i] = 1 - 1j * hb/m * dt/h2 - 1j/hb * dt * V[i]
    A[i, i+1] = 1j * hb/(2*m) * dt/h2
    A[i, i-1] = 1j * hb/(2*m) * dt/h2

A[0, 0] = 1 - 1j * hb/m * dt/h2 - 1j/hb * dt * V[0]
A[N-1, N-1] = 1 - 1j * hb/m * dt/h2 - 1j/hb * dt * V[-1]
A[0, 1] = 1j * hb/(2*m) * dt/h2
A[N-1, N-2] = 1j * hb/(2*m) * dt/h2

ub_0 = np.zeros((N), dtype=np.complex128)         #construction de la liste conditions aux bords
ub_0[0] = 0 * 1j * hb/(2*m) * dt/h2
ub_0[-1] = 0 * 1j * hb/(2*m) * dt/h2
un_init = (2*np.pi*sig**2)**(-1/4) * np.exp(-(l_part - x0)**2/(4*sig**2)) * np.exp(-1j * kx * l_part)
#un_init[140:] = 0

un = [un_init]
un_mod = []
ui = np.copy(un_init)
print("Matrice and vectors initialization successed")

for i in range(N_t):
    ui = A @ ui + ub_0

    if i%pas_img == 0:
        un.append(ui)
        un_mod.append(ui * np.conjugate(ui))

#############################################################
######################  Animation   #########################
#############################################################

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

for i in range(0, int(N_t/pas_img)):

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([-L-1, L+1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel("x")
    #ax.set_ylabel("psi/|ps")
    ax.grid()

    x_new = np.linspace(-L, L, 300)
    power_smooth = spline(l_part, un[i], x_new)
    ax.plot(x_new, np.real(power_smooth), label = "psi (real)", color = "orange")
    ax.plot(x_new, np.imag(power_smooth), label = "psi (imag)", color = "red")

    power_smooth_mod = spline(l_part, un_mod[i], x_new)
    ax.plot(x_new, np.real(power_smooth_mod), label = "|psi|²", color = "blue")

    ax.plot(l_part, V, label = "Potentiel", color = "green")

    ax.legend()
    name_pic = name(int(i), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    plt.close(fig)

    print(i / int(N_t/pas_img))

print("Images successed")

  #ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        #test
