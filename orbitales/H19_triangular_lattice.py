import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\FFMPEG\bin\ffmpeg.exe'
from matplotlib import colors
import os
import glob
#plt.rcParams['text.usetex'] = True

mpl.use('Agg')


Lx = 20                               #longueur
Nb_x = 100                            #nombre de points de discrétisation en longueur
x = np.linspace(-Lx/2, Lx/2, Nb_x)
dx = Lx/(Nb_x-1)                     #pas longueur

Ly = 20                              #longueur
Nb_y = 100                           #nombre de points de discrétisation en longueur
y = np.linspace(-Ly/2, Ly/2, Nb_y)
dy = Ly/(Nb_y-1)                     #pas longueur

X, Y = np.meshgrid(x, y)

tf = 1                             #temps d'étude
dt = 1/300                          #pas de temps
t = np.arange(0, tf+dt, dt)
Nb_t = len(t)                       #nombre de point de discrétisation en temps

digit = 4

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = i+'.png'

    return(i)


def Hamiltonien(V):
    vec = np.ones(Nb_x**2)
    H = (1/dx**2 + 1/dy**2 + V) * np.diag(vec) - 1/(2*dx**2) * np.diag(vec[:-1], -1) - 1/(2*dx**2) * np.diag(vec[:-1], 1) -\
       1/(2*dy**2) * np.diag(vec[:-Nb_y], Nb_y) - 1/(2*dy**2) * np.diag(vec[:-Nb_y], -Nb_y)
    for i in range(Nb_x, Nb_x**2, Nb_x):
        H[i, i-1] = 0
        H[i-1, i] = 0

    return H

def Potentiel(X, Y):
        #########    fonctions potentiels      ##########
    V = np.zeros((Nb_y, Nb_x))

    list_x = np.zeros(19)
    list_y = np.zeros(19)

    L = np.sqrt(3)

    list_x[0:3] = list_x[3:6] = [-2,0,2]
    list_x[6:10] = list_x[10:14] = [-3,-1,1,3]
    list_x[14:] = [-4,-2,0,2,4]

    list_y[0:3] = [2*L, 2*L, 2*L]
    list_y[3:6] = [-2*L, -2*L, -2*L]
    list_y[6:10] = [L, L, L, L]
    list_y[10:14] = [-L, -L, -L, -L]
    list_y[14:] = [0,0,0,0,0]

    V_1 = - 1/((X-list_x[0])**2 + (Y-list_y[0])**2)**(1/2)
    V_int = np.sum(1/((list_x[1:] - list_x[0])**2 + (list_y[1:] - list_y[0])**2)**(1/2))
    for i in range(1, 19):
        list_x_copy = np.delete(np.copy(list_x), i)
        list_y_copy = np.delete(np.copy(list_y), i)
        V_1 = V_1 - 1/((X-list_x[i])**2 + (Y-list_y[i])**2)**(1/2)
        V_int = V_int + np.sum(1/((list_x_copy - list_x[i])**2 + (list_y_copy - list_y[i])**2)**(1/2))

    V = V_1 + V_int                                                                                              #Hydrogène

    #V = -1/((X-2)**2 + Y*Y)**(1/2) - 1/((X+2)**2 + Y*Y)**(1/2) + 1/4                                                     #H2+
    return V.flatten()

def V_propres(A):
    return eigh(A)


def plot_vecteurs_propres():
    extension="img_vecteurs_propres/*.png"
    for f in glob.glob(extension):
        os.remove(f)

    N_t = 30
    T = np.linspace(0, 1, N_t)
    n = 0
    l = 0
    m = 0
    for i in range(50):
        m = m+1
        if m>l:
            l = l+1
            if l>n-1:
                n = n+1
                l = 0
            m = -l

        #, vmin = -0.1, vmax = 0.1
        vec_1 = np.real(vecteurs_p[i].reshape(Nb_y, Nb_x) * np.conjugate(vecteurs_p[i].reshape(Nb_y, Nb_x)))
        vec_2 = np.real(vecteurs_p[i+1].reshape(Nb_y, Nb_x) * np.conjugate(vecteurs_p[i+1].reshape(Nb_y, Nb_x)))

        for k in range(4 * N_t):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.patch.set_facecolor('black')

            if k<N_t:
                im = ax.imshow(vec_2 * T[k] + vec_1 * (1-T[k]), cmap = "magma", interpolation = "catrom")
            else:
                im = ax.imshow(vec_2, cmap = "magma", interpolation = "catrom")

            #ax.set_title("N = "+str(i+1))
            cbar = fig.colorbar(im)
            cbar.set_ticks([])

            #ax.set_xlabel("x")
            #ax.set_ylabel("y")

            ax.axis('off')

            #ax.set_title("n = " + str(n) + ", l = " + str(l) + ", m = " + str(m), color = "white")
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            name_pic = name(int(i * 3 * N_t + k), digit)
            plt.savefig("img_vecteurs_propres/"+name_pic, bbox_inches='tight', dpi=300)
            ax.clear()
            plt.close(fig)

V = Potentiel(X, Y)
print("Potentiel initialization successed")

H = Hamiltonien(V)
print("Hamiltonien initialization successed")

valeurs_p, vecteurs_p = V_propres(H)
vecteurs_p = np.transpose(vecteurs_p)
print("Eigenvalues and eigenvectors calculation successed")

plot_vecteurs_propres()

#ffmpeg -r 30 -i img_vecteurs_propres/%04d.png -vcodec libx264 -y -an 2d_H_eigen_vec.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

#ffmpeg -i 2d_schrodinger_carre_imp.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts intermediate2.ts
#ffmpeg -i "concat:intermediate1.ts|intermediate2.ts" -c copy -bsf:a aac_adtstoasc output.mp4
