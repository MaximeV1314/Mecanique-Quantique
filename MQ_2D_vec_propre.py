import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from scipy import linalg

from scipy.sparse.linalg import svds
import matplotlib as mpl
import os
import glob
import time

start_time = time.time()

mpl.use('Agg')

"""
Code renvoyant une animation de la dynamique d'un paquet d'ondes gaussien dans le potentiel choisi en utilisant la base des vecteurs propres de H,
et renvoyant les valeurs propres et les vecteurs propres de H associé au potentiel étudié.

Créer un fichier "img_vecteurs_propres" dans le dossier pour visualiser les fonctions propres de l'hamiltonien.

Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img_dynamique".
Exécuter le code. Une fois finit, ouvrir l'invité de commande depuis le dossier initial et taper  :

    ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

une vidéo .mp4 devrait être créée.
"""

Lx = 100                               #longueur
Nb_x = 60                            #nombre de points de discrétisation en longueur
x = np.linspace(-Lx/2, Lx/2, Nb_x)
dx = Lx/(Nb_x-1)                     #pas longueur
kx = 0                               #impulsion
X0 = -1                               #position initial du paquet d'onde

Ly = 100                              #longueur
Nb_y = 60                           #nombre de points de discrétisation en longueur
y = np.linspace(-Ly/2, Ly/2, Nb_y)
dy = Ly/(Nb_y-1)                     #pas longueur
ky = 1
Y0 = 0

X, Y = np.meshgrid(x, y)

sigma = 0.4                        #largeur à mi-hauteur du packet d'onde gaussien initial
B = 0

tf = 5                             #temps d'étude
dt = 0.1                          #pas de temps
t = np.arange(0, tf+dt, dt)
Nb_t = len(t)                       #nombre de point de discrétisation en temps

digit = 4
N = 30

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = i+'.png'

    return(i)

def onde_init():
    #onde initiale (paquet d'ondes gaussien)
    onde_init = np.exp(-((X-X0)/(2*sigma))**2) * np.exp(-((Y-Y0)/(2*sigma))**2) * np.exp(1j * kx * X) * np.exp(1j * ky * Y) * ( 2 * np.pi * sigma**2)**(-1/4)
    return (onde_init/np.linalg.norm(onde_init)).flatten()

def Hamiltonien(V):
    #construction de l'Hamiltonien
    vec = np.ones(Nb_x**2)
    H = (1/dx**2 + 1/dy**2 + V + 1j * B*Y.flatten()/dx - 1j * B*X.flatten()/dy) * np.diag(vec) - 1/(2*dx**2) * np.diag(vec[:-1], -1) - (1/(2*dx**2) + 1j * B*Y.flatten()/dx) * np.diag(vec[:-1], 1) -\
       (1/(2*dy**2) - 1j * B*X.flatten()/dy) * np.diag(vec[:-Nb_y], Nb_y) - 1/(2*dy**2) * np.diag(vec[:-Nb_y], -Nb_y)
    for i in range(Nb_x, Nb_x**2, Nb_x):
        H[i, i-1] = 0
        H[i-1, i] = 0

    return H

def Potentiel(X, Y):
        #########    fonctions potentiels      ##########
    V = np.zeros((Nb_y, Nb_x))

    #V[0:int(Nb_y/2) - 3, int(Nb_x/2) - 1:int(Nb_x/2) + 1] = V[int(Nb_y/2) + 3:, int(Nb_x/2) - 1:int(Nb_x/2) + 1] = 1000    #fente

    #V = 10*(X**2 + Y**2)                                                                                                   #harmonique

    #V[0:5, :] = V[-6:-1, :] = V[:, 0:5] = V[:, -6:-1] = 1000000000

    #V[0:int(Nb_y/2) - 7, int(Nb_x/2) - 1:int(Nb_x/2) + 1] = V[int(Nb_y/2) + 7:, int(Nb_x/2) - 1:int(Nb_x/2) + 1] =\        #double fente
    #V[int(Nb_y/2) - 3: int(Nb_y/2) + 3,int(Nb_x/2) - 1:int(Nb_x/2) + 1] = 1000

    V = -1/(X*X + Y*Y)**(1/2)                                                                                               #Hydrogène

    #V = -1/((X-2)**2 + Y*Y)**(1/2) - 1/((X+2)**2 + Y*Y)**(1/2) + 1/4                                                     #H2+
    return V.flatten()

def V_propres(A):
    #return eigsh(A, k = Nb_x, which = "SM")
    #return eigh(A)
    return linalg.eig(A)[0:2]

def an(valeurs_p, vecteurs_p, psi_0):
    #calcul des coefficients par la méthode des rectangles

    an_psi = []
    for i in range(N_vp):
        an_psi.append(np.sum(np.conjugate(vecteurs_p[i]) * psi_0) * dx * dy)

    return np.array(an_psi)

def dynamique():
    #--------------------------animation ---------------------------------#
    extension="img_dynamique/*.png"
    for f in glob.glob(extension):
        os.remove(f)

    for i in range(0, Nb_t):

        psi_tn = np.zeros(Nb_x * Nb_y)
        for k in range(N_vp):
            psi_tn = psi_tn + An[k] * vecteurs_p[k] * np.exp(-1j * valeurs_p[k] * t[i])

        if i==0:
            psi_max = np.max(np.real(psi_tn * np.conjugate(psi_tn)))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Dynamique (densité de probabilité)")

        psi_mod = np.real(psi_tn.reshape(Nb_y, Nb_x) * np.conjugate(psi_tn.reshape(Nb_y, Nb_x)))
        im = ax.imshow(psi_mod, cmap = "magma", interpolation = "gaussian")
        #, vmin = 0, vmax = psi_max/2
        plt.colorbar(im)

        name_pic = name(int(i), digit)
        plt.savefig("img_dynamique/"+name_pic, bbox_inches='tight', dpi=300)

        ax.clear()
        plt.close(fig)

        #print(i/Nb_t)

def plot_valeurs_propres():
    #plot des valeurs propres

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Valeurs porpres N°")
    ax.set_ylabel("E (u.a.)")
    #ax.set_xlim(0, 50)
    #ax.set_ylim(0,valeurs_p[50])

    ax.plot(np.arange(1, N+1, 1), np.sort(valeurs_p[:N]), ".")
    ax.grid()

    plt.savefig("valeurs_propres_2d.png", bbox_inches='tight', dpi=300)

def plot_vecteurs_propres():
    #plot des vecteurs propres

    extension="img_vecteurs_propres/*.png"
    for f in glob.glob(extension):
        os.remove(f)

    m = 0
    n = 0
    l=0
    index = np.argsort(valeurs_p)
    for i in range(N):

        m = m+1
        if m>l:
            l = l+1
            if l>n-1:
                n = n+1
                l = 0
            m = 0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        #, vmin = -0.1, vmax = 0.1
        im = ax.imshow(np.real(vecteurs_p[index[i]].reshape(Nb_y, Nb_x) * np.conjugate(vecteurs_p[index[i]].reshape(Nb_y, Nb_x))), cmap = "magma", interpolation = "catrom")
        ax.set_title("n = " + str(n) + ", l = " + str(l) + ", m = " + str(m), color = "black")
        plt.colorbar(im)

        name_pic = name(int(i), digit)
        plt.savefig("img_vecteurs_propres/"+name_pic, bbox_inches='tight', dpi=300)
        ax.clear()
        plt.close(fig)

psi_0 = onde_init()

V = Potentiel(X, Y)
print("Potentiel initialization successed")

H = Hamiltonien(V)
print("Hamiltonien initialization successed")

valeurs_p, vecteurs_p = V_propres(H)
vecteurs_p = np.transpose(vecteurs_p)
print("Eigenvalues and eigenvectors calculation successed")

N_vp = len(valeurs_p)
An = an(valeurs_p, vecteurs_p, psi_0)
print("Initial coefficients calculation successed")

print("\n--- %s seconds ---" % (time.time() - start_time))

#dynamique()
plot_valeurs_propres()
plot_vecteurs_propres()

#ffmpeg -r 30 -i img_dynamique/%04d.png -vcodec libx264 -y -an 2d_schrodinger_carre_imp.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

#ffmpeg -i 2d_schrodinger_carre_imp.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts intermediate2.ts
#ffmpeg -i "concat:intermediate1.ts|intermediate2.ts" -c copy -bsf:a aac_adtstoasc output.mp4
