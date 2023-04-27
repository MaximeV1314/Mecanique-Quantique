import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import colors
import os
import glob

mpl.use('Agg')

"""
Code renvoyant une animation 2D de la dynamique d'un paquet d'ondes gaussien dans le potentiel choisi en utilisant le schéma numérique Runge-Kutta à l'ordre 4
et renvoyant la norme au carré sur tout l'espace ( = 1 normalement) de l'onde pour différent temps d'étude.
Attention, le temps de calcul est assez long (environ 15min). L'algorithme passant par les états propres est bien plus rapide à exécuter.

Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img_dynamique".
Exécuter le code. Une fois finit, ouvrir le cmd depuis le dossier initial et taper  :

    ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

une vidéo .mp4 devrait être créée.
"""

#############################################################
######################  paramètres   ########################
#############################################################

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = 'img_dynamique/'+i+'.png'

    return(i)

Lx = 5                              #longueur
Nb_x = 30                          #nombre de points de discrétisation en longueur
x = np.linspace(-Lx/2, Lx/2, Nb_x)
dx = Lx/(Nb_x-1)                     #pas longueur

Ly = 5                              #longueur
Nb_y = 30                          #nombre de points de discrétisation en longueur
y = np.linspace(-Ly/2, Ly/2, Nb_y)
dy = Ly/(Nb_y-1)                     #pas longueur

X, Y = np.meshgrid(x, y)

kx = 0                              #impulsion
X0 = -1                              #position initial du paquet d'onde
ky = 1
Y0 = 0

sigma = 0.4                        #largeur à mi-hauteur du packet d'onde gaussien initial

tf = 5                             #temps d'étude
dt = 0.001                          #pas de temps
t = np.arange(0, tf+dt, dt)
Nb_t = len(t)                       #nombre de point de discrétisation en temps
pas_img = 100                        #pas du nombre d'image
digit = 4

def onde_initiale(X, Y):
    ####################   paquet d'onde initial     ###################
    onde_init = np.exp(-((X-X0)/(2*sigma))**2) * np.exp(-((Y-Y0)/(2*sigma))**2) * np.exp(1j * kx * X) * np.exp(1j * ky * Y) * ( 2 * np.pi * sigma**2)**(-1/4)   #paqiet d'ondes gaussien 2D
    return (onde_init/np.linalg.norm(onde_init)).flatten()  #matrice ----> vecteur

def matrice2(V):

    ############   construction de la matrice dynamique   ###############

    vec = np.ones(Nb_x**2)
    A = (- 1j/dx**2 - 1j/dy**2 - 1j * V) * np.diag(vec) + 1j/(2*dx**2) * np.diag(vec[:-1], -1) + 1j/(2*dx**2) * np.diag(vec[:-1], 1) +\
       1j/(2*dy**2) * np.diag(vec[:-Nb_y], Nb_y) + 1j/(2*dy**2) * np.diag(vec[:-Nb_y], -Nb_y)
    for i in range(Nb_x, Nb_x**2, Nb_x):
        A[i, i-1] = 0
        A[i-1, i] = 0

    return A

def potentiel(X, Y):

    #########    fonctions potentiels      ##########
    V = np.zeros((Nb_x, Nb_y))
    #V[0:int(Nb_y/2) - 2, int(Nb_x/2) - 1:int(Nb_x/2) + 1] = V[int(Nb_y/2) + 1:, int(Nb_x/2) - 1:int(Nb_x/2) + 1] = 100     #fente
    #return V.flatten()                                             #potentiel carré
    return 10 * (X**2 + Y**2).flatten()                             #potentiel harmonique



def euler(X, Y):

    #########    résolution par la méthode d'Euler       ############

    V = potentiel(X, Y)
    A = matrice2(V)
    psi_t0 = onde_initiale(X, Y)

    psi_tn = psi_t0                         #initialisation
    psi = [psi_t0]                          #liste des fonctions ondes en tout temps

    for i in range(Nb_t):
        psi_tn = A @ psi_tn * dt + psi_tn

        if i%pas_img==0:
            psi.append(psi_tn)

    return psi

def rk4(X, Y):

    #######      résolution par Runge-Kutta 4      ########

    V = potentiel(X, Y)
    A = matrice2(V)
    psi_t0 = onde_initiale(X, Y)

    psi_tn = psi_t0
    psi = [psi_t0]

    for i in range(Nb_t):
        d1 = psi_tn
        d2 = psi_tn + dt/2 * A @ psi_tn
        d3 = psi_tn + dt/2 * A @ d2
        d4 = psi_tn + dt * A @ d3
        psi_tn = psi_tn + dt / 6 * (A @ d1 + 2 * A @ (d2 + d3) + A @ d4)

        if i%pas_img==0:
            psi.append(psi_tn)

    print("Calcul : w")

    return psi

def norme(psi):

    ########     vérification de la norme des ondes    #########

    for tn in range(0, len(psi), 1):
        print("Norme t = ", t[tn], " : ", np.linalg.norm(psi[tn]))


def graphique(psi):

    ########     animation       #########
    psi_min = np.min(np.real(psi* np.conjugate(psi)))
    psi_max = np.max(np.real(psi* np.conjugate(psi)))

    extension="img_dynamique/*.png"
    for f in glob.glob(extension):
      os.remove(f)

    for i in range(0, int(Nb_t/pas_img)):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        psi_mod = np.real(psi[i].reshape(Nb_y, Nb_x) * np.conjugate(psi[i].reshape(Nb_y, Nb_x)))
        im = ax.imshow(psi_mod, cmap = "jet",  norm=colors.SymLogNorm(linthresh = 0.01, vmin = psi_min, vmax = psi_max), interpolation = "gaussian")
        plt.colorbar(im)

        name_pic = name(int(i), digit)
        plt.savefig(name_pic, bbox_inches='tight', dpi=300)

        ax.clear()
        plt.close(fig)

        print(i / int(Nb_t/pas_img))

print("Initialisation : Succès")
psi = rk4(X, Y)
print("Calcul des Psi : Succès")
graphique(psi)
print("Enregistrement des images : Succès")
norme(psi)

#ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

