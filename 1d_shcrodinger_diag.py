import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\FFMPEG\bin\ffmpeg.exe'     #donner le chemin vers ffmpeg.exe

"""
Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img_dynamique".
Exécuter le code. Une fois finit, ouvrir le cmd depuis le dossier initial et taper  :

    ffmpeg -r 30 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

une vidéo .mp4 devrait être créée.
"""

#############################################################
######################  paramètres   ########################
#############################################################

Lx = 30                              #longueur
Nb_x = 1000                          #nombre de points de discrétisation en longueur
x = np.linspace(-Lx/2, Lx/2, Nb_x)
dx = Lx/(Nb_x-1)                     #pas longueur

x0 = 0
kx = -3

sigma = 1                        #largeur à mi-hauteur du packet d'onde gaussien initial

tf = 10                             #temps d'étude
dt = 0.01                          #pas de temps
t = np.arange(0, tf+dt, dt)
Nb_t = len(t)                       #nombre de point de discrétisation en temps

def onde_initiale():
    ####################   paquet d'onde initial     ###################
    onde_init = np.exp(-((x-x0)/(2*sigma))**2) * np.exp(1j * kx * x) * ( 2 * np.pi * sigma**2)**(-1/4)  #paqiet d'ondes gaussien 2D
    return (onde_init)  #matrice ----> vecteur

def Hamiltonien():
    #construction de l'Hamiltonien
    vec = np.ones(Nb_x)
    H = (1/dx**2 + V) * np.diag(vec) - 1/(2*dx**2) * np.diag(vec[:-1], -1) \
    - 1/(2*dx**2) * np.diag(vec[:-1], 1)
    return H


def potentiel():
    #########    fonctions potentiels      ##########
    V = x**2
    #V = 0 * x
    return V

def calcul_vp_vec_p(H):
    #---------------------------------------diagonalisation de la matrice H ---------------------------------#
    (vp, Tvec_p)=eigh(H)
    vec_p = np.transpose(Tvec_p)

    return (np.real(vp), vec_p)

def an(valeurs_p, vecteurs_p, psi_0):
    #calcul des coefficients par la méthode des rectangles

    an_psi = []
    for i in range(N_vp):
        an_psi.append(np.sum(np.conjugate(vecteurs_p[i]) * psi_0) * dx)

    return np.array(an_psi)


def animation_(An, vp, vec_p, V):
        #--------------------------animation ---------------------------------#

        fig = plt.figure() # initialise la figure
        line_proba, = plt.plot([], [], color = "blue", linewidth=3, label = "psi^2", zorder = 5)
        #line_real, = plt.plot([], [], color = "orange", label = "real")
        #line_imag, = plt.plot([], [], color = "red", label = "imag")
        plt.xlim(-Lx/2, Lx/2)
        plt.ylim(-0.01, 0.15)
        plt.xlabel("x")
        plt.ylabel("densité de probabilité")
        #plt.legend()

        #plt.title("paquet d'onde dans un potentiel avec trois puits carrés")

        plt.grid()
        plt.plot(x, V/300, color = "green")

        def animate(n, vp, vec_p, t):

            psi_i = np.zeros(Nb_x)
            for k in range(N_vp):
                psi_i = psi_i + An[k] * vec_p[k] * np.exp(-1j * vp[k] * t[n])
            psi_i = psi_i/np.sqrt(dx)

            print(np.sum(np.real(psi_i * np.conjugate(psi_i))))
            line_proba.set_data(x, np.sqrt(np.real(psi_i * np.conjugate(psi_i))))
            #line_real.set_data(x, np.real(psi_i))
            #line_imag.set_data(x, np.imag(psi_i))
            return line_proba,

        ani = animation.FuncAnimation(fig, animate, frames = Nb_t, fargs = (vp, vec_p,t),
                                      interval=1, blit=True)
        #writervideo = animation.FFMpegWriter(fps=50)
        #ani.save('triple_puit_1.mp4',writer=writervideo, dpi = 300)
        plt.show()

psi_0 = onde_initiale()

V = potentiel()
print("Potentiel initialization successed")

H = Hamiltonien()
print("Hamiltonien initialization successed")

vp, vec_p = calcul_vp_vec_p(H)
print("Eigenvalues and eigenvectors calculation successed")

N_vp = len(vp)
An = an(vp, vec_p, psi_0)
print("Initial coefficients calculation successed")

animation_(An, vp, vec_p, V)
#ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"