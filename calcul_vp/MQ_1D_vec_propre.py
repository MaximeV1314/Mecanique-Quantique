import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
import matplotlib as mpl
from matplotlib.pyplot import cm
mpl.rcParams['animation.ffmpeg_path'] = r'C:\FFMPEG\bin\ffmpeg.exe'     #donner le chemin vers ffmpeg.exe
plt.rcParams['text.usetex'] = True

#############################################################
######################  paramètres   ########################
#############################################################

Lx = 70                              #longueur
Nb_x = 1000                          #nombre de points de discrétisation en longueur
x = np.linspace(-Lx/2, Lx/2, Nb_x)
dx = Lx/(Nb_x-1)                     #pas longueur

x0 = -10
kx = 3

sigma = 0.8                        #largeur à mi-hauteur du packet d'onde gaussien initial

tf = 20                             #temps d'étude
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
    #V = x**2
    #V = 0 * x

    V = 0*x
    V[485:515] = np.ones(30) * 5
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

def graphique_valeurs_propres(valeurs_propres, vecteurs_propres):
        #--------------------------plot des valeurs propres de l'Hamiltonien ---------------------------------#

        plt.plot(np.arange(0, N_vp, 1), np.sort(valeurs_propres) , ".", color = "blue", label = "valeurs propres calculees")
        #plt.plot(np.arange(0, N_vp, 1), np.sqrt(2) * (np.arange(0, N_vp, 1) + 1/2), "-", color = "red", label = "valeurs propres theoriques")    #vp harmonique théorique
        plt.plot(np.arange(0, N_vp, 1), np.arange(0, N_vp, 1)**2 * np.pi**2/(2 * (Lx)**2), "-", color = "red", label = "valeurs propres theoriques")
        #vp boîte théorique


        #plt.plot(np.arange(0, len(valeurs_propres), 1), 5 * np.ones(len(valeurs_propres)), "-.r", label = "V0") #V0 pour le potentiel double puit et double puit carré
        #plt.title("Graphique des valeurs propres")
        plt.grid()
        plt.xlabel("valeur propre N°")
        plt.ylabel("E (u.a.)")
        plt.legend()
        plt.show()

def graphique_vecteurs_propres(valeurs_propres, vecteurs_propres):
    #--------------------------plot des vecteurs propres de l'hamiltonien ---------------------------------#
    index_liste = np.argsort(valeurs_propres)

    for i in range(0,10):
        plt.plot(x, np.real(vecteurs_propres[index_liste[i]])+0.2*valeurs_propres[index_liste[i]], color = cm.viridis(i/10)) #vec. prop. oh
        #plt.plot(x, np.real(vecteurs_propres[index_liste[i]] + 1.5*np.sqrt(valeurs_propres[index_liste[i]])), color = cm.viridis(i/8))       #vec.prop. carré
    #plt.plot(np.linspace(-self.paquet_onde.L/2, self.paquet_onde.L/2, len(valeurs_propres)), np.sqrt(5) * np.ones(len(valeurs_propres)), "-.r", label = "V0") #V0 carré

    #plt.title("Fonction propres séparées par leurs valeurs propres associées")
    plt.grid()
    plt.show()

def animation_(An, vp, vec_p, V):
        #--------------------------animation ---------------------------------#

        fig = plt.figure() # initialise la figure
        line_proba, = plt.plot([], [], color = "black", linewidth=3, label = "$$|\psi|^2$$", zorder = 5)
        line_real, = plt.plot([], [], color = "blue", label = "$$Re(\psi)$$")
        line_imag, = plt.plot([], [], color = "red", label = "$$Im(\psi)$$")
        plt.xlim(-Lx/2, Lx/2)
        plt.ylim(-0.1, 0.15)
        #plt.xlabel("")
        #plt.ylabel("$$|\psi|^2$$")
        plt.legend(loc = "upper left")

        #plt.title("paquet d'onde dans un potentiel avec trois puits carrés")

        plt.grid()
        plt.plot(x, V/50, color = "green")

        def animate(n, vp, vec_p, t):

            psi_i = np.zeros(Nb_x)
            for k in range(N_vp):
                psi_i = psi_i + An[k] * vec_p[k] * np.exp(-1j * vp[k] * t[n])
            psi_i = psi_i/np.sqrt(dx)

            #print(np.sum(np.real(psi_i * np.conjugate(psi_i))))
            line_proba.set_data(x, 5*np.real(psi_i * np.conjugate(psi_i)))
            line_real.set_data(x, np.real(psi_i))
            line_imag.set_data(x, np.imag(psi_i))
            return line_proba, line_real, line_imag

        ani = animation.FuncAnimation(fig, animate, frames = Nb_t, fargs = (vp, vec_p,t),
                                      interval=1, blit=True)
        writervideo = animation.FFMpegWriter(fps=50)
        ani.save('ef_1D_2.mp4',writer=writervideo, dpi = 300)
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
#graphique_valeurs_propres(vp, vec_p)
#graphique_vecteurs_propres(vp, vec_p)