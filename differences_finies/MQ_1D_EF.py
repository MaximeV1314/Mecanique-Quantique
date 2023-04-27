import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

"""
Code renvoyant une animation de la dynamique d'un paquet d'ondes gaussien dans le potentiel choisi en utilisant le schéma numérique Runge-Kutta à l'ordre 4
et renvoyant la norme au carré sur tout l'espace ( = 1 normalement) de l'onde pour différent temps d'étude.

/!\ Si l'animation ne se lance pas au bout de quelques minutes ou qu'il y a un message d'erreur, lancez
    le code via l'invité de commande.
"""

#############################################################
######################  paramètres   ########################
#############################################################

L = 15                              #longueur boîte 1D
Nb_x = 200                          #nombre de points de discrétisation en longueur
x = np.linspace(-L/2, L/2, Nb_x)
dx = L/(Nb_x-1)                     #pas longueur

kx = 2                              #impulsion
x0 = 0                              #position initial du paquet d'onde
F = 1                               #force pour l'oscillation de Bloch
sigma = 0.5                         #largeur à mi-hauteur du packet d'onde gaussien initial

tf = 15                             #temps d'étude
dt = 0.001                          #pas de temps
t = np.arange(0, tf+dt, dt)
Nb_t = len(t)                       #nombre de points de discrétisation en temps
pas_img = 10                        #pas du nombre d'image

#############################################################
######################  fonctions  ########################
#############################################################

def onde_initiale(x):
    ####################   paquet d'onde initial     ###################
    onde_init = np.exp(-((x-x0)/(2*sigma))**2) * np.exp(1j * kx * x) * ( 2 * np.pi * sigma**2)**(-1/4) #paquet d'ondes gaussien
    return onde_init/np.linalg.norm(onde_init)

def matrice2(V):

    ############   construction de la matrice dynamique   ###############

    vec = np.ones(Nb_x)
    A = (- 1j/dx**2 - 1j * V) * np.diag(vec)\
     + 1j/(2*dx**2) * np.diag(vec[:-1], -1)\
     + 1j/(2*dx**2) * np.diag(vec[:-1], 1)

    return A

def potentiel(x):

    #########    fonctions potentiels      ##########

    #return np.zeros(Nb_x, dtype = complex)             #potentiel carré

    #V = np.zeros(Nb_x)                                  #potentiel double carré
    #V[int(Nb_x/2) - 30:int(Nb_x/2) + 30] = 5  #
    #return V

    return x**2                                     #harmonique

    #return 0.2 * x**4 - 2 * x**2 + 5                    #potentiel double puit

def euler(x):

    #########    résolution par la méthode d'Euler       ############

    V = potentiel(x)
    A = matrice2(V)
    psi_t0 = onde_initiale(x)

    psi_tn = psi_t0                         #initialisation
    psi = [psi_t0]                          #liste des fonctions ondes en tout temps

    for i in range(Nb_t):
        psi_tn = A @ psi_tn * dt + psi_tn   #schéma d'Euler

        if i%pas_img==0:
            psi.append(psi_tn)

    return psi

def rk4(x):

    #######      résolution par Runge-Kutta 4      ########

    V = potentiel(x)
    A = matrice2(V)
    psi_t0 = onde_initiale(x)

    psi_tn = psi_t0
    psi = [psi_t0]

    for i in range(Nb_t):
        d1 = psi_tn
        d2 = psi_tn + dt/2 * A @ psi_tn
        d3 = psi_tn + dt/2 * A @ d2
        d4 = psi_tn + dt * A @ d3
        psi_tn = psi_tn + dt / 6 * (A @ d1 + 2 * A @ (d2 + d3) + A @ d4)    #schéma de RK4

        if i%pas_img==0:
            psi.append(psi_tn)

    return psi

def norme(psi):

    ########     vérification de la norme des ondes    #########

    for tn in range(0, len(psi), 10):
        print("Norme t = ", t[tn], " : ", np.linalg.norm(psi[tn]))


def graphique(x, psi):

    ########     animation       #########

    fig = plt.figure() # initialise la figure
    line_proba, = plt.plot([], [], color = "black", label = "|psi|^2", linewidth=3, zorder = 10)
    line_real, = plt.plot([], [], color = "blue", label = "real")
    line_imag, = plt.plot([], [], color = "red", label = "imag")
    plt.xlim(-L/2, L/2)
    plt.ylim(-0.1, 0.1)
    plt.xlabel("x")
    plt.ylabel("densité de probabilité")
    plt.legend()

    #plt.title("paquet d'ondes dans un double puit de potentiel (carré)")

    plt.grid()
    plt.plot(x, potentiel(x)/30 - 0.08, color = "green")

    def animate(i):

        global x, psi
        line_proba.set_data(x, np.real(psi[i] * np.conjugate(psi[i])))
        line_real.set_data(x, np.real(psi[i]))
        line_imag.set_data(x, np.imag(psi[i]))
        return line_proba, line_real, line_imag

    ani = animation.FuncAnimation(fig, animate, frames=int((Nb_t)/pas_img),
                                  interval=10, blit=True)
    #writervideo = animation.FFMpegWriter(fps=50)
    #ani.save('potentiel_double_puit_carre.mp4',writer=writervideo, dpi = 300)
    plt.show()


psi = rk4(x)
print("RK4 : successed")
graphique(x, psi)
print("Animation : successed")
norme(psi)

