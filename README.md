# mecanique-quantique
Dossier de simulations quantiques

Le premier dossier calcule une évolution dynamique (1D ou 2D) en utilisant la méthode des différences finies (avec la méthode d'Euler ou de Runge-Kutta à l'ordre 4 (défaut)). La MDF permet de résoudre plus rapidement l'équation de Schrödinger pour un hamiltonien dépendant du temps, mais est tout de même très lent dès que l'on diiminue le as d'espace. Le second dossier calcul les vecteurs et valeurs propres de l'hamiltonien (1D ou 2D) en passant par une discrétisation de l'espace. Si l'on veut étudier un hamiltonien particulier, alors nous pouvons stocker les valeurs et vecteurs propres dans un dossier (à implémenter dans le code ) et les utiliser sans avoir à les calculer à chaque étude d'un système dynamique. Cet algorythme permet de visualiser les ondes propres de l'hamiltonien (soit en animation (pour le 2D), soit en images (pour la 1D et la 2D)) en  mais n'est pas compatible pour un hamiltonien dépendant du temps et n'étudie pas la dynamique.

Le potentiel peut être choisi directement dans le code dans la fonction "Potential". On peut aussi créer un potentiel V(X, Y) directement dans la fonction.

Les conditions aux bords sont des conditions de Dirichlet.

Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img_dynamique".
Exécuter le code. Une fois fini, ouvrir le cmd depuis le dossier initial et taper  :
    ffmpeg -r 30 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
une vidéo .mp4 devrait être créée.

Améliorations prévues :
- implémenter la possibilité d'enregistrer les valeurs et vecteurs propres,
- implémenter des conditions périodiques,
- créer une interface,
