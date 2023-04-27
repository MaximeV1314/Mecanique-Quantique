# mecanique-quantique
Dossier de simulations quantiques

Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img_dynamique".

Exécuter le code. Une fois finit, ouvrir le cmd depuis le dossier initial et taper  :
    ffmpeg -r 30 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

une vidéo .mp4 devrait être créée.
