# builds Docker images from a Dockerfile "dockerfille_git"
docker build -t git-image . 

# run container real_estat_price_container from image docker "git-image" in the background 
docker run -it git-image bash

# HTTPS with Access Token method (BILAL-IA)
RUN git config --global url.”https://de0c5eb978257e0a7ca618607b857999ab749be8:@github.com/".insteadOf “https://github.com/"

# version git
git --version


# run container real_estat_price_container from image docker "real_estat_price_api" in the background 
docker run -d --name real_estat_price_container -p 8000:8000 real_estat_price_api

docker run -p  5000:5000 -v $(pwd):/home/bilal/Documents/AI-AEC/"A57 Mise en place écosystème d'IA"/projets/projet_02_mlflow --name stupefied_pare




# Run mlflow
docker run -p 5000:5000 -v /home/bilal/projet_02_mlflow:/home/bilal/projet_02_mlflow mlflow
