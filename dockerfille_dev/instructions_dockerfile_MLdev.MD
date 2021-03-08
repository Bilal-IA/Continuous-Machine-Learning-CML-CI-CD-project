# builds Docker images from a Dockerfile "dockerfille_dev"
docker build -t real_estat_price_api .

# run container real_estat_price_container from image docker "real_estat_price_api" in the background 
docker run -d --name real_estat_price_container -p 8000:8000 real_estat_price_api


# Serveur MLDev:
http://0.0.0.0:8000/apidocs

# Copy the update files (Ml model and Flask web Api) and Restart Service with container
# docker cp flask_api.py real_estat_price_container:/usr/app/flask_api.py && docker cp model.pkl real_estat_price_container:/usr/app/model.pkl &&  docker restart real_estat_price_container