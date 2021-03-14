# builds Docker images from a Dockerfile "dockerfille_dev"
docker build -t mlflow .

# run container real_estat_price_container from image docker "mlflow" in the background 
docker run -d --name mlflow_contaier -p 5000:5000 mlflow

# Serveur MLflow:
http://0.0.0.0:5000


# Copy the update folder (mlruns) and Restart Service with container (in the background )
# os.system("docker cp dockerfil_MLflow/mlruns/. zen_nobel:/usr/app_mlflow/mlruns\
# && docker restart mlflow_contaier")