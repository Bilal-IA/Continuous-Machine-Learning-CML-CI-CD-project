FROM python:3.8.5

#LABEL maintainer "Bilal khomri<bilal.khomri@gmail.com>"

#RUN mkdir /mlflow/

# copy the dependencies,flask_apy and ML-Pkl files to the working directory (automatic)
COPY . /usr/app_mlflow/

RUN pip install mlflow==1.14

EXPOSE 5000
# set the working directory in the container
WORKDIR /usr/app_mlflow/

#CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts  --host 0.0.0.0
CMD  bash -c "mlflow server --host 0.0.0.0"