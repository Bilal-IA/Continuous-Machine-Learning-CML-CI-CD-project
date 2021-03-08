# set base image 
FROM python:3.9-slim-buster
# copy the dependencies,flask_apy and ML-Pkl files to the working directory (automatic)
COPY . /usr/app/
EXPOSE 5000
# set the working directory in the container
WORKDIR /usr/app/
# install dependencies
RUN pip install -r requirements.txt
# command to run on container start
CMD python flask_api.py