FROM ubuntu:18.04
RUN apt update && \
 apt install -y vim && \
 apt install -y python3.6 && \
 apt-get install -y python3-pip
copy iris_classification /iris_classification
copy requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
CMD [ "python3", "/iris_classification/manage.py", "runserver", "0.0.0.0:8000" ]
