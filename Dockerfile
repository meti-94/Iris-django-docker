FROM ubuntu:18.04
RUN apt update && \
 apt install -y vim && \
 apt-get install -y python3-pip && \
 apt install gcc && \
 pip3 install uwsgi 

COPY ./iris_classification /site
RUN pip3 install -r /site/requirements.txt
COPY ./DocerUtilities/uwsgi.ini /etc/uwsgi/uwsgi.ini

COPY ./DocerUtilities/RUN.sh /RUN.sh
RUN chmod +x /RUN.sh

WORKDIR /site
CMD [ "/RUN.sh" ]
