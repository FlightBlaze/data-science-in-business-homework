FROM python:3.7
LABEL maintainer="flightblaze@gmail.com"
COPY ./server /
WORKDIR /
RUN pip install -r requirements.txt
EXPOSE 8180
EXPOSE 8181
VOLUME /server/saved
COPY ./docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]