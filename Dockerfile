FROM python:3.7.2-slim

MAINTAINER "aadesh.shendge@yahoo.com"

COPY . /src

WORKDIR /src

RUN pip install -r requirements.txt

#CMD tail -F anything

CMD echo "run any .py file for example python src/miners/imdb_miner.py"