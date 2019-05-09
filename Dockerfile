FROM python:3.7

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt &&\
    python3 -m pytest -s test/test_shakespeare_tf_aio.py
