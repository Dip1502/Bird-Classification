FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

