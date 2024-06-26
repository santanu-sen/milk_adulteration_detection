FROM python:3.11.4-slim-buster

RUN apt update -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt
EXPOSE 80

CMD ["python", "app.py"]
