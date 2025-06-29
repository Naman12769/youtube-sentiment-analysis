FROM python:3.8.5-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

CMD ["python3","app.py"]
