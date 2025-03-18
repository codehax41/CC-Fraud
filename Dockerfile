FROM python:3.11
WORKDIR /app
COPY . /app

RUN apt update -y

RUN apt-get update && pip install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]