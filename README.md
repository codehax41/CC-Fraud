## Run from terminal:

docker build -t creditcard.azurecr.io/cc:latest .

docker login creditcard.azurecr.io

docker push creditcard.azurecr.io/cc:latest
