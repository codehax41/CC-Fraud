## Run from terminal:

docker build -t creditcard.azurecr.io/mltest:latest .

docker login creditcard.azurecr.io

docker push creditcard.azurecr.io/mltest:latest