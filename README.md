# 512FruitApiCaller

Docker Setup: 
download the following files to a project folder in an IDE:
Dockerfile, Inceptionv3_Model.keras, .dockerignore, VGGFinal.keras, gunicorn.conf.py, main.py, and requirements.txt 
use the command 
“docker build —tag food_grader .” 
in the project folder in the terminal to build the docker image. 
Uploading takes a few steps but I had to login to docker through azure in the terminal and tag the image using
“docker tag food_grader foodgrader.azurecr.io/food_grader:latest”
and push it using
“docker push foodgrader.azurecr.io/food_grader:latest”. 
that may be more than you need and there’s more configuration stuff on azure that needed to happen which I don’t think is necessary to include but I can give it to you if you need.

Caller Setup: 
Download both the Color Mask.py and Quadrents.py files and open them in an ide that has access to your live camera (I was using Pycharm) 
Edit requests call on line 28 and 39 respectively to have your API link instead of the work link
After that the program should be good to run

