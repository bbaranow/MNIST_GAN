FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
RUN git clone https://github.com/bbaranow/MNIST_GAN.git