FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${PATH}:/root/.poetry/bin"
RUN git clone https://github.com/bbaranow/MNIST_GAN.git
RUN cd /tf/MNIST_GAN && poetry build
RUN pip install /tf/MNIST_GAN/dist/*whl
RUN pip install papermill