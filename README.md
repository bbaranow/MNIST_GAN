MNIST_GAN
==============================

This project is created to show how to "productionalize" code from the python notebook.



## Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── docs                                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── MNIST_GAN           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                                <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Scripts to train models
    │   │
    │   ├── validate                            <- Scripts to validate models
    │   │
    │   └── visualization                       <- Scripts to create exploratory and results oriented visualizations

## Command-line interface

To train model and generate random images:
1. Make sure you have docker installed and NVIDIA GPU support:  https://www.tensorflow.org/install/docker
1. Pull and run docker image
    ```
    docker pull ganbbaranow/tf-gan
    docker run -it --rm -p 8888:8888 --gpus all --network=host ganbbaranow/tf-gan /bin/bash
    ```

1. Go to `/tf/MNIST_GAN` directory
1. Run a model by running the command
    ```
    train_model
    ```
   The model will be saved in `models/` directory
1. Generate new images by running the command
    ```
    generate_images
    ```
    The images will be generated under `reports/` directory as `generated_examples.png`
    
