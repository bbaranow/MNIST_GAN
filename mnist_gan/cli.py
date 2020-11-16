import click

@click.command()
@click.option('--epochs', default=1, prompt='Number of epochs')
@click.option('--num_examples', default=1, prompt='Number of examples')
def do_train(epochs, num_examples):
    import tensorflow as tf
    from mnist_gan.data.make_dataset import download_dataset, preprocess_dataset, make_tf_dataset
    from mnist_gan.training.models import make_generator_model, make_discriminator_model
    from mnist_gan.training.train import train

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    epochs = epochs
    noise_dim = 100
    num_examples_to_generate = num_examples
    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train_images, _ = download_dataset()
    train_images = preprocess_dataset(train_images)
    tf_train_images = make_tf_dataset(train_images)
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    train(tf_train_images, epochs, 256, 100, generator, discriminator, seed)

@click.command()
@click.option('--num_examples', default=1, prompt='Number of examples')
def generate(num_examples):
    import tensorflow as tf
    from matplotlib import pyplot as plt

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    noise_dim = 100
    num_examples = num_examples

    model = tf.keras.models.load_model('models/generator.h5')
    seed = tf.random.normal([num_examples, noise_dim])
    model.compile()
    predictions = model(seed, training=False)
    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('reports/generated_examples.png')


