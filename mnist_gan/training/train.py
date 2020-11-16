import tensorflow as tf
import time
import numpy as np
from mnist_gan.utils.visualization_utils import generate_and_save_images, plot_loss

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, BATCH_SIZE, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(dataset, epochs, BATCH_SIZE, noise_dim, generator, discriminator, seed):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    all_gl = np.array([]); all_dl = np.array([])
    for epoch in range(epochs):
        gl = []; dl = []
        start = time.time()

        for image_batch in dataset:
            gg, dd = train_step(image_batch, BATCH_SIZE, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer)
            gl.append(gg); dl.append(dd)

        # To produce images for the GIF
        all_gl = np.append(all_gl,np.array([gl]))
        all_dl = np.append(all_dl,np.array([dl]))
  
        generate_and_save_images(generator, epoch + 1,seed)
        plot_loss(gl, dl, epoch + 1)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  #To generate after the final epoch
    generate_and_save_images(generator, epochs, seed)

    filename = 'models/generator.h5'
    tf.keras.models.save_model(
        generator,
        filename,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )
