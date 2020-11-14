from matplotlib import pyplot as plt
import numpy as np

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def plot_loss(gl, dl, epoch):
    plt.figure(figsize=(16,2))
    plt.plot(np.arange(len(gl)),gl,label='Gen_loss')
    plt.plot(np.arange(len(dl)),dl,label='Disc_loss')
    plt.legend()
    plt.title('Epoch '+str(epoch)+' Loss')
    ymax = plt.ylim()[1]
    plt.show()

def plot_all_time_lostt(all_gl, all_dl):

    plt.figure(figsize=(16,2))
    plt.plot(np.arange(len(all_gl)),all_gl,label='Gen_loss')
    plt.plot(np.arange(len(all_dl)),all_dl,label='Disc_loss')
    plt.legend()
    plt.ylim((0,np.min([1.1*np.max(all_gl),2*ymax])))
    plt.title('All Time Loss')
    plt.show()