import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

from model import DenoiseAutoEncoder
from utils import plot_noise_images, model_reconstruct_images, visualize_compressed

def main():
    np.random.seed(11)
    tf.random.set_seed(11)
    keras.backend.clear_session()

    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train, X_valid, X_test = X_train[5000:] / 255., X_train[:5000] / 255., X_test / 255.
    y_train, y_valid = y_train[5000:], y_train[:5000]

    noise_factor = 0.2
    X_train_noisy = np.clip(X_train[..., np.newaxis] + noise_factor * np.random.rand(*X_train[..., np.newaxis].shape), 0, 1)
    X_valid_noisy = np.clip(X_valid[..., np.newaxis] + noise_factor * np.random.rand(*X_valid[..., np.newaxis].shape), 0, 1)
    X_test_noisy = np.clip(X_test[..., np.newaxis] + noise_factor * np.random.rand(*X_test[..., np.newaxis].shape), 0, 1)

    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    cluster_labels = [f"Cluster {i}" for i in range(10)]

    plot_noise_images(X_train, X_train_noisy, "Original and Noisy Training Images")

    image_shape = X_train_noisy.shape[1:]
    model = DenoiseAutoEncoder(image_shape)

    model_reconstruct_images(model, X_train_noisy, model_status='before')

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(X_train_noisy, X_train, epochs=30, shuffle=True,
                        validation_data=(X_valid_noisy, X_valid), verbose=1)

    pd.DataFrame(history.history).plot()
    plt.grid()
    plt.show()

    model_reconstruct_images(model, X_train_noisy, model_status='after')

    # Gaussian blur for test noisy images (optional visualization)
    X_test_blurred = np.array([gaussian_filter(img.squeeze(), sigma=1) for img in X_test_noisy])
    X_test_blurred = X_test_blurred[..., np.newaxis]

    plot_noise_images(X_test_noisy, X_test_blurred, "Noisy Test Images and Gaussian Blurred Versions")

    # Visualize latent space with t-SNE
    X_val_noise_pred = model.encoder_.predict(X_valid_noisy).reshape(X_valid.shape[0], -1)
    X_val_noise_tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200).fit_transform(X_val_noise_pred)
    visualize_compressed(X_val_noise_tsne, y_valid, X_valid, result='Autoencoder_Results', class_names=cluster_labels)

    denoised = model.predict(X_test_noisy, verbose=0)
    avg_psnr = np.mean([psnr(X_test[i].squeeze(), denoised[i].squeeze(), data_range=1.0) for i in range(len(X_test))])
    avg_ssim = np.mean([ssim(X_test[i].squeeze(), denoised[i].squeeze(), data_range=1.0) for i in range(len(X_test))])

    print(f"Average PSNR > 30 (preferable and indicates excellent denoising): {avg_psnr:.4f}")
    print(f"Average SSIM ~ 1 (Closer to 1.0 is ideal, reflects structural similarity): {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
