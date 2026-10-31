import imageio.v3 as image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Membaca citra menggunakan imageio
path = "mobill.jpg"
img = image.imread(path)

# Mengonversi ke citra grayscale menggunakan numpy (jika citra asli RGB)
if len(img.shape) == 3:
    img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
else:
    img_gray = img

# Peningkatan Kualitas Citra menggunakan scipy.ndimage
# Menerapkan median filter dengan ukuran matriks tetangga 3x3 untuk mengurangi noise
img_filtered = ndimage.median_filter(img_gray, size=3)

# Menampilkan citra awal dan citra yang ditingkatkan
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Citra Awal (Grayscale)')

plt.subplot(1, 2, 2)
plt.imshow(img_filtered, cmap='gray')
plt.title('Citra Ditingkatkan (Median Filter)')

plt.tight_layout()
plt.show()