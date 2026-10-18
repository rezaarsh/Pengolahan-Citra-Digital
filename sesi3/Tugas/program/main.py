
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

nama_file = "masgerung.jpg"

# 1. Membaca citra
img = imageio.imread(nama_file)
print("Citra awal:", img.shape)

# Memastikan gambar memiliki kanal RGB
if img.ndim != 3 or img.shape[2] < 3:
    raise ValueError("Gambar harus memiliki minimal 3 kanal RGB.")

img = img[:, :, :3]

# 2. Mengubah citra RGB ke grayscale
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

img_gray = (
    0.299 * R +
    0.587 * G +
    0.114 * B
).astype(np.uint8)

print("Citra grayscale:", img_gray.shape)

# 3. Menghitung jumlah piksel
tinggi, lebar = img_gray.shape
jumlah_piksel = img_gray.size

print("Tinggi:", tinggi, "piksel")
print("Lebar:", lebar, "piksel")
print("Jumlah piksel:", jumlah_piksel)

# 4. Menghitung histogram intensitas 0-255
histogram = np.bincount(
    img_gray.ravel(),
    minlength=256
)

# 5. Menampilkan jumlah piksel setiap intensitas
for intensitas in range(256):
    print(
        f"Intensitas {intensitas:3d}: "
        f"{histogram[intensitas]}"
    )

# 6. Mencari intensitas dominan
intensitas_dominan = np.argmax(histogram)
jumlah_intensitas = histogram[intensitas_dominan]

print("Intensitas dominan:", intensitas_dominan)
print("Jumlah piksel pada intensitas dominan:", jumlah_intensitas)

# Memastikan total histogram sama dengan jumlah piksel
print("Total histogram:", np.sum(histogram))
assert np.sum(histogram) == jumlah_piksel

# 7. Menampilkan citra grayscale dan histogram
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.bar(range(256), histogram, width=1, color="gray")
plt.title("Histogram Citra Grayscale")
plt.xlabel("Nilai Intensitas")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)

plt.tight_layout()
plt.show()
