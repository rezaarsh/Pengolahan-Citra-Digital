
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra
img = cv2.imread('cuking.jpg')

# Memastikan citra berhasil dibaca
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan!")

# Mengubah format warna BGR menjadi RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mendapatkan ukuran citra
tinggi, lebar = img.shape[:2]

# 1. TRANSLATION
M_trans = np.float32([
    [1, 0, 100],
    [0, 1, 50]
])

translation = cv2.warpAffine(
    img, M_trans, (lebar, tinggi)
)

# 2. SCALING
scaling_05 = cv2.resize(
    img,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_LINEAR
)

scaling_2 = cv2.resize(
    img,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_LINEAR
)

# 3. ROTATION
def rotasi_citra(img, sudut):
    tinggi, lebar = img.shape[:2]

    pusat = (lebar // 2, tinggi // 2)

    matriks = cv2.getRotationMatrix2D(
        pusat, sudut, 1.0
    )

    return cv2.warpAffine(
        img, matriks, (lebar, tinggi)
    )

rotasi_45 = rotasi_citra(img, 45)
rotasi_90 = rotasi_citra(img, 90)
rotasi_180 = rotasi_citra(img, 180)

# # 4. FLIP
horizontal_flip = cv2.flip(img, 1)
vertical_flip = cv2.flip(img, 0)

# 5. MENAMPILKAN HASIL TRANSFORMASI

hasil = [
    ("Citra Asli", img),
    ("Translation", translation),
    ("Scaling 0.5", scaling_05),
    ("Scaling 2", scaling_2),
    ("Rotation 45 Derajat", rotasi_45),
    ("Rotation 90 Derajat", rotasi_90),
    ("Rotation 180 Derajat", rotasi_180),
    ("Horizontal Flip", horizontal_flip),
    ("Vertical Flip", vertical_flip)
]

# Membuat subplot
plt.figure(figsize=(15, 12))

for i, (judul, gambar) in enumerate(hasil):
    plt.subplot(3, 3, i + 1)
    plt.imshow(gambar)
    plt.title(judul)
    plt.axis('off')

plt.tight_layout()
plt.show()
