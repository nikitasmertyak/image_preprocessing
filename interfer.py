import numpy as np
import cv2
import matplotlib.pyplot as plt 
img = cv2.imread('int_0_clr_still.tif')
print(img)
#Преобразовать в оттенки серого по формуле
img = img[:,:,2]

#Показать исходное изображение
plt.subplot(231),plt.imshow(img),plt.title('original')

#Выполните преобразование Фурье и отобразите результат
fft2 = np.fft.fft2(img)
plt.subplot(232),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')

#Переместите исходную точку преобразования изображения в центр прямоугольника частотной области и отобразите эффект.
shift2center = np.fft.fftshift(fft2)
plt.subplot(233),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')

#Логарифмически преобразовать результат преобразования Фурье и отобразить эффект
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')

#Выполните логарифмическое преобразование централизованного результата и отобразите результат
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')

plt.show()
