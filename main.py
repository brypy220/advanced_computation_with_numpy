import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
from scipy.datasets import face
from PIL import Image, ImageOps
import pprint as pp

#642
# my_array = np.array([1.1, 9.2, 8.1, 4.7])
# print(my_array.shape)
# print(my_array[2])
# print(my_array.ndim)
#
# array_2d = np.array([[1, 2, 3, 9], [5, 6, 7, 8]])
#
# print(f'array_2d has {array_2d.ndim} dimensions')
# print(f'Its shape is {array_2d.shape}')
# print(f'It has {array_2d.shape[0]} rows and {array_2d.shape[1]} columns')
# print(array_2d)
# print(array_2d[1, 2])
# print(array_2d[1, :])
#
# mystery_array = np.array([[[0, 1, 2, 3],
#                           [4, 5, 6, 7]],
#
#                           [[7, 86, 6, 98],
#                            [5, 1, 0, 4]],
#
#                           [[5, 36, 32, 48],
#                            [97, 0, 27, 18]]])
#
# # print(f' mystery_array has {mystery_array.ndim} dimensions\n Its shape is {mystery_array.shape}.\n The value in the '
# #       f'last line of code is {mystery_array[2,1,3]}\n A 1-dimensional vector is {mystery_array[2,1,:]}\n A (3,2) '
# #       f'matrix is{mystery_array[:, :, 0]}')
#
# pp = pprint.PrettyPrinter(indent=4)
# print(f'We have {mystery_array.ndim} dimensions')
# print(f'The shape is {mystery_array.shape}')
# print(f'The value in the last line of code is {mystery_array[2, 1, 3]}')
# pp.pprint(mystery_array[2, 1])
# pp.pprint(mystery_array[:, :, 0])
#
# #643
#
# a = np.arange(10, 30)
# print(a)
#
# # only the last 3
# print(a[-3:])
#
# # only the 4th, 5th, and 6th numbers
# print(a[3:6])
#
# # omit the first 12
# print(a[12:])
#
# # even numbers
# print(a[::2])
#
# # reverse the list
# print(a[::-1])
#
# # print out indices of the non-zero elements in this array:
# array_0 = np.array([6, 0, 9, 0, 0, 5, 0])
# array_not0 = np.nonzero(array_0)
# # array_not0 = np.delete(array_0, (1, 3, 4, 6))
# # print(array_not0)
# pp.pprint(array_not0)
#
# z = random((3, 3, 3))
# print(z)
#
# # OR you can write the full path aka without an import statement
# # z = np.random.random((3, 3, 3))
# # print(z.shape)
# # print(z)
#
# # todo  Challenge 7
# # x = np.linspace(0, 100, num=9)
# # print(x)
# # print(x.shape)
# #
# # y = np.linspace(start=-3, stop=3, num=9)
# # plt.plot(x, y)
# # # plt.show()
#
# # todo Generate an array called noise with shape 128x128x3 that has random values, then use matplotlib's .imshow() to
# # display the array as an image.
#
# noise = np.random.random((128, 128, 3))
# print(noise.shape)
# plt.imshow(noise)
# # plt.show()

# todo 644 Broadcasting, Scalars, and Matrix Multiplication

v1 = np.array([4, 5, 2, 7])
v2 = np.array([2, 1, 3, 3])

pp.pprint(v1 + v2)

pp.pprint(v1 * v2)

# Broacasting

array_2d = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8]])

pp.pprint(array_2d + 10)

# Matrix Multiplication

a1 = np.array([
    [1, 3],
    [0, 1],
    [6, 2],
    [9, 7]
])

b1 = np.array([
    [4, 1, 3],
    [5, 8, 5]
])
pp.pprint(a1 @ b1)
c = np.matmul(a1, b1)
pp.pprint(c)

# Manipulating Images as ndarrays
"""Make sure to write 'plt.show() underneath the code you want to display'"""
img = face()
plt.imshow(img)
# plt.show()

# pp.pprint(type(img))
# pp.pprint(img.shape)
# pp.pprint(img.ndim)
# print(img.size)
#
# # todo Grayscale Image
sRGB_array = img / 255

grey_vals = np.array([0.2126, 0.7152, 0.0722])

img_gray = sRGB_array @ grey_vals
# img_gray = np.matmul(sRGB_array, grey_vals)
gray_img = np.matmul(sRGB_array, grey_vals)
plt.imshow(gray_img, cmap='gray')
plt.show()
# # todo Flip the grayscale image upside down
# plt.imshow(img_gray, origin='lower', cmap='gray')
# # or you can reverse the order of the rows and the columns in the NumPy array with the .flip() function
# plt.imshow(np.flip(img_gray), cmap='gray')

# # todo rotate the color image:
#
# angle = 90
# new_img = ndimage.rotate(img, angle, reshape=True)
# plt.imshow(new_img)
# or you can rotate the array with .rot90()

#plt.imshow(np.rot90(img))
#
# # todo Invert(solarize) the color image.
# sRGB_array = img * 255
# # or sRGB_array = 255 - img
# invert_vals = np.array([0.2126, 0.7152, 0.0722])
#
# solarized_img = sRGB_array @ invert_vals
#
# plt.imshow(solarized_img)
# solarized_img = 255 - img # or 255 * img, since we divided earlier we logically can do the opposite.
# plt.imshow(solarized_img)

# using an Image of Lord Nikon
# file = 'lord_nikon.jpeg'
# my_img = Image.open(file)
# img_array = np.array(my_img)
# pp.pprint(img_array.ndim)
# pp.pprint(img_array.shape)
# plt.imshow(img_array)
# plt.imshow(img_array * 255)
# plt.show()