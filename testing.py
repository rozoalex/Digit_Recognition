

from os import listdir

import skimage.transform
import numpy
import skimage.io
numpy.set_printoptions(threshold=numpy.nan)




files = [f for f in listdir('.')]  # if os.path.isfile(f)
input_imgs = numpy.array([])
filenames = []
for f in files:
    print('processing file: ' + f)
    if f.endswith(".png"):
        arr = skimage.io.imread(f, as_grey=True)
        resized_arr = skimage.transform.resize(arr, (28 ,28))
        print("Image detected, name: " + f)
        print(arr.shape)
        print(resized_arr.shape)
        filenames.append(f)
        if input_imgs.size == 0:
            input_imgs = resized_arr.flatten('C')
        else:
            input_imgs = numpy.vstack((input_imgs, resized_arr.flatten('C')))
        #print(resized_arr.flatten().shape)
        print(input_imgs.shape)
        print("\n")
print(filenames)