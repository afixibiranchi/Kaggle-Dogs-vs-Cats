import glob
import numpy as np
import os
import random
from skimage import color, io
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image
from PIL import ImageChops

def returner():
    files_path='path to the "trainingSet"'

    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')
    cat_files = sorted(glob.glob(cat_files_path))
    dog_files = sorted(glob.glob(dog_files_path))

    n_files = len(cat_files) + len(dog_files)
    print(n_files)

    size = [128,128]
    size_image = 128

    allX = np.zeros((n_files, size_image, size_image), dtype='float64')
    allY = np.zeros(n_files)

    count = 0
    for f in cat_files:
    
   
        try:

            image = Image.open(f)
            image.thumbnail(size, Image.ANTIALIAS)
            greyscale = image.convert('L')
            image_size = greyscale.size
            thumb = greyscale.crop( (0, 0, size[0], size[1]) )
            offset_x = max( (size[0] - image_size[0]) / 2, 0 )
            offset_y = max( (size[1] - image_size[1]) / 2, 0 )
            thumb = ImageChops.offset(thumb, int(offset_x), int(offset_y))  
            data = np.asarray(thumb)
            new_img = imresize(data, (size_image, size_image))

            allX[count] = np.array(new_img)
            allY[count] = 0
            count += 1

        except:
            continue


    for f in dog_files:
        

        try:
            image = Image.open(f)
            image.thumbnail(size, Image.ANTIALIAS)
            greyscale = image.convert('L')
            image_size = greyscale.size
            thumb = greyscale.crop( (0, 0, size[0], size[1]) )
            offset_x = max( (size[0] - image_size[0]) / 2, 0 )
            offset_y = max( (size[1] - image_size[1]) / 2, 0 )
            thumb = ImageChops.offset(thumb, int(offset_x), int(offset_y))  
            data = np.asarray(thumb)
            new_img = imresize(data, (size_image, size_image))

            allX[count] = np.array(new_img)
            allY[count] = 1
            count += 1
        except:
            continue


    X, X_test, Y, Y_test = train_test_split(allX, allY, test_size=0.2, random_state=42)


    

    return X, X_test, Y, Y_test


