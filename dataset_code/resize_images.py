import urllib
from joblib import Parallel, delayed
from PIL import Image
import os

def resize(im, minSize):
    w = im.size[0]
    h = im.size[1]
    if w < h:
        new_width = minSize
        new_height = int(minSize * (float(h) / w))
    if h <= w:
        new_height = minSize
        new_width = int(minSize * (float(w) / h))
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    return im

def resize_dataset(id):
    try:
        img = Image.open(dataset_root + 'img/' + id)
        img = resize(img, 300)
        image_path = dest_path + str(id) 
        directory = dest_path + str(id).split('/')[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        img.save(image_path)
    except:
        print("ERROR")


dataset_root = '/media/ssd2/YFCC100M-GEO100/'
dest_path = dataset_root + 'img_resized/'

img_names = []
for line in open(dataset_root + 'photo2gps.txt', 'r'):
    img_name = line.split(' ')[0]
    img_names.append(img_name)

print("Resizing")
Parallel(n_jobs=64)(delayed(resize_dataset)(id) for id in img_names)
print("DONE")