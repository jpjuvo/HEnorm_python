import normalizeStaining
from multiprocessing import Pool
import os

threads = 8

imagesDir = 'imgs/'
imageFiles = []
# r=root, d=directories, f = files
for r, d, f in os.walk(imagesDir):
    for file in f:
        if '.png' in file:
            imageFiles.append(os.path.join(r, file))

with Pool(threads) as p:
    p.map(normalizeStaining.normalizeStaining, imageFiles)