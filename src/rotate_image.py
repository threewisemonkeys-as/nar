from PIL import Image
import os
# assign directory
directory = '../data/images'
angles=[0,90,180,270]
# iterate over files in 
# that directory
for filename in os.listdir(directory):
    # print(filename)
    img=Image.open('../data/images/%s' %(filename))
    # img.save('../data/images_rotate/%s0' %(filename))
    for angle in angles:
        rotated = img.rotate(angle)
        rotated.save('../data/images_rotate/%d%s' %(angle,filename))
