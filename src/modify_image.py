import PIL.ImageOps 
from PIL import Image
import os
import pickle
# assign directory
def rotate_image(input_dir,output_dir):
    angles=[0,90,180,270]
    # iterate over files in 
    # that directory
    for filename in os.listdir(input_dir):
        # print(filename)
        img=Image.open(input_dir+'/%s' %(filename))
        # img.save('../data/images_rotate/%s0' %(filename))
        for angle in angles:
            rotated = img.rotate(angle)
            rotated.save(output_dir+'%d%s' %(angle,filename))

def affine_image(output_dir,pickle_path):
    trans_images=pickle.load(open(pickle_path,"rb"))
    for key in trans_images:
        inverted_image = PIL.ImageOps.invert(trans_images[key])
        inverted_image.save(output_dir+'/'+key[0]+str(key[1][0])+str(key[1][1])+'.png')

affine_image("../data/images_affine/","../data/trans_images.pickle")
