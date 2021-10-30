from PIL import Image
import os

current_directory = os.path.dirname(os.path.realpath(__file__))

for name in os.listdir(current_directory):
    if name.endswith(".JPEG"):
        img = Image.open(name)
        rgb_img = img.convert('RGB')
        rgb_img.save(name[:-5] + ".jpg")
        os.remove(name)