import os
from datetime import datetime

for image_file_name in os.listdir('A:\INTERNSHIP_PROJECT\face-mask-detector\dataset\with_mask'):
    if image_file_name.endswith(".tif"):
        now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')

        im = Image.open('A:\INTERNSHIP_PROJECT\face-mask-detector\dataset\with_mask'+image_file_name)
        new_width  = 1282
        new_height = 797
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save('A:\INTERNSHIP_PROJECT\face-mask-detector\dataset\with_mask\resized' + now + '.jpg')