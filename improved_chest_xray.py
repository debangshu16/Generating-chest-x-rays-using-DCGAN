import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from glob import glob

def find_last_model_checkpoint():
    last_model_point=0
    for f in (glob('generated_models/Generator_model_*')):
        file = f.split('/')[-1]
        checkpoint_no = int(file.split('_')[-1])
        if checkpoint_no > last_model_point:
            last_model_point = checkpoint_no

    return int(last_model_point)

generator = load_model('generated_models/Generator_model_{}'.format(find_last_model_checkpoint()))
noise = np.random.normal(0,1,(1,100))
generated_img = generator.predict(noise)
generated_img = 0.5* generated_img + 0.5
print (generated_img.shape)

import matplotlib.image as mimg
fig = plt.figure(figsize=(12,12))


fig.add_subplot(1,2,1)
plt.imshow(generated_img[0,:,:,0],cmap='bone')
plt.title('Generated image')

actual_image=mimg.imread('D:/NIH DATASET/images/images_002/images/00001339_000.png')
fig.add_subplot(1,2,2)
plt.imshow(actual_image,cmap='bone')
plt.title('Actual image')

plt.show()
