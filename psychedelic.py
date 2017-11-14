from keras.preprocessing.image import load_img,img_to_array
import numpy as np 
from keras import backend as K 
from keras.applications.inception_v3 import InceptionV3,preprocess_input

from scipy.ndimage import zoom

from scipy.misc import imsave



K.set_learning_phase(0)

def process_image(image_path):
	img=load_img(image_path)
	img=img_to_array(img)
	img=np.expand_dims(img,axis=0)
	img=preprocess_input(img)

	return img 


model=InceptionV3(include_top=False)
print('Model has been loaded')


settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed8': 1.5,
    },
}

#define loss

layer_dict={layer.name:layer for layer in model.layers}

loss=K.variable(0.)

dream=model.input 

for name in settings['features']:
    coef=settings['features'][name]
    output=layer_dict[name].output
    scaling=K.prod(K.cast(K.shape(output),'float32'))
    loss+=coef*K.sum(K.square(output))/scaling



grads=K.gradients(loss,dream)[0]
grads/=K.maximum(K.mean(K.abs(grads)),1e-7)

outputs=[loss,grads]
fetch_loss_and_gradients=K.function([dream],outputs)



def evaluate_loss_grads(x):
    out=fetch_loss_and_gradients([x])
    return out[0],out[1]




def gradient_ascent(img,iterations,step_scale):
    for i in range(iterations):
        losses,grad_val=evaluate_loss_grads(img)
        img+=step_scale*grad_val
        print(losses)

    return img 

def resize_img(x,size):
    y=np.copy(x)
    factors=(1,float(size[0])/y.shape[1],float(size[1])/y.shape[2],1)
    return zoom(y,factors,order=1)



def deprocess(x):
    image=x.reshape(x.shape[1],x.shape[2],3)
    image/=2
    image+=0.5
    image*=255
    image=np.clip(image,0,255).astype('uint8')
    return image


def save_png(x):
    im=np.copy(x)
    imsave('mydream.png',im)


step=0.01
num_octaves=5
octave_scale=1.4


img=process_image('shanks.jpeg')
original_shape=img.shape[1:3]
successive_shapes=[original_shape]
for i in range(1,num_octaves):
    shape=tuple([int(dim/octave_scale**i) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes=successive_shapes[::-1]
original_img=np.copy(img)
shrunk_original_img=resize_img(original_img,successive_shapes[0])

for shape in successive_shapes:
    print(img.shape)
    img=resize_img(img,shape)
    print(img.shape,shape)
    img=gradient_ascent(img,iterations=25,step_scale=step)
    upscaled_shrunk_original_img=resize_img(shrunk_original_img,shape)
    same_size_original=resize_img(original_img,shape)
    lost_detail=same_size_original- upscaled_shrunk_original_img

    img+=lost_detail
    shrunk_original_img=resize_img(original_img,shape)


img=deprocess(img)
save_png(np.copy(img))












