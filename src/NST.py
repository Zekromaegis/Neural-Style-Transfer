import tensorflow as tf
from tensorflow.keras import Model
from PIL import Image
# from IPython.core.display import display
import random

def get_white_noise_image(image_shape):
    height,width,channels = image_shape[1:]
    mode = 'RGB' if channels == 3 else 'RGBA'
    pil_map = Image.new(mode, (width, height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * width * height)
    pil_map.putdata(list(random_grid))
    pil_map = tf.keras.preprocessing.image.img_to_array(pil_map)
    pil_map = pil_map.reshape(1,*pil_map.shape)
    return pil_map
    
def mini_model(layer_names, model):
    outputs = [model.get_layer(name).output for name in layer_names]
    model = Model([model.input], outputs)
    return model

class NST:
    def __init__(self, Content_Image_path, Style_Image_path, model, content_layer, style_layers, alpha, beta):
        
        self.model = mini_model(list(style_layers.keys()) + [content_layer],model)
        self.num_style_layers = len(style_layers)
        self.model.trainable = False
        
        self.image_shape = self.model.get_input_shape_at(0)
        self.Content_Image = tf.Variable(self.get_image(Content_Image_path))
        self.Style_Image = tf.Variable(self.get_image(Style_Image_path))
        self.Generated_Image = tf.Variable(self.get_image(Content_Image_path))
        # tf.Variable(get_white_noise_image(self.image_shape))
        
        self.content_layer = content_layer
        self.style_layers = style_layers
        
        self.alpha = alpha
        self.beta = beta

    def as_image(self,image='C'):
        ''' C, S, G '''
        if image == 'C':
            image = self.Content_Image
        elif image == 'S':
            image = self.Style_Image
        elif image == 'G':
            image = self.Generated_Image
        return tf.keras.preprocessing.image.array_to_img(image[0])
    
    def get_image(self,Image_path):
        mode = 'RGB' if self.image_shape[-1] == 3 else 'RGBA'
        image = Image.open(Image_path).resize(self.image_shape[1:3]).convert(mode)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape(1,*image.shape)
        return image
    
    def gram_matrix(self,image):
        im = tf.reshape(tf.transpose(image), [image.shape[-1], -1])
        return tf.matmul(im, tf.transpose(im))
    
    def content_cost(self,a_C,a_G):
        m,height,width,channels = a_G.shape
        a_C_unrolled = tf.reshape(a_C, shape=[-1, channels])
        a_G_unrolled = tf.reshape(a_G, shape=[-1, channels])
        return tf.reduce_sum(tf.math.squared_difference(a_C_unrolled, a_G_unrolled)) / (4 * height * width * channels)
    
    def layer_style_cost(self,a_S,a_G):
        gs = self.gram_matrix(a_S)
        gg = self.gram_matrix(a_G)
        m, height, width, channels = a_G.shape
        return tf.reduce_sum(tf.math.squared_difference(gs, gg)) / (4 * height**2 * width**2 * channels**2)
    
    def style_cost(self,style_output,generated_output):
        cost = 0
        for out_s,out_g,key in zip(style_output, generated_output[:self.num_style_layers], self.style_layers.keys()):
            cost += self.layer_style_cost(out_s,out_g) * self.style_layers[key]
        return cost / self.num_style_layers

    def total_cost(self):
        content_output = self.model(self.Content_Image)[self.num_style_layers:][0]
        generated_output = self.model(self.Generated_Image)
        style_output = self.model(self.Style_Image)[:self.num_style_layers]
        
        return (
            self.alpha * self.content_cost(content_output, generated_output[self.num_style_layers:][0]) +
            self.beta * self.style_cost(style_output,generated_output[:self.num_style_layers])
        )

'''
optimizer = tf.optimizers.Adam(2.0)
STYLE_LAYERS = {
    'block1_conv1': 0.2,
    'block2_conv1': 0.2,
    'block3_conv1': 0.2,
    'block4_conv1': 0.2,
    'block5_conv1': 0.2,
}
CONTENT_LAYERS = 'block4_conv2'
N = NST(content,style,model,CONTENT_LAYERS,STYLE_LAYERS,10,40)
g = N.as_image('G')
for i in range(10):
    optimizer.minimize(N.total_cost,[N.Generated_Image])
    if i%3 == 0:
        print(i,N.total_cost())
        display(N.as_image('G'))
'''