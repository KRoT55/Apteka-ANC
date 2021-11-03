#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Convolution2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense


from tensorflow.python.keras import optimizers
from tensorflow.python.keras  import models
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping , TensorBoard


# In[2]:


# Каталог с данными для обучения
train_dir = 'D:/100_test_full/test_aug_iaa'
# Каталог с данными для проверки
val_dir = 'D:/100_test_full/val_aug_iaa'
# Каталог с данными для тестирования
test_dir = 'D:/100_test_full/train'
# Размеры изображения
img_width, img_height = 224, 224
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 75
# Размер мини-выборки
batch_size = 64
# Количество изображений для обучения
nb_train_samples = 205632    
# Количество изображений для проверки
nb_validation_samples = 50616    
# Количество изображений для тестирования
nb_test_samples = 1290 


# In[ ]:





# In[ ]:





# In[3]:


datagen1 = ImageDataGenerator(rescale=1. / 255, rotation_range=180,
                             width_shift_range=0.8,height_shift_range=0.8,
                             shear_range=0.8,zoom_range=0.8,
                             horizontal_flip=True)
datagen = ImageDataGenerator(rescale=1. / 255)


# In[4]:


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[5]:


val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[6]:


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[7]:


#old_own 
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128, (3, 3)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(256, (3, 3)))
model.add(Conv2D(256, (3, 3)))
model.add(Conv2D(256, (3, 3)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))

model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))

model.add(Dense(4096))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(215))
model.add(Activation('softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary()) 

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto',restore_best_weights=True)])

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)


# In[8]:


# #new_vgg
# model = Sequential()
# model.add(ZeroPadding2D((1,1),input_shape=input_shape))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2),padding='same'))

# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(215, activation='softmax'))


# model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])



# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=val_generator,
#     validation_steps=nb_validation_samples // batch_size)

# scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)


# In[9]:


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.layers import Flatten, Dense, Dropout

# from tensorflow.keras import backend as K
# K.image_data_format()

# import numpy as np
# import pandas as pd
# import h5py

# import matplotlib.pyplot as plt

# inc_model=InceptionV3(include_top=False, 
#                       weights='imagenet', 
#                       input_shape=((150, 150, 3)))

# bottleneck_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = bottleneck_datagen.flow_from_directory('./data1/',
#                                         target_size=(150, 150),
#                                         batch_size=100,
#                                         class_mode=None,
#                                         shuffle=False)

# validation_generator = bottleneck_datagen.flow_from_directory('./val_data/',
#                                                                target_size=(150, 150),
#                                                                batch_size=100,
#                                                                class_mode=None,
#                                                                shuffle=False)

# bottleneck_features_train = inc_model.predict_generator(train_generator, len(train_generator))
# np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

# bottleneck_features_validation = inc_model.predict_generator(validation_generator, len(validation_generator))
# np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)
# train_data = np.load(open('./bottleneck_features/bn_features_train.npy', 'rb'))
# train_labels = np.array(getTrainLabels(108, 95))

# validation_data = np.load(open('bottleneck_features/bn_features_validation.npy', 'rb'))
# validation_labels = np.array(getTrainLabels(27, 95))


# fc_model = Sequential()
# fc_model.add(Flatten(input_shape=train_data.shape[1:]))
# fc_model.add(Dense(64, activation='relu', name='dense_one'))
# fc_model.add(Dropout(0.2, name='dropout_one'))
# fc_model.add(Dense(64, activation='relu', name='dense_two'))
# fc_model.add(Dropout(0.2, name='dropout_two'))
# fc_model.add(Dense(95, activation='softmax', name='output'))

# fc_model.compile(optimizer='rmsprop', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])


# fc_model.fit(train_data,
#              train_labels,
#              #epochs=5,
#              nb_epoch=5000,
#              batch_size=32,
#              #shuffle=True,
#              #steps_per_epoch = 1000, 
#             validation_data=(validation_data, validation_labels))

# fc_model.save_weights('bottleneck_features/fc_inception_cats_dogs_250.hdf5')

# fc_model.evaluate(validation_data, validation_labels)
# weights_filename='bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5'

# x = Flatten()(inc_model.output)
# x = Dense(64, activation='relu', name='dense_one')(x)
# x = Dropout(0.5, name='dropout_one')(x)
# x = Dense(64, activation='relu', name='dense_two')(x)
# x = Dropout(0.5, name='dropout_two')(x)
# top_model=Dense(95, activation='softmax', name='output')(x)
# model = Model(
#     inputs=
#     inc_model.input, 
#     outputs=
#     top_model
#     )


# for layer in inc_model.layers[:205]:
#     layer.trainable = False
    
# model.compile(loss='binary_crossentropy',
#               optimizer=SGD(lr=1e-4, momentum=0.9),
#                 #optimizer='rmsprop',
#               metrics=['accuracy'])
# filepath="./weights/iception-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]


# In[10]:


print(scores)


# In[11]:


test_work_dir = 'C:/nn/test_work_dir'
#test_work_dir = test_work_dir(rescale=1. / 255)

work_generator = datagen.flow_from_directory(
    test_work_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[ ]:





# In[12]:


import numpy as np
a=(np.argmax(model.predict(work_generator),axis=-1))

print(a)


# In[13]:


#model.save('testsave12'+str(epochs)+'.h5')
model.save('save_own_dataaug6'+str(epochs)+'-vgg_like.h5')


# In[14]:


label_map=(train_generator.class_indices)
print (label_map)


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


# i=0 
# j=0
# a=0
# b=45778000
# k=0.014
# y=2011

# while (i!=34):
#     l=b* k
#     l=int(l)
#     a=int(a) #   print("\n"+str(a)+"\n")

#     a= b -l +502595
#     j=j+1
#     print ("year :" +str(y+j)+' people:'+str(a)+'\n')
#     print (str(a)+'='+str(b)+'-'+str(l)+'+'+'502595'+'\n'+'\n')
#     b=int(a)
#     i=i+1
    


# In[16]:


# import numpy as np
# import imgaug as ia
# import imgaug.augmenters as iaa


# ia.seed(1)

# # Example batch of images.
# # The array has shape (32, 64, 64, 3) and dtype uint8.
# # images = np.array(
# #     [ia.quokka(size=(64, 64)) for _ in range(32)],
# #     dtype=np.uint8)
# images = 'C:/nn/test_work_dir/1/cam1.png'


# seq = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontal flips
#     iaa.Crop(percent=(0, 0.1)), # random crops
#     # Small gaussian blur with random sigma between 0 and 0.5.
#     # But we only blur about 50% of all images.
#     iaa.Sometimes(
#         0.5,
#         iaa.GaussianBlur(sigma=(0, 0.5))
#     ),
#     # Strengthen or weaken the contrast in each image.
#     iaa.LinearContrast((0.75, 1.5)),
#     # Add gaussian noise.
#     # For 50% of all images, we sample the noise once per pixel.
#     # For the other 50% of all images, we sample the noise per pixel AND
#     # channel. This can change the color (not only brightness) of the
#     # pixels.
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#     # Make some images brighter and some darker.
#     # In 20% of all cases, we sample the multiplier once per channel,
#     # which can end up changing the color of the images.
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     # Apply affine transformations to each image.
#     # Scale/zoom them, translate/move them, rotate them and shear them.
#     iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-25, 25),
#         shear=(-8, 8)
#     )
# ], random_order=True) # apply augmenters in random order

# images_aug = seq(images=images)


# In[17]:


test_work_dir = 'C:/nn/test_work_dir'
#test_work_dir = test_work_dir(rescale=1. / 255)

work_generator = datagen.flow_from_directory(
    test_work_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
model_pred=models.load_model('save_own_dataaug6'+str(epochs)+'-vgg_like.h5')
import numpy as np
a=(np.argmax(model.predict(work_generator),axis=-1))
print (label_map)
print(a)


# In[18]:


import numpy as np

test_work_dir_pred = 'C:/nn/test_work_dir_pred'
#test_work_dir = test_work_dir(rescale=1. / 255)

work_generator_pred = datagen.flow_from_directory(
    test_work_dir_pred,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#model_pred=models.load_model('save_own_dataaug225.h5')

label_map=(train_generator.class_indices)
print (label_map)

a=(np.argmax(model_pred.predict(work_generator_pred),axis=-1))

print(a)


# In[19]:


import numpy as np

test_work_dir_pred = 'C:/nn/test_work_dir_pred'
#test_work_dir = test_work_dir(rescale=1. / 255)

work_generator_pred = datagen.flow_from_directory(
    test_work_dir_pred,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model_pred=models.load_model('save_own_dataaug6'+str(epochs)+'-vgg_like.h5')

label_map=(train_generator.class_indices)
print (label_map)

a=(np.argmax(model_pred.predict(work_generator_pred),axis=-1))
b=model_pred.predict(work_generator_pred)
print(a)
print(b)


# In[ ]:





# In[ ]:




