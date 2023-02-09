#!/usr/bin/env python
# coding: utf-8

# In[8]:

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from numpy import *
import math
import pickle

# In[9]:

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=1

# In[10]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# In[28]:
'''
print(dataset.class_names)
print(len(dataset))
print(3304/32)

'''
# In[50]:


for i,j in dataset.take(1):
    print(i.shape)
    print(j.numpy())
cls=dataset.class_names
print(cls)


# In[33]:


n=(int(0.8*104))
print(n)
train=dataset.take(n)   #train
temp=dataset.skip(n)
print(len(temp))


# In[34]:


n1=int(0.1*104)
val=temp.take(n1)      #valid
print(len(val))


# In[35]:


test=temp.skip(n1)
print(len(test))       #test


# In[36]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
print(len(train_ds),len(val_ds),len(test_ds))


# In[37]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)



# In[18]:


#model
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# In[19]:


data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])



# In[20]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[21]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 2

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)

# In[22]:

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# In[38]:

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=1,
)


# In[39]:


scores = model.evaluate(test_ds)
print(scores)


# In[48]:


def predict1(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    print(predictions)
    print(predictions[0])
    val=argmax(predictions[0])
    print(val)
    print(cls)
    predicted_class = cls[val]
    confidence = round(100 * (max(predictions[0])), 2)
    return predicted_class, confidence,val

# In[56]:

from matplotlib.image import imread
image=imread('afresh.jpg')
predicted_class, confidence,val= predict1(model,image)
actual_class = cls[val]
plt.imshow(image)
plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
plt.axis("off")


model.save("disease_prediction.h5")

'''pickle.dump(model, open('model4.pkl', 'wb'))
model4 = pickle.load(open('model4.pkl', 'rb'))'''



