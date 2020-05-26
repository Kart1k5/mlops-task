#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Convolution2D , Dense , MaxPooling2D ,Flatten
import numpy as np


# In[2]:


df = mnist.load_data('mydata.db')


# In[3]:


(x_train , y_train) , (x_test , y_test)  = df


# In[4]:


image1 = x_train[45400]


# In[5]:


image1_label = y_train[45400]


# In[6]:


image1_1d = image1.reshape(28*28)


# In[7]:


x_train = x_train.astype('float32')
x_train = np.expand_dims(x_train,axis=0)
x_train = x_train.reshape(-1,28,28,1)
x_test = np.expand_dims(x_test,axis=0)
x_test = x_test.reshape(-1,28,28,1)


# In[8]:


y_train_cat = to_categorical(y_train)


# In[9]:


model=Sequential()
model.add(Convolution2D(filters = 64 ,kernel_size=(2,2) ,input_shape =(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024 ,activation='relu'))
model.add(Dense(512 ,activation='relu'))
model.add(Dense(10 ,activation='softmax'))


# In[10]:


from keras.optimizers import RMSprop


# In[11]:


model.compile(optimizer=RMSprop() , loss = 'categorical_crossentropy',metrics=['accuracy'])


# In[12]:


h = model.fit(x_train,y_train_cat,epochs=5)


# In[13]:


test_img = x_test[0].reshape(28*28,)


# In[24]:


accuracy = max(h.history['accuracy'])
accuracy = str(accuracy)


# In[25]:


file = open('accuracy.txt' , 'w')
file.write(accuracy)
file.close()


# In[ ]:




