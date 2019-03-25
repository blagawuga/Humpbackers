
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


DATA_DIR = '/data_file'# location to save the data file
data = input_data.read_data_sets(DATA_DIR, one_hot = True)


# In[5]:


get_ipython().system('cd')


# In[6]:


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))


# In[7]:


y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x,W)


# In[8]:


# Defining the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels = y_true))


# In[9]:


# Learning algorithm
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[10]:


# Evaluation metrics
correct_mask = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))


# In[11]:


NUM_STEPS = 1000
MINIBATCH_SIZE = 100
with tf.Session() as Sess:
    # Train Step
    Sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        Sess.run(gd_step, feed_dict = {x: batch_xs, y_true: batch_ys})
    
    #Test Step
    ans = Sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})


# In[12]:


print( "Accuracy: {}%".format(ans*100))

