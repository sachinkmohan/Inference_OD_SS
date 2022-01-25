#!/usr/bin/env python
# coding: utf-8

# ### Loading the converted tensor RT pb graph

# In[1]:


import tensorflow as tf
from tensorflow.python.platform import gfile

import cv2
import numpy as np

from tensorflow.python.keras.backend import set_session

import Inference_OD

#GRAPH_PB_PATH_TRT = './converted_trt_graph/trt_graph_ss_model.pb'
GRAPH_PB_PATH_FROZEN_SS='./frozen_model_ss/frozen_model_ss_plf.pb'

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
#tf_config.gpu_options.allow_growth = False
tf_sess = tf.Session(config=tf_config)

with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH_FROZEN_SS,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)

   #print(names)


# ### Importing the graph

# In[ ]:

tf.import_graph_def(graph_def, name='')

# ### loading the first and last layers

# In[ ]:

tf_input = tf_sess.graph.get_tensor_by_name('input_1:0')
print(tf_input)

tf_predictions = tf_sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')
print(tf_predictions)

# ### Real time prediction of the mask from the camera

graph = tf.get_default_graph()

#Capture the video from the camera


with graph.as_default():
    set_session(sess)
    inputs, predictions = tf_sess.run([tf_input, tf_predictions], feed_dict={
    tf_input: Inference_OD.image_resized3[None, ...]
})
#cv2.imwrite('file5.jpeg', 255*predictions.squeeze())
pred_image = 255*predictions.squeeze()

##converts pred_image to CV_8UC1 format so that ColorMap can be applied on it
u8 = pred_image.astype(np.uint8)

#Color map autumn is applied to the CV_8UC1 pred_image
im_color = cv2.applyColorMap(u8, cv2.COLORMAP_AUTUMN)
cv2.imshow('input image', Inference_OD.image_resized3)
cv2.imshow('prediction mask',im_color)
cv2.waitKey(0)

cv2.destroyAllWindows()

