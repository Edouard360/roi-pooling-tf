import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

wd = os.getcwd() # get the working directory
# wd = wd + '/example/of/new/path/'

roi_pooling_module_2 = tf.load_op_library(wd+"/module_2/roi_pooling.so")
roi_pooling_op_2 = roi_pooling_module_2.roi_pooling2
roi_pooling_module_2_grad = tf.load_op_library(wd+"/module_2/roi_pooling_op_grad.so")
roi_pooling_grad_op_2 = roi_pooling_module_2_grad.roi_pooling2_grad

init = tf.global_variables_initializer()
sess = tf.Session()

w,h = 7,7
test_array = np.arange(1,1+w).reshape(-1,1)*np.arange(1,1+h).reshape(1,-1)
print(test_array)

input_2 = test_array.reshape(1,1,h,w).astype(np.float32)
#print("Input shape: ",input_2.shape)
input_tf_2 = tf.constant(input_2)
rois_tf_2 = tf.constant([[(0, 0, h, w)]])
output_shape_tf_2 = tf.constant((2, 2))
output_tf_2, argmax_tf_2 = roi_pooling_op_2(input_tf_2, rois_tf_2, output_shape_tf_2)
output_2, argmax_2 = sess.run([output_tf_2, argmax_tf_2])
#print("Output shape: ",output_2.shape)
print(output_2.squeeze())

# Here we register our gradient op as the gradient function for our ROI pooling op. 
@ops.RegisterGradient("RoiPooling2")
def _roi_pooling_grad(op, grad0, grad1):
    # The input gradients are the gradients with respect to the outputs of the pooling layer
    input_grad = grad0
    
    # We need the argmax data to compute the gradient connections
    argmax = op.outputs[1]
    
    # Grab the shape of the inputs to the ROI pooling layer
    input_shape = array_ops.shape(op.inputs[0])
    
    # Compute the gradient -> roi_pooling_op_grad is not defined ...
    backprop_grad = roi_pooling_grad_op_2(input_grad, argmax, input_shape)
    
    # Return the gradient for the feature map, but not for the other inputs
    return [backprop_grad, None, None]

variable_2 = tf.get_variable(
    "variable_2",
    shape=[1,1,h,w],
    dtype=tf.float32,
    initializer=tf.constant_initializer(input_2))

init=tf.global_variables_initializer()
sess.run(init)

output_tf_2, argmax_tf_2 = roi_pooling_op_2(variable_2, rois_tf_2, output_shape_tf_2)
output_2, argmax_2 = sess.run([output_tf_2, argmax_tf_2])
loss_tf_2 = tf.reduce_sum(output_tf_2)
grad_tf_2 = tf.gradients(loss_tf_2,variable_2)
grad_2 = sess.run(grad_tf_2[0])

print(grad_2.squeeze())