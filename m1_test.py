import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

wd = os.getcwd() # get the working directory
# wd = wd + '/example/of/new/path/'

roi_pooling_module_1 = tf.load_op_library(wd+"/module_1/roi_pooling.so")
roi_pooling_op_1 = roi_pooling_module_1.roi_pooling1
roi_pooling_grad_op_1 = roi_pooling_module_1.roi_pooling1_grad

init = tf.global_variables_initializer()
sess = tf.Session()

w,h = 7,7
test_array = np.arange(1,1+w).reshape(-1,1)*np.arange(1,1+h).reshape(1,-1)
print(test_array)

input_1 = test_array.reshape(1,w,h,1).astype(np.float32)
#print("Input shape: ",input_1.shape)
input_tf_1 = tf.constant(input_1)
rois_tf_1 = tf.constant([(0,0,0, h, w)], dtype=tf.float32)
output_tf_1 = roi_pooling_op_1(input_tf_1, rois_tf_1,2,2,1.0)[0]
output_1 = sess.run(output_tf_1)
#print("Output shape: ",output_1.shape)
print(output_1.squeeze())

@ops.RegisterShape("RoiPooling1")
def _roi_pool_shape(op):
    dims_data = op.inputs[0].get_shape().as_list()
    channels = dims_data[3]
    dims_rois = op.inputs[1].get_shape().as_list()
    num_rois = dims_rois[0]

    pooled_height = op.get_attr('pooled_height')
    pooled_width = op.get_attr('pooled_width')

    output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, channels])
    return [output_shape, output_shape]

@ops.RegisterGradient("RoiPooling1")
def _roi_pool_grad(op, grad, _):
    data = op.inputs[0]
    rois = op.inputs[1]
    argmax = op.outputs[1]
    pooled_height = op.get_attr('pooled_height')
    pooled_width = op.get_attr('pooled_width')
    spatial_scale = op.get_attr('spatial_scale')

    # compute gradient - changed - roi_pooling_op.roi_pool_grad
    data_grad = roi_pooling_grad_op_1(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)

    return [data_grad, None]  # List of one Tensor, since we have one input

variable_1 = tf.get_variable(
    "variable_1",
    shape=[1,w,h,1],
    dtype=tf.float32,
    initializer=tf.constant_initializer(input_1))

init=tf.global_variables_initializer()
sess.run(init)

output_tf_1 = roi_pooling_op_1(variable_1, rois_tf_1,2,2,1.0)[0]
loss_tf_1 = tf.reduce_sum(output_tf_1)
grad_tf_1 = tf.gradients(loss_tf_1,variable_1)
grad_1 = sess.run(grad_tf_1[0])
print(np.squeeze(grad_1))