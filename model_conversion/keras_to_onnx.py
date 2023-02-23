import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tf2onnx import tf_loader
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.optimizer import optimize_graph
from tf2onnx import utils, constants
from tf2onnx.handler import tf_op

model = tf.keras.models.load_model("trained_models/asl_model2.h5")

# Convert the model to concrete function
full_model = tf.function(lambda inputs: model(inputs))    
full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

# Persist the input and output parameters
input_names = [inp.name for inp in full_model.inputs]
output_names = [out.name for out in full_model.outputs]
print("Inputs:", input_names)
print("Outputs:", output_names)

# Freeze the model
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

# Convert the model using extra opset
extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')
with tf_loader.tf_session(graph=tf_graph):
    g = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)
onnx_graph = optimize_graph(g)
model_proto = onnx_graph.make_model("converted")
utils.save_protobuf("trained_models/asl_model.onnx", model_proto)
print("Conversion complete!")
