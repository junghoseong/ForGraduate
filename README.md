# ForGraduate

SWIG_ARMHF := {destdir}/{install_dir}/edgetpu/swig/_edgetpu_cpp_wrapper.cpython-*-arm-linux-gnueabihf.so
SWIG_ARM64 := {destdir}/{install_dir}/edgetpu/swig/_edgetpu_cpp_wrapper.cpython-*-aarch64-linux-gnu.so
SWIG_AMD64 := {destdir}/{install_dir}/edgetpu/swig/_edgetpu_cpp_wrapper.cpython-*-x86_64-linux-gnu.so

AppendFullyConnectedAndSoftmaxLayerToModel

PyObject* AppendFullyConnectedAndSoftmaxLayerToModel(
    const std::string& in_model_path, const std::string& out_model_path,
    const float* weights, size_t weights_size, const float* biases,
    size_t biases_size, float out_tensor_min, float out_tensor_max) {
  coral::EdgeTpuErrorReporter reporter;
  const auto& status = coral::learn::AppendFullyConnectedAndSoftmaxLayerToModel(
               in_model_path, out_model_path, weights, weights_size, biases,
               biases_size, out_tensor_min, out_tensor_max, &reporter);
  if(status == coral::kEdgeTpuApiError) {
    PyErr_SetString(PyExc_RuntimeError, reporter.message().c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

This function assumes the input tflite model is an embedding extractor, e.g., a
classification model without the last FC+Softmax layer. It does the following:
  *) Quantizes learned weights and biases from float32 to uint8;
  *) Appends quantized weights and biases as FC layer;
  *) Adds a Softmax layer;
  *) Stores the result in tflite file format specified by `out_model_path`;
Args:
  in_model_path: string, path to input tflite model;
  out_model_path: string, path to output tflite model;
  weights: 1 dimensional float32 np.ndarray, flattened learned weights. Learned
    weights is a num_classes x embedding_vector_dim matrix;
  biases: 1 dimensional float32 np.ndarray of length num_classes;
  out_tensor_min: float, expected min value of FC layer, for quantization parameter;
  out_tensor_max: float, expected max value of FC layer, for quantization parameter;
Raises:
  RuntimeError: with corresponding reason for failure.
"""
