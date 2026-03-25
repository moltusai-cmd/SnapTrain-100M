/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
C:\Users\ncouf\bitnet\qairt_sdk\qairt\2.26.2.240911\bin\arm64x-windows-msvc\qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; batch=None; bias_bitwidth=8; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=-1; define_symbol=None; disable_batchnorm_folding=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=0; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=None; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=None; input_type=[]; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; out_names=['weights_latent_out', 'prediction']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[]; quantization_overrides=; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.26.2.240911233520_20465";
extern "C" {
static ModelError_t addTensor_input_x(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input_x[] = {1, 16};
  VALIDATE(model.addTensor("input_x", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input_x",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_input_x,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_target_y(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_target_y[] = {1, 8};
  VALIDATE(model.addTensor("target_y", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "target_y",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_target_y,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_weights_latent_in(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_weights_latent_in[] = {8, 16};
  VALIDATE(model.addTensor("weights_latent_in", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "weights_latent_in",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_weights_latent_in,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_node_abs_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_abs_1 */
  Qnn_Param_t params_node_abs_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_node_abs_1[] = {
    "weights_latent_in"
  };
  uint32_t dimensions_abs_1[] = {8, 16};
  Qnn_Tensor_t outputs_node_abs_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "abs_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_abs_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_abs_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params_node_abs_1, // Node Params
                         1, // Num Node Params
                         inputs_node_abs_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_abs_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_n0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR n0 */
  uint32_t dimensions_n0_axes[] = {2};
  uint32_t n0_axes[] = {0, 1};
  Qnn_Param_t params_n0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "n0_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_n0_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)n0_axes,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="keep_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}
  };
  const char*  inputs_n0[] = {
    "abs_1"
  };
  uint32_t dimensions_result[] = {1, 1};
  Qnn_Tensor_t outputs_n0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "result",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_result,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "n0", // Node Name
                         "qti.aisw", // Package Name
                         "ReduceMean", // Qnn Node Type
                         params_n0, // Node Params
                         2, // Num Node Params
                         inputs_n0, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_n0, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_n1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR n1 */
  const char*  inputs_n1[] = {
    "result"
  };
  uint32_t dimensions_mean[] = {1};
  Qnn_Tensor_t outputs_n1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "mean",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_mean,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "n1", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_n1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_n1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_clamp(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_clamp */
  Qnn_Param_t params_node_clamp[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 340282346638528859811704183484516925440.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000010000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_node_clamp[] = {
    "mean"
  };
  uint32_t dimensions_clamp[] = {1};
  Qnn_Tensor_t outputs_node_clamp[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "clamp",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_clamp,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_clamp", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_clamp, // Node Params
                         3, // Num Node Params
                         inputs_node_clamp, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_clamp, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_div(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_div */
  Qnn_Param_t params_node_div[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs_node_div[] = {
    "weights_latent_in",
    "clamp"
  };
  uint32_t dimensions_div[] = {8, 16};
  Qnn_Tensor_t outputs_node_div[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "div",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_div,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_div", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_div, // Node Params
                         1, // Num Node Params
                         inputs_node_div, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_div, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_round_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_round_1 */
  Qnn_Param_t params_node_round_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 11}}}}
  };
  const char*  inputs_node_round_1[] = {
    "div"
  };
  uint32_t dimensions_round_1[] = {8, 16};
  Qnn_Tensor_t outputs_node_round_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "round_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_round_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_round_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params_node_round_1, // Node Params
                         1, // Num Node Params
                         inputs_node_round_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_round_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_clamp_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_clamp_1 */
  Qnn_Param_t params_node_clamp_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = -1.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_node_clamp_1[] = {
    "round_1"
  };
  uint32_t dimensions_clamp_1[] = {8, 16};
  Qnn_Tensor_t outputs_node_clamp_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "clamp_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_clamp_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_clamp_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_clamp_1, // Node Params
                         3, // Num Node Params
                         inputs_node_clamp_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_clamp_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_mul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_mul */
  Qnn_Param_t params_node_mul[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs_node_mul[] = {
    "clamp_1",
    "clamp"
  };
  uint32_t dimensions_mul[] = {8, 16};
  Qnn_Tensor_t outputs_node_mul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "mul",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_mul,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_mul", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_mul, // Node Params
                         1, // Num Node Params
                         inputs_node_mul, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_mul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_abs_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_abs_2 */
  Qnn_Param_t params_node_abs_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_node_abs_2[] = {
    "input_x"
  };
  uint32_t dimensions_abs_2[] = {1, 16};
  Qnn_Tensor_t outputs_node_abs_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "abs_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_abs_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_abs_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params_node_abs_2, // Node Params
                         1, // Num Node Params
                         inputs_node_abs_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_abs_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_max_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_max_1 */
  uint32_t dimensions_node_max_1_axes[] = {2};
  uint32_t node_max_1_axes[] = {0, 1};
  Qnn_Param_t params_node_max_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_max_1_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_max_1_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_max_1_axes,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="keep_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_max_1[] = {
    "abs_2"
  };
  uint32_t dimensions_max_1[] = {1};
  Qnn_Tensor_t outputs_node_max_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "max_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_max_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_max_1", // Node Name
                         "qti.aisw", // Package Name
                         "ReduceMax", // Qnn Node Type
                         params_node_max_1, // Node Params
                         2, // Num Node Params
                         inputs_node_max_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_max_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_val_10(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_val_10[] = {1};
  VALIDATE(model.addTensor("val_10", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "val_10",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_val_10,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(val_10),
                                                .dataSize=BINLEN(val_10)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_node_div_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_div_1 */
  Qnn_Param_t params_node_div_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs_node_div_1[] = {
    "max_1",
    "val_10"
  };
  uint32_t dimensions_div_1[] = {1};
  Qnn_Tensor_t outputs_node_div_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "div_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_div_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_div_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_div_1, // Node Params
                         1, // Num Node Params
                         inputs_node_div_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_div_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_clamp_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_clamp_2 */
  Qnn_Param_t params_node_clamp_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 340282346638528859811704183484516925440.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000010000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_node_clamp_2[] = {
    "div_1"
  };
  uint32_t dimensions_clamp_2[] = {1};
  Qnn_Tensor_t outputs_node_clamp_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "clamp_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_clamp_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_clamp_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_clamp_2, // Node Params
                         3, // Num Node Params
                         inputs_node_clamp_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_clamp_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_div_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_div_2 */
  Qnn_Param_t params_node_div_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs_node_div_2[] = {
    "input_x",
    "clamp_2"
  };
  uint32_t dimensions_div_2[] = {1, 16};
  Qnn_Tensor_t outputs_node_div_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "div_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_div_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_div_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_div_2, // Node Params
                         1, // Num Node Params
                         inputs_node_div_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_div_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_round_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_round_2 */
  Qnn_Param_t params_node_round_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 11}}}}
  };
  const char*  inputs_node_round_2[] = {
    "div_2"
  };
  uint32_t dimensions_round_2[] = {1, 16};
  Qnn_Tensor_t outputs_node_round_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "round_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_round_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_round_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params_node_round_2, // Node Params
                         1, // Num Node Params
                         inputs_node_round_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_round_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_clamp_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_clamp_3 */
  Qnn_Param_t params_node_clamp_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 7.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = -7.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_node_clamp_3[] = {
    "round_2"
  };
  uint32_t dimensions_clamp_3[] = {1, 16};
  Qnn_Tensor_t outputs_node_clamp_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "clamp_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_clamp_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_clamp_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_clamp_3, // Node Params
                         3, // Num Node Params
                         inputs_node_clamp_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_clamp_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_mul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_mul_1 */
  Qnn_Param_t params_node_mul_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs_node_mul_1[] = {
    "clamp_3",
    "clamp_2"
  };
  uint32_t dimensions_mul_1[] = {1, 16};
  Qnn_Tensor_t outputs_node_mul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "mul_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_mul_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_mul_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_mul_1, // Node Params
                         1, // Num Node Params
                         inputs_node_mul_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_mul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t */
  uint32_t dimensions_node_t_perm[] = {2};
  uint32_t node_t_perm[] = {1, 0};
  Qnn_Param_t params_node_t[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t[] = {
    "mul"
  };
  uint32_t dimensions_t[] = {16, 8};
  Qnn_Tensor_t outputs_node_t[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t, // Node Params
                         1, // Num Node Params
                         inputs_node_t, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul */
  Qnn_Param_t params_node_matmul[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul[] = {
    "mul_1",
    "t"
  };
  uint32_t dimensions_prediction[] = {1, 8};
  Qnn_Tensor_t outputs_node_matmul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "prediction",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_prediction,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_sub(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_sub */
  Qnn_Param_t params_node_sub[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs_node_sub[] = {
    "prediction",
    "target_y"
  };
  uint32_t dimensions_sub[] = {1, 8};
  Qnn_Tensor_t outputs_node_sub[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sub",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sub,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_sub", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_sub, // Node Params
                         1, // Num Node Params
                         inputs_node_sub, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_sub, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_abs_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_abs_3 */
  Qnn_Param_t params_node_abs_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_node_abs_3[] = {
    "sub"
  };
  uint32_t dimensions_abs_3[] = {1, 8};
  Qnn_Tensor_t outputs_node_abs_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "abs_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_abs_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_abs_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params_node_abs_3, // Node Params
                         1, // Num Node Params
                         inputs_node_abs_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_abs_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_max_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_max_2 */
  uint32_t dimensions_node_max_2_axes[] = {2};
  uint32_t node_max_2_axes[] = {0, 1};
  Qnn_Param_t params_node_max_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_max_2_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_max_2_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_max_2_axes,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="keep_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_max_2[] = {
    "abs_3"
  };
  uint32_t dimensions_max_2[] = {1};
  Qnn_Tensor_t outputs_node_max_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "max_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_max_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_max_2", // Node Name
                         "qti.aisw", // Package Name
                         "ReduceMax", // Qnn Node Type
                         params_node_max_2, // Node Params
                         2, // Num Node Params
                         inputs_node_max_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_max_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_div_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_div_3 */
  Qnn_Param_t params_node_div_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs_node_div_3[] = {
    "max_2",
    "val_10"
  };
  uint32_t dimensions_div_3[] = {1};
  Qnn_Tensor_t outputs_node_div_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "div_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_div_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_div_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_div_3, // Node Params
                         1, // Num Node Params
                         inputs_node_div_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_div_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_clamp_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_clamp_4 */
  Qnn_Param_t params_node_clamp_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 340282346638528859811704183484516925440.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000010000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_node_clamp_4[] = {
    "div_3"
  };
  uint32_t dimensions_clamp_4[] = {1};
  Qnn_Tensor_t outputs_node_clamp_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "clamp_4",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_clamp_4,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_clamp_4", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_clamp_4, // Node Params
                         3, // Num Node Params
                         inputs_node_clamp_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_clamp_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_div_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_div_4 */
  Qnn_Param_t params_node_div_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs_node_div_4[] = {
    "sub",
    "clamp_4"
  };
  uint32_t dimensions_div_4[] = {1, 8};
  Qnn_Tensor_t outputs_node_div_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "div_4",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_div_4,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_div_4", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_div_4, // Node Params
                         1, // Num Node Params
                         inputs_node_div_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_div_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_round_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_round_3 */
  Qnn_Param_t params_node_round_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 11}}}}
  };
  const char*  inputs_node_round_3[] = {
    "div_4"
  };
  uint32_t dimensions_round_3[] = {1, 8};
  Qnn_Tensor_t outputs_node_round_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "round_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_round_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_round_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params_node_round_3, // Node Params
                         1, // Num Node Params
                         inputs_node_round_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_round_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_clamp_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_clamp_5 */
  Qnn_Param_t params_node_clamp_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 7.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = -7.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_node_clamp_5[] = {
    "round_3"
  };
  uint32_t dimensions_clamp_5[] = {1, 8};
  Qnn_Tensor_t outputs_node_clamp_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "clamp_5",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_clamp_5,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_clamp_5", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_clamp_5, // Node Params
                         3, // Num Node Params
                         inputs_node_clamp_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_clamp_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_mul_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_mul_2 */
  Qnn_Param_t params_node_mul_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs_node_mul_2[] = {
    "clamp_5",
    "clamp_4"
  };
  uint32_t dimensions_mul_2[] = {1, 8};
  Qnn_Tensor_t outputs_node_mul_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "mul_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_mul_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_mul_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_mul_2, // Node Params
                         1, // Num Node Params
                         inputs_node_mul_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_mul_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_1 */
  uint32_t dimensions_node_t_1_perm[] = {2};
  uint32_t node_t_1_perm[] = {1, 0};
  Qnn_Param_t params_node_t_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_1_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_1_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_1_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_1[] = {
    "mul_2"
  };
  uint32_t dimensions_t_1[] = {8, 1};
  Qnn_Tensor_t outputs_node_t_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_1", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_1, // Node Params
                         1, // Num Node Params
                         inputs_node_t_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_1 */
  Qnn_Param_t params_node_matmul_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_1[] = {
    "t_1",
    "mul_1"
  };
  uint32_t dimensions_matmul_1[] = {8, 16};
  Qnn_Tensor_t outputs_node_matmul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_1", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_1, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_val_14(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_val_14[] = {1};
  VALIDATE(model.addTensor("val_14", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "val_14",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_val_14,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(val_14),
                                                .dataSize=BINLEN(val_14)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_node_mul_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_mul_3 */
  Qnn_Param_t params_node_mul_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs_node_mul_3[] = {
    "matmul_1",
    "val_14"
  };
  uint32_t dimensions_mul_3[] = {8, 16};
  Qnn_Tensor_t outputs_node_mul_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "mul_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_mul_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_mul_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_mul_3, // Node Params
                         1, // Num Node Params
                         inputs_node_mul_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_mul_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_sub_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_sub_1 */
  Qnn_Param_t params_node_sub_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs_node_sub_1[] = {
    "weights_latent_in",
    "mul_3"
  };
  uint32_t dimensions_weights_latent_out[] = {8, 16};
  Qnn_Tensor_t outputs_node_sub_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "weights_latent_out",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_weights_latent_out,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_sub_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_node_sub_1, // Node Params
                         1, // Num Node Params
                         inputs_node_sub_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_sub_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

extern "C" __declspec(dllexport)
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for qnn_training_model*/
  QnnModel qnn_training_model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("qnn_training_model", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(qnn_training_model.initialize(backendHandle, interface, contextHandle, "qnn_training_model", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_input_x(qnn_training_model), err);
  VALIDATE(addTensor_target_y(qnn_training_model), err);
  VALIDATE(addTensor_weights_latent_in(qnn_training_model), err);
  VALIDATE(addNode_node_abs_1(qnn_training_model), err);
  VALIDATE(addNode_n0(qnn_training_model), err);
  VALIDATE(addNode_n1(qnn_training_model), err);
  VALIDATE(addNode_node_clamp(qnn_training_model), err);
  VALIDATE(addNode_node_div(qnn_training_model), err);
  VALIDATE(addNode_node_round_1(qnn_training_model), err);
  VALIDATE(addNode_node_clamp_1(qnn_training_model), err);
  VALIDATE(addNode_node_mul(qnn_training_model), err);
  VALIDATE(addNode_node_abs_2(qnn_training_model), err);
  VALIDATE(addNode_node_max_1(qnn_training_model), err);
  VALIDATE(addTensor_val_10(qnn_training_model), err);
  VALIDATE(addNode_node_div_1(qnn_training_model), err);
  VALIDATE(addNode_node_clamp_2(qnn_training_model), err);
  VALIDATE(addNode_node_div_2(qnn_training_model), err);
  VALIDATE(addNode_node_round_2(qnn_training_model), err);
  VALIDATE(addNode_node_clamp_3(qnn_training_model), err);
  VALIDATE(addNode_node_mul_1(qnn_training_model), err);
  VALIDATE(addNode_node_t(qnn_training_model), err);
  VALIDATE(addNode_node_matmul(qnn_training_model), err);
  VALIDATE(addNode_node_sub(qnn_training_model), err);
  VALIDATE(addNode_node_abs_3(qnn_training_model), err);
  VALIDATE(addNode_node_max_2(qnn_training_model), err);
  VALIDATE(addNode_node_div_3(qnn_training_model), err);
  VALIDATE(addNode_node_clamp_4(qnn_training_model), err);
  VALIDATE(addNode_node_div_4(qnn_training_model), err);
  VALIDATE(addNode_node_round_3(qnn_training_model), err);
  VALIDATE(addNode_node_clamp_5(qnn_training_model), err);
  VALIDATE(addNode_node_mul_2(qnn_training_model), err);
  VALIDATE(addNode_node_t_1(qnn_training_model), err);
  VALIDATE(addNode_node_matmul_1(qnn_training_model), err);
  VALIDATE(addTensor_val_14(qnn_training_model), err);
  VALIDATE(addNode_node_mul_3(qnn_training_model), err);
  VALIDATE(addNode_node_sub_1(qnn_training_model), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&qnn_training_model};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

extern "C" __declspec(dllexport)
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}
//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "QnnModel.hpp"
#include "QnnModelPal.hpp"
#include "QnnTypeMacros.hpp"

#define FREE_MEMORY(ptr1, ptr2, ptr3) \
  do {                                \
    free(ptr1);                       \
    free(ptr2);                       \
    free(ptr3);                       \
  } while (0)

namespace qnn_wrapper_api {

ModelError_t QnnModel::initialize(const Qnn_BackendHandle_t &backendHandle,
                                  const QNN_INTERFACE_VER_TYPE &qnnInterface,
                                  const Qnn_ContextHandle_t &context,
                                  const char *graphName,
                                  bool debug,
                                  uint8_t doNodeValidations,
                                  const QnnGraph_Config_t **graphConfigs) {
  if (backendHandle == nullptr) {
    PRINT_ERROR("QnnModel::initialize() nullptr passed as backend handle.");
    return MODEL_CONTEXT_ERROR;
  }
  if (context == nullptr) {
    PRINT_ERROR("QnnModel::initialize() nullptr passed as context handle.");
    return MODEL_CONTEXT_ERROR;
  }
  if (graphName == nullptr) {
    PRINT_ERROR("QnnModel::initialize() nullptr passed as graphName.");
    return MODEL_GRAPH_ERROR;
  }

  if (!m_graphName.empty()) {
    // only one graph is allowed per QnnModel
    PRINT_ERROR("QnnModel::initialize() model for graph %s already initialized.", graphName);
    return MODEL_GRAPH_ERROR;
  }

  if (!m_doNodeValidations) {
    PRINT_WARNING(
        "Node validation disabled. Backend will not perform op "
        "validation prior to adding Node. \n");
  }

  m_qnnInterface      = qnnInterface;
  m_backendHandle     = backendHandle;
  m_graphName         = graphName;
  m_debug             = debug;
  m_doNodeValidations = doNodeValidations;

  if (m_qnnInterface.graphCreate(context, graphName, graphConfigs, &m_graph) !=
          QNN_GRAPH_NO_ERROR ||
      m_graph == nullptr) {
    PRINT_ERROR("QnnModel::initialize() not able to create graph in given context.");
    return MODEL_GRAPH_ERROR;
  }

  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::addTensor(const char *nodeName, Qnn_Tensor_t *tensor, bool saveTensor) {
  ModelError_t err;
  if (!tensor) {
    PRINT_ERROR("QnnModel::addTensor() NULL tensor pointer provided.\n");
    return MODEL_TENSOR_ERROR;
  }
  VALIDATE_TENSOR_VERSION((*tensor), err);

  // Verify tensor being added is not a duplicate
  std::string mapEntry = std::string(QNN_TENSOR_GET_NAME(tensor));
  if (m_modelTensorsMap.find(mapEntry) != m_modelTensorsMap.end()) {
    PRINT_ERROR("QnnModel::addTensor() creating tensor %s for node %s. Tensor already exists.\n",
                mapEntry.c_str(),
                nodeName);

    return MODEL_TENSOR_ERROR;
  }

  const std::map<Qnn_DataType_t, float> dataTypeToSize = {
      {QNN_DATATYPE_INT_8, 1},
      {QNN_DATATYPE_INT_16, 2},
      {QNN_DATATYPE_INT_32, 4},
      {QNN_DATATYPE_INT_64, 8},
      {QNN_DATATYPE_UINT_8, 1},
      {QNN_DATATYPE_UINT_16, 2},
      {QNN_DATATYPE_UINT_32, 4},
      {QNN_DATATYPE_UINT_64, 8},
      {QNN_DATATYPE_FLOAT_16, 2},
      {QNN_DATATYPE_FLOAT_32, 4},
      {QNN_DATATYPE_FLOAT_64, 8},
      {QNN_DATATYPE_BOOL_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_4, 0.5},
      {QNN_DATATYPE_SFIXED_POINT_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_16, 2},
      {QNN_DATATYPE_SFIXED_POINT_32, 4},
      {QNN_DATATYPE_UFIXED_POINT_4, 0.5},
      {QNN_DATATYPE_UFIXED_POINT_8, 1},
      {QNN_DATATYPE_UFIXED_POINT_16, 2},
      {QNN_DATATYPE_UFIXED_POINT_32, 4},
  };

  if (dataTypeToSize.find(QNN_TENSOR_GET_DATA_TYPE(tensor)) == dataTypeToSize.end()) {
    PRINT_ERROR(
        "QnnModel::addTensor() invalid QNN data type provided, %u, for tensor %s on node %s\n",
        QNN_TENSOR_GET_DATA_TYPE(tensor),
        QNN_TENSOR_GET_NAME(tensor),
        nodeName);
    return MODEL_TENSOR_ERROR;
  }

  // sanity check tensor data if addTensor used for static tensor
  if ((QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_STATIC) ||
      (QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_UPDATEABLE_STATIC)) {
    if (QNN_TENSOR_GET_MEM_TYPE(tensor) != QNN_TENSORMEMTYPE_RAW) {
      PRINT_ERROR(
          "QnnModel::addTensor(): Expected raw memType in provided static tensor %s for node %s",
          mapEntry.c_str(),
          nodeName);
      return MODEL_TENSOR_ERROR;
    }
    // verify size expressed by the dims matches the raw tensor size
    uint32_t qnnTensorSize =
        std::lround(std::accumulate(QNN_TENSOR_GET_DIMENSIONS(tensor),
                                    QNN_TENSOR_GET_DIMENSIONS(tensor) + QNN_TENSOR_GET_RANK(tensor),
                                    dataTypeToSize.find(QNN_TENSOR_GET_DATA_TYPE(tensor))->second,
                                    std::multiplies<float>()));
    if (qnnTensorSize != QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize) {
      PRINT_ERROR(
          "QnnModel::addTensor(): Adding STATIC tensor, length mismatch between clientBuf"
          "size and tensor Dims(dim * rank * sizeof(datatype) for, nodeName: %s, tensorName: %s."
          "Got tensorSize: %d, tensor.clientBuf.dataSize: %d.\n",
          nodeName,
          QNN_TENSOR_GET_NAME(tensor),
          qnnTensorSize,
          QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize);
      return MODEL_TENSOR_ERROR;
    }
  }

  if (m_debug && QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_NATIVE) {
    // for debug, make all tensors accessible by client
    QNN_TENSOR_SET_TYPE(tensor, QNN_TENSOR_TYPE_APP_READ);
  } else if (m_debug && QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_UPDATEABLE_NATIVE) {
    // for debug, make all tensors accessible by client
    QNN_TENSOR_SET_TYPE(tensor, QNN_TENSOR_TYPE_UPDATEABLE_APP_READ);
  }

  if (m_qnnInterface.tensorCreateGraphTensor(m_graph, tensor) != QNN_TENSOR_NO_ERROR) {
    PRINT_ERROR("QnnModel::addTensor() Creating tensor for node: %s, tensorName: %s.\n",
                nodeName,
                QNN_TENSOR_GET_NAME(tensor));
    return MODEL_TENSOR_ERROR;
  }

  if (saveTensor) {
    Qnn_Tensor_t tensorCopy;
    VALIDATE(deepCopyQnnTensors(*tensor, tensorCopy), err);

    // save network input/outputs tensors to use for setting the Qnn graph's input and output
    // tensors for populating GraphInfo_t for caller
    if ((QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_APP_WRITE) ||
        (QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE)) {
      m_modelInputTensors.push_back(tensorCopy);
    } else if ((QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_APP_READ) ||
               (QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_UPDATEABLE_APP_READ)) {
      m_modelOutputTensors.push_back(tensorCopy);
    }

    // save created tensors for later lookup to populate graph node construction
    m_modelTensorsMap[mapEntry] = tensorCopy;
  }

  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::addTensor(const char *nodeName, Qnn_Tensor_t tensor, bool saveTensor) {
  return addTensor(nodeName, &tensor, saveTensor);
}

ModelError_t QnnModel::getQnnTensor(const char *&nodeName,
                                    const char *&tensorName,
                                    Qnn_Tensor_t &tensor) {
  std::string mapEntry = std::string(tensorName);
  if (m_modelTensorsMap.find(tensorName) == m_modelTensorsMap.end()) {
    PRINT_ERROR(
        "QnnModel::getQnnTensor() tensor %s not found on node %s\n", mapEntry.c_str(), nodeName);
    return MODEL_TENSOR_ERROR;
  }
  tensor = m_modelTensorsMap[mapEntry];

  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::addNode(Qnn_OpConfigVersion_t version,
                               const char *name,
                               const char *packageName,
                               const char *type,
                               Qnn_Param_t *params,
                               uint32_t numOfParams,
                               const char **inputNames,
                               uint32_t numOfInputs,
                               Qnn_Tensor_t *outputTensors,
                               uint32_t numOfOutputs) {
  ModelError_t nodeError;
  Qnn_OpConfig_t opDefinition = QNN_OPCONFIG_INIT;
  opDefinition.version        = version;
  VALIDATE_OP_CONFIG_VERSION((opDefinition), nodeError);

  // populate Qnn param for node
  Qnn_Param_t *nodeParams = (Qnn_Param_t *)malloc(numOfParams * sizeof(Qnn_Param_t));

  // populate input tensors for node
  Qnn_Tensor_t *inputs = (Qnn_Tensor_t *)malloc(numOfInputs * sizeof(Qnn_Tensor_t));

  // populate output tensors of node
  Qnn_Tensor_t *outputs = (Qnn_Tensor_t *)malloc(numOfOutputs * sizeof(Qnn_Tensor_t));

  if (nodeParams == nullptr || inputs == nullptr || outputs == nullptr) {
    PRINT_ERROR(
        "QnnModel::addNode() failed for allocate memory for creating QNN OpConfig for node %s.\n",
        name);
    FREE_MEMORY(nodeParams, inputs, outputs);
    return MODEL_MEMORY_ALLOCATE_ERROR;
  }
  uint32_t nodeParamsCounter = 0;
  for (size_t i = 0; i < numOfParams; i++) {
    switch (params[i].paramType) {
      case QNN_PARAMTYPE_TENSOR: {
        Qnn_Tensor_t &tensor = params[i].tensorParam;
        // Note: set saveTensor to false as no need to save tensor beyond this
        //         function call for params
        nodeError = addTensor(name, &tensor, false);
        if (nodeError != MODEL_NO_ERROR) {
          PRINT_ERROR("QnnModel::addNode() addTensor() failed for tensor param %s on node %s.\n",
                      QNN_TENSOR_GET_NAME(tensor),
                      name);
          FREE_MEMORY(nodeParams, inputs, outputs);
          return nodeError;
        }
        nodeParams[nodeParamsCounter].paramType     = QNN_PARAMTYPE_TENSOR;
        nodeParams[nodeParamsCounter].name          = params[i].name;
        nodeParams[nodeParamsCounter++].tensorParam = tensor;
        break;
      }
      case QNN_PARAMTYPE_SCALAR: {
        nodeParams[nodeParamsCounter].paramType     = QNN_PARAMTYPE_SCALAR;
        nodeParams[nodeParamsCounter].name          = params[i].name;
        nodeParams[nodeParamsCounter++].scalarParam = params[i].scalarParam;
        break;
      }
      default: {
        PRINT_ERROR("QnnModel::addNode() unknown param type passed for param %s on node %s.\n",
                    params[i].name,
                    name);
        FREE_MEMORY(nodeParams, inputs, outputs);
        return MODEL_PARAMS_ERROR;
      }
    }
  }

  size_t inputsCounter = 0;
  for (size_t j = 0; j < numOfInputs; j++) {
    nodeError = getQnnTensor(name, inputNames[j], inputs[inputsCounter++]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() getQnnTensor() failed for tensor %s on node %s.\n",
                  inputNames[j],
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
  }

  size_t outputsCounter        = 0;
  m_modelOutputTensorMap[name] = {};
  for (size_t k = 0; k < numOfOutputs; k++) {
    // create node output tensors first
    nodeError = addTensor(name, outputTensors[k]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() addTensor() failed for tensor %s on node %s\n",
                  QNN_TENSOR_GET_NAME(outputTensors[k]),
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
    const char *outTensorName = QNN_TENSOR_GET_NAME(outputTensors[k]);
    m_modelOutputTensorMap[name].push_back(outTensorName);
    nodeError = getQnnTensor(name, outTensorName, outputs[outputsCounter++]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() getQnnTensor() failed for tensor %s on node %s.\n",
                  outTensorName,
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
  }

  // define and add node to graph
  QNN_OP_CFG_SET_NAME(opDefinition, name);
  QNN_OP_CFG_SET_PACKAGE_NAME(opDefinition, packageName);
  QNN_OP_CFG_SET_TYPE_NAME(opDefinition, type);
  QNN_OP_CFG_SET_PARAMS(opDefinition, numOfParams, nodeParams);
  QNN_OP_CFG_SET_INPUTS(opDefinition, numOfInputs, inputs);
  QNN_OP_CFG_SET_OUTPUTS(opDefinition, numOfOutputs, outputs);

  if (m_doNodeValidations) {
    auto validationStatus = m_qnnInterface.backendValidateOpConfig(m_backendHandle, opDefinition);
    if (validationStatus == QNN_BACKEND_ERROR_NOT_SUPPORTED) {
      PRINT_DEBUG("QnnModel::addNode() validation API not supported.\n");
    } else if (validationStatus != QNN_SUCCESS) {
      PRINT_ERROR("QnnModel::addNode() validating node %s failed.\n", name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return MODEL_GRAPH_OP_VALIDATION_ERROR;
    }
  }

  if (m_qnnInterface.graphAddNode(m_graph, opDefinition) != QNN_GRAPH_NO_ERROR) {
    PRINT_ERROR("QnnModel::addNode() adding node %s failed.\n", name);
    FREE_MEMORY(nodeParams, inputs, outputs);
    return MODEL_GRAPH_ERROR;
  }

  FREE_MEMORY(nodeParams, inputs, outputs);
  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::freeCachedTensors() {
  ModelError_t err = MODEL_NO_ERROR;

  // cleanup cached tensors
  for (std::map<std::string, Qnn_Tensor_t>::iterator tensorIt = m_modelTensorsMap.begin();
       tensorIt != m_modelTensorsMap.end();) {
    Qnn_Tensor_t &tensor = tensorIt->second;
    if (QNN_TENSOR_GET_TYPE(tensor) != QNN_TENSOR_TYPE_APP_WRITE &&
        QNN_TENSOR_GET_TYPE(tensor) != QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE &&
        QNN_TENSOR_GET_TYPE(tensor) != QNN_TENSOR_TYPE_UPDATEABLE_APP_READ &&
        QNN_TENSOR_GET_TYPE(tensor) != QNN_TENSOR_TYPE_APP_READ) {
      VALIDATE(freeQnnTensor(tensor), err);
      tensorIt = m_modelTensorsMap.erase(tensorIt);
    } else {
      tensorIt++;
    }
  }

  return err;
}

ModelError_t QnnModel::finalize(Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal) {
  ModelError_t err;

  // finalize the graph
  if (m_qnnInterface.graphFinalize(m_graph, profile, signal) != QNN_GRAPH_NO_ERROR) {
    PRINT_ERROR("QnnModel::finalize() finalizing graph failed.\n");
    return MODEL_GRAPH_ERROR;
  }

  VALIDATE(freeCachedTensors(), err);

  return err;
}

ModelError_t getGraphInfoFromModels(QnnModel *models,
                                    uint32_t numModels,
                                    GraphInfoPtr_t **graphsInfo) {
  ModelError_t err = MODEL_NO_ERROR;
  if (models == nullptr || graphsInfo == nullptr || numModels <= 0) {
    PRINT_ERROR(
        "getGraphInfoFromModels() models and graphsInfo uninitialized or number of models is "
        "<= 0.\n");
    return MODEL_GRAPH_ERROR;
  }

  *graphsInfo = (GraphInfo_t **)malloc(numModels * sizeof(GraphInfo_t *));
  if (*graphsInfo == nullptr) {
    PRINT_ERROR("getGraphInfoFromModels() graphsInfo malloc returned nullptr.\n");
    return MODEL_GRAPH_ERROR;
  }

  GraphInfo_t *graphArr = (GraphInfo_t *)malloc(numModels * sizeof(GraphInfo_t));
  if (graphArr == nullptr) {
    PRINT_ERROR("getGraphInfoFromModels() graphArr malloc returned nullptr.\n");
    return MODEL_GRAPH_ERROR;
  }

  for (uint32_t i = 0; i < numModels; i++) {
    QnnModel &model   = models[i];
    graphArr[i].graph = model.getQnnGraph();
    graphArr[i].graphName =
        strnDup(model.getQnnGraphName().c_str(), model.getQnnGraphName().size());
    if (graphArr[i].graphName == nullptr) {
      PRINT_ERROR("getGraphInfoFromModels() failed to construct graphName. Received nullptr.\n");
      return MODEL_GRAPH_ERROR;
    }

    // allocate and add graph input/output TensorsWrapper. Note: no need to make deep copies of
    // the tensor's pointer members as they are already allocated on heap in the addTensor
    // function call.
    std::vector<Qnn_Tensor_t> graphInputTensors = model.getGraphInputTensors();
    size_t numInputTensors                      = graphInputTensors.size();
    size_t inputTensorsSize                     = numInputTensors * sizeof(Qnn_Tensor_t);
    graphArr[i].inputTensors                    = (Qnn_Tensor_t *)malloc(inputTensorsSize);
    memscpy(graphArr[i].inputTensors, inputTensorsSize, graphInputTensors.data(), inputTensorsSize);
    graphArr[i].numInputTensors = (uint32_t)numInputTensors;
    // allocate and add graph outputTensors
    std::vector<Qnn_Tensor_t> graphOutputTensors = model.getGraphOutputTensors();
    size_t numOutputTensors                      = graphOutputTensors.size();
    size_t outputTensorsSize                     = numOutputTensors * sizeof(Qnn_Tensor_t);
    graphArr[i].outputTensors                    = (Qnn_Tensor_t *)malloc(outputTensorsSize);
    memscpy(
        graphArr[i].outputTensors, outputTensorsSize, graphOutputTensors.data(), outputTensorsSize);
    graphArr[i].numOutputTensors = (uint32_t)numOutputTensors;

    // have return object point to the populated graph struct
    (*graphsInfo)[i] = graphArr + i;

    // graph composition is complete by this stage, free if any cached tensors remaining
    VALIDATE(model.freeCachedTensors(), err);
  }

  return err;
}

ModelError_t freeGraphsInfo(GraphInfoPtr_t **graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    PRINT_ERROR("freeGraphsInfo() invalid graphsInfo.");
    return MODEL_TENSOR_ERROR;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    PRINT_INFO("Freeing graph in freeGraphInfo");
    free((*graphsInfo)[i]->graphName);
    freeQnnTensors((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
    freeQnnTensors((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
  }

  free(**graphsInfo);
  free(*graphsInfo);
  *graphsInfo = nullptr;

  return MODEL_NO_ERROR;
}
}  // namespace qnn_wrapper_api
//==============================================================================
//
// Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// clang-format off
#include <windows.h>
#include <psapi.h>
#include <stdlib.h>
#include <winevt.h>
// clang-format on

#include <set>
#include <string>

#include "QnnModelPal.hpp"

#define STRINGIFY(x) #x
#define TOSTRING(x)  STRINGIFY(x)

static std::set<HMODULE> mod_handles;
static thread_local char *sg_lastErrMsg = const_cast<char *>("");

void *qnn_wrapper_api::dlSym(void *handle, const char *symbol) {
  FARPROC sym_addr = NULL;
  HANDLE cur_proc;
  DWORD size, size_needed;
  HMODULE *mod_list;
  HMODULE mod = 0;

  if ((!handle) || (!symbol)) {
    return NULL;
  }

  cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, NULL, 0, &size) == 0) {
    sg_lastErrMsg = const_cast<char *>("enumerate modules failed before memory allocation");
    return NULL;
  }

  mod_list = static_cast<HMODULE *>(malloc(size));
  if (!mod_list) {
    sg_lastErrMsg = const_cast<char *>("malloc failed");
    return NULL;
  }

  if (EnumProcessModules(cur_proc, mod_list, size, &size_needed) == 0) {
    sg_lastErrMsg = const_cast<char *>("enumerate modules failed after memory allocation");
    free(mod_list);
    return NULL;
  }

  // DL_DEFAULT needs to bypass those modules with DL_LOCAL flag
  if (handle == DL_DEFAULT) {
    for (size_t i = 0; i < (size / sizeof(HMODULE)); i++) {
      auto iter = mod_handles.find(mod_list[i]);
      if (iter != mod_handles.end()) {
        continue;
      }
      // once find the first non-local module with symbol
      // return its address here to avoid unnecessary looping
      sym_addr = GetProcAddress(mod_list[i], symbol);
      if (sym_addr) {
        free(mod_list);
        return *(void **)(&sym_addr);
      }
    }
  } else {
    mod = static_cast<HMODULE>(handle);
  }

  free(mod_list);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    sg_lastErrMsg = const_cast<char *>("can't resolve symbol");
    return NULL;
  }

  return *(void **)(&sym_addr);
}

char *qnn_wrapper_api::dlError(void) {
  char *retStr = sg_lastErrMsg;

  sg_lastErrMsg = const_cast<char *>("");

  return retStr;
}

char *qnn_wrapper_api::strnDup(const char *source, size_t maxlen) {
  size_t length = strnlen(source, maxlen);

  char *destination = (char *)malloc((length + 1) * sizeof(char));
  if (destination == nullptr) return nullptr;

  // copy length bytes to destination and leave destination[length] to be
  // null terminator
  strncpy_s(destination, length + 1, source, length);

  return destination;
}
//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstdlib>
#include <cstring>
#include <string>

#include "QnnModelPal.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnWrapperUtils.hpp"

namespace qnn_wrapper_api {
size_t memscpy(void *dst, size_t dstSize, const void *src, size_t copySize) {
  if (!dst || !src || !dstSize || !copySize) return 0;

  size_t minSize = dstSize < copySize ? dstSize : copySize;

  memcpy(dst, src, minSize);

  return minSize;
}

ModelError_t getQnnGraphConfigFromInfo(const char *graphName,
                                       const GraphConfigInfo_t **graphsConfigInfo,
                                       const uint32_t numGraphsConfigInfo,
                                       const QnnGraph_Config_t **&graphConfigs) {
  if (!graphsConfigInfo || numGraphsConfigInfo == 0) {
    PRINT_DEBUG("getQnnGraphConfigFromInfo() no custom configs passed for graph:%s.\n", graphName);
    return MODEL_NO_ERROR;
  }

  size_t found = 0;

  for (uint32_t i = 0; i < numGraphsConfigInfo; i++) {
    if (!graphsConfigInfo[i]) {
      PRINT_ERROR(
          "getQnnGraphConfigFromInfo() lookup error while trying to query graphName:%s. "
          "numGraphsConfigInfo > num of element in graphsConfigInfo\n",
          graphName);
      return MODEL_INVALID_ARGUMENT_ERROR;
    }
    if (strcmp(graphsConfigInfo[i]->graphName, graphName) == 0) {
      graphConfigs = graphsConfigInfo[i]->graphConfigs;
      found++;
    }
  }

  if (!found) {
    PRINT_ERROR(
        "getQnnGraphConfigFromInfo() unable to find graphName:%s in provided "
        "graphsConfigInfo object.\n",
        graphName);
    return MODEL_INVALID_ARGUMENT_ERROR;
  } else if (found > 1) {
    PRINT_ERROR(
        "getQnnGraphConfigFromInfo() duplicate GraphConfigInfo entries found with "
        "graphName:%s.\n",
        graphName);
    return MODEL_INVALID_ARGUMENT_ERROR;
  } else {
    return MODEL_NO_ERROR;
  }
}

ModelError_t deepCopyQnnTensors(Qnn_Tensor_t &src, Qnn_Tensor_t &dst) {
  ModelError_t err;
  VALIDATE_TENSOR_VERSION(src, err);

  dst.version = src.version;
  QNN_TENSOR_SET_NAME(
      dst, strnDup(QNN_TENSOR_GET_NAME(src), std::string(QNN_TENSOR_GET_NAME(src)).size()));
  if (QNN_TENSOR_GET_NAME(dst) == nullptr) {
    return MODEL_TENSOR_ERROR;
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

  // Only metadata (i.e. non-static data) is copied from source to destination. The union still
  // must be initialized so that the clientBuf/memHandle do not contain garbage data
  if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
    Qnn_ClientBuffer_t clientBuf = {nullptr, 0};
    QNN_TENSOR_SET_CLIENT_BUF(dst, clientBuf);
  } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
    QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
  } else {
    return MODEL_TENSOR_ERROR;
  }

  Qnn_QuantizeParams_t srcQParam      = QNN_TENSOR_GET_QUANT_PARAMS(src);
  Qnn_QuantizationEncoding_t encoding = srcQParam.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    // need to allocate and copy memory for scaleOffset as it is a pointer array
    Qnn_QuantizeParams_t srcQParamCpy      = srcQParam;
    Qnn_AxisScaleOffset_t &axisScaleOffset = srcQParamCpy.axisScaleOffsetEncoding;
    Qnn_ScaleOffset_t **scaleOffset        = &axisScaleOffset.scaleOffset;
    size_t scaleOffsetSize = axisScaleOffset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
    *scaleOffset           = (Qnn_ScaleOffset_t *)malloc(scaleOffsetSize);
    memscpy(*scaleOffset,
            scaleOffsetSize,
            srcQParam.axisScaleOffsetEncoding.scaleOffset,
            scaleOffsetSize);
    QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParamCpy);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    // need to allocate and copy memory for scaleOffset as it is a pointer array
    Qnn_QuantizeParams_t srcQParamCpy          = srcQParam;
    Qnn_BwAxisScaleOffset_t &bwAxisScaleOffset = srcQParamCpy.bwAxisScaleOffsetEncoding;
    size_t scaleSize                           = bwAxisScaleOffset.numElements * sizeof(float);
    float **scales                             = &bwAxisScaleOffset.scales;
    int32_t **offsets                          = &bwAxisScaleOffset.offsets;
    *scales                                    = (float *)malloc(scaleSize);
    memscpy(*scales, scaleSize, srcQParam.bwAxisScaleOffsetEncoding.scales, scaleSize);

    // Only copy offsets if present, nullptr implies all offsets are 0
    if (bwAxisScaleOffset.offsets != nullptr) {
      size_t offsetSize = bwAxisScaleOffset.numElements * sizeof(int32_t);
      *offsets          = (int32_t *)malloc(offsetSize);
      memscpy(*offsets, offsetSize, srcQParam.bwAxisScaleOffsetEncoding.offsets, offsetSize);
    }
    QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParamCpy);
  } else {
    QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParam);
  }

  // need to allocate and copy memory for all the pointer members
  uint32_t rank = QNN_TENSOR_GET_RANK(src);
  QNN_TENSOR_SET_RANK(dst, rank);
  uint32_t *dimensions = nullptr;
  // If tensor is 0D (rank == 0), do not malloc!
  if (rank != 0) {
    size_t dimSize = rank * sizeof(uint32_t);
    dimensions     = (uint32_t *)malloc(dimSize);
    if (dimensions == nullptr) {
      PRINT_ERROR("deepCopyQnnTensors() Allocation error while copying tensor %s",
                  QNN_TENSOR_GET_NAME(src));
      return MODEL_TENSOR_ERROR;
    }
    memscpy(dimensions, dimSize, QNN_TENSOR_GET_DIMENSIONS(src), dimSize);
  }
  QNN_TENSOR_SET_DIMENSIONS(dst, dimensions);

  QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(dst, QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src));
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return err;
}

ModelError_t freeQnnTensor(Qnn_Tensor_t &tensor) {
  ModelError_t err;
  VALIDATE_TENSOR_VERSION(tensor, err);

  // free all pointer allocations in struct
  free((void *)QNN_TENSOR_GET_NAME(tensor));
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
  auto quant = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  auto encoding = quant.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      free(quant.axisScaleOffsetEncoding.scaleOffset);
  }
  else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
     free(quant.bwAxisScaleOffsetEncoding.scales);
     if (quant.bwAxisScaleOffsetEncoding.offsets != nullptr) {
         free(quant.bwAxisScaleOffsetEncoding.offsets);
     }
  }
  return MODEL_NO_ERROR;
}

ModelError_t freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensor(tensors[i]);
  }
  free(tensors);

  return MODEL_NO_ERROR;
}

std::string getModelErrorName(ModelError_t modelError) {
  switch (modelError) {
    case MODEL_NO_ERROR:
      return "MODEL_NO_ERROR";
    case MODEL_TENSOR_ERROR:
      return "MODEL_TENSOR_ERROR";
    case MODEL_PARAMS_ERROR:
      return "MODEL_PARAMS_ERROR";
    case MODEL_NODES_ERROR:
      return "MODEL_NODES_ERROR";
    case MODEL_GRAPH_ERROR:
      return "MODEL_GRAPH_ERROR";
    case MODEL_CONTEXT_ERROR:
      return "MODEL_CONTEXT_ERROR";
    case MODEL_GENERATION_ERROR:
      return "MODEL_GENERATION_ERROR";
    case MODEL_SETUP_ERROR:
      return "MODEL_SETUP_ERROR";
    case MODEL_UNKNOWN_ERROR:
      return "MODEL_UNKNOWN_ERROR";
    case MODEL_INVALID_ARGUMENT_ERROR:
      return "MODEL_INVALID_ARGUMENT_ERROR";
    case MODEL_FILE_ERROR:
      return "MODEL_FILE_ERROR";
    case MODEL_GRAPH_OP_VALIDATION_ERROR:
      return "MODEL_GRAPH_OP_VALIDATION_ERROR";
    default:
      return "INVALID_ERROR_CODE";
  }
}

}  // namespace qnn_wrapper_api
