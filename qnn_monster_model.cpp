/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; batch=None; bias_bitwidth=8; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=-1; define_symbol=None; disable_batchnorm_folding=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=0; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=None; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=None; input_type=[]; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; out_names=['grad_qkv', 'prediction', 'grad_ffn1', 'grad_ffn2']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[]; quantization_overrides=; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.26.2.240911233520_20465";
extern "C" {
static ModelError_t addTensor_src(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_src[] = {64, 512};
  VALIDATE(model.addTensor("src", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "src",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_src,
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

static ModelError_t addTensor_tgt(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_tgt[] = {64, 512};
  VALIDATE(model.addTensor("tgt", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "tgt",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_tgt,
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

static ModelError_t addTensor_w_qkv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_w_qkv[] = {512, 512};
  VALIDATE(model.addTensor("w_qkv", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "w_qkv",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_w_qkv,
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

static ModelError_t addTensor_w_ffn1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_w_ffn1[] = {2048, 512};
  VALIDATE(model.addTensor("w_ffn1", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "w_ffn1",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_w_ffn1,
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

static ModelError_t addTensor_w_ffn2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_w_ffn2[] = {512, 2048};
  VALIDATE(model.addTensor("w_ffn2", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "w_ffn2",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_w_ffn2,
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
    "w_qkv"
  };
  uint32_t dimensions_t[] = {512, 512};
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
    "src",
    "t"
  };
  uint32_t dimensions_matmul[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul,
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
    "matmul"
  };
  uint32_t dimensions_t_1[] = {512, 64};
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
    "matmul",
    "t_1"
  };
  uint32_t dimensions_matmul_1[] = {64, 64};
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

static ModelError_t addTensor_val_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_val_0[] = {1};
  VALIDATE(model.addTensor("val_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "val_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_val_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(val_0),
                                                .dataSize=BINLEN(val_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
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
    "matmul_1",
    "val_0"
  };
  uint32_t dimensions_div[] = {64, 64};
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

static ModelError_t addNode_node_softmax(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_softmax */
  Qnn_Param_t params_node_softmax[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}
  };
  const char*  inputs_node_softmax[] = {
    "div"
  };
  uint32_t dimensions_softmax[] = {64, 64};
  Qnn_Tensor_t outputs_node_softmax[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "softmax",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_softmax,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_softmax", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params_node_softmax, // Node Params
                         2, // Num Node Params
                         inputs_node_softmax, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_softmax, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_2 */
  Qnn_Param_t params_node_matmul_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_2[] = {
    "softmax",
    "matmul"
  };
  uint32_t dimensions_matmul_2[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_2", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_2, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_2 */
  uint32_t dimensions_node_t_2_perm[] = {2};
  uint32_t node_t_2_perm[] = {1, 0};
  Qnn_Param_t params_node_t_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_2_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_2_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_2_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_2[] = {
    "w_ffn1"
  };
  uint32_t dimensions_t_2[] = {512, 2048};
  Qnn_Tensor_t outputs_node_t_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_2", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_2, // Node Params
                         1, // Num Node Params
                         inputs_node_t_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_3 */
  Qnn_Param_t params_node_matmul_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_3[] = {
    "matmul_2",
    "t_2"
  };
  uint32_t dimensions_matmul_3[] = {64, 2048};
  Qnn_Tensor_t outputs_node_matmul_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_3", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_3, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_relu */
  Qnn_Param_t params_node_relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_node_relu[] = {
    "matmul_3"
  };
  uint32_t dimensions_relu[] = {64, 2048};
  Qnn_Tensor_t outputs_node_relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "relu",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_relu,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_relu, // Node Params
                         1, // Num Node Params
                         inputs_node_relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_3 */
  uint32_t dimensions_node_t_3_perm[] = {2};
  uint32_t node_t_3_perm[] = {1, 0};
  Qnn_Param_t params_node_t_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_3_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_3_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_3_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_3[] = {
    "w_ffn2"
  };
  uint32_t dimensions_t_3[] = {2048, 512};
  Qnn_Tensor_t outputs_node_t_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_3", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_3, // Node Params
                         1, // Num Node Params
                         inputs_node_t_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_4 */
  Qnn_Param_t params_node_matmul_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_4[] = {
    "relu",
    "t_3"
  };
  uint32_t dimensions_matmul_4[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_4",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_4,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_4", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_4, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_5 */
  Qnn_Param_t params_node_matmul_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_5[] = {
    "matmul_4",
    "t"
  };
  uint32_t dimensions_matmul_5[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_5",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_5,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_5", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_5, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_5 */
  uint32_t dimensions_node_t_5_perm[] = {2};
  uint32_t node_t_5_perm[] = {1, 0};
  Qnn_Param_t params_node_t_5[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_5_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_5_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_5_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_5[] = {
    "matmul_5"
  };
  uint32_t dimensions_t_5[] = {512, 64};
  Qnn_Tensor_t outputs_node_t_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_5",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_5,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_5", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_5, // Node Params
                         1, // Num Node Params
                         inputs_node_t_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_6 */
  Qnn_Param_t params_node_matmul_6[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_6[] = {
    "matmul_5",
    "t_5"
  };
  uint32_t dimensions_matmul_6[] = {64, 64};
  Qnn_Tensor_t outputs_node_matmul_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_6",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_6,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_6", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_6, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_6, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_6, // Output Tensors 
                         1// Num Output Tensors 
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
    "matmul_6",
    "val_0"
  };
  uint32_t dimensions_div_1[] = {64, 64};
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
            .rank= 2,
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

static ModelError_t addNode_node_softmax_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_softmax_1 */
  Qnn_Param_t params_node_softmax_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}
  };
  const char*  inputs_node_softmax_1[] = {
    "div_1"
  };
  uint32_t dimensions_softmax_1[] = {64, 64};
  Qnn_Tensor_t outputs_node_softmax_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "softmax_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_softmax_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_softmax_1", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params_node_softmax_1, // Node Params
                         2, // Num Node Params
                         inputs_node_softmax_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_softmax_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_7 */
  Qnn_Param_t params_node_matmul_7[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_7[] = {
    "softmax_1",
    "matmul_5"
  };
  uint32_t dimensions_matmul_7[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_7",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_7,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_7", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_7, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_7, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_8 */
  Qnn_Param_t params_node_matmul_8[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_8[] = {
    "matmul_7",
    "t_2"
  };
  uint32_t dimensions_matmul_8[] = {64, 2048};
  Qnn_Tensor_t outputs_node_matmul_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_8",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_8,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_8", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_8, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_8, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_relu_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_relu_1 */
  Qnn_Param_t params_node_relu_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_node_relu_1[] = {
    "matmul_8"
  };
  uint32_t dimensions_relu_1[] = {64, 2048};
  Qnn_Tensor_t outputs_node_relu_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "relu_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_relu_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_relu_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_relu_1, // Node Params
                         1, // Num Node Params
                         inputs_node_relu_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_relu_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_9 */
  Qnn_Param_t params_node_matmul_9[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_9[] = {
    "relu_1",
    "t_3"
  };
  uint32_t dimensions_matmul_9[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_9",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_9,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_9", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_9, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_9, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_10(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_10 */
  Qnn_Param_t params_node_matmul_10[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_10[] = {
    "matmul_9",
    "t"
  };
  uint32_t dimensions_matmul_10[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_10[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_10",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_10,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_10", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_10, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_10, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_10, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_9 */
  uint32_t dimensions_node_t_9_perm[] = {2};
  uint32_t node_t_9_perm[] = {1, 0};
  Qnn_Param_t params_node_t_9[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_9_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_9_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_9_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_9[] = {
    "matmul_10"
  };
  uint32_t dimensions_t_9[] = {512, 64};
  Qnn_Tensor_t outputs_node_t_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_9",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_9,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_9", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_9, // Node Params
                         1, // Num Node Params
                         inputs_node_t_9, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_11(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_11 */
  Qnn_Param_t params_node_matmul_11[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_11[] = {
    "matmul_10",
    "t_9"
  };
  uint32_t dimensions_matmul_11[] = {64, 64};
  Qnn_Tensor_t outputs_node_matmul_11[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_11",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_11,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_11", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_11, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_11, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_11, // Output Tensors 
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
    "matmul_11",
    "val_0"
  };
  uint32_t dimensions_div_2[] = {64, 64};
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

static ModelError_t addNode_node_softmax_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_softmax_2 */
  Qnn_Param_t params_node_softmax_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}
  };
  const char*  inputs_node_softmax_2[] = {
    "div_2"
  };
  uint32_t dimensions_softmax_2[] = {64, 64};
  Qnn_Tensor_t outputs_node_softmax_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "softmax_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_softmax_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_softmax_2", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params_node_softmax_2, // Node Params
                         2, // Num Node Params
                         inputs_node_softmax_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_softmax_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_12(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_12 */
  Qnn_Param_t params_node_matmul_12[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_12[] = {
    "softmax_2",
    "matmul_10"
  };
  uint32_t dimensions_matmul_12[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_12[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_12",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_12,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_12", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_12, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_12, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_12, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_13(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_13 */
  Qnn_Param_t params_node_matmul_13[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_13[] = {
    "matmul_12",
    "t_2"
  };
  uint32_t dimensions_matmul_13[] = {64, 2048};
  Qnn_Tensor_t outputs_node_matmul_13[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_13",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_13,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_13", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_13, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_13, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_13, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_relu_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_relu_2 */
  Qnn_Param_t params_node_relu_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_node_relu_2[] = {
    "matmul_13"
  };
  uint32_t dimensions_relu_2[] = {64, 2048};
  Qnn_Tensor_t outputs_node_relu_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "relu_2",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_relu_2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_relu_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_relu_2, // Node Params
                         1, // Num Node Params
                         inputs_node_relu_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_relu_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_14(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_14 */
  Qnn_Param_t params_node_matmul_14[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_14[] = {
    "relu_2",
    "t_3"
  };
  uint32_t dimensions_matmul_14[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_14[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_14",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_14,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_14", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_14, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_14, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_14, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_15(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_15 */
  Qnn_Param_t params_node_matmul_15[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_15[] = {
    "matmul_14",
    "t"
  };
  uint32_t dimensions_matmul_15[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_15[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_15",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_15,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_15", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_15, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_15, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_15, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_13(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_13 */
  uint32_t dimensions_node_t_13_perm[] = {2};
  uint32_t node_t_13_perm[] = {1, 0};
  Qnn_Param_t params_node_t_13[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_13_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_13_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_13_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_13[] = {
    "matmul_15"
  };
  uint32_t dimensions_t_13[] = {512, 64};
  Qnn_Tensor_t outputs_node_t_13[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_13",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_13,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_13", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_13, // Node Params
                         1, // Num Node Params
                         inputs_node_t_13, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_13, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_16(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_16 */
  Qnn_Param_t params_node_matmul_16[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_16[] = {
    "matmul_15",
    "t_13"
  };
  uint32_t dimensions_matmul_16[] = {64, 64};
  Qnn_Tensor_t outputs_node_matmul_16[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_16,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_16", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_16, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_16, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_16, // Output Tensors 
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
    "matmul_16",
    "val_0"
  };
  uint32_t dimensions_div_3[] = {64, 64};
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
            .rank= 2,
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

static ModelError_t addNode_node_softmax_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_softmax_3 */
  Qnn_Param_t params_node_softmax_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}
  };
  const char*  inputs_node_softmax_3[] = {
    "div_3"
  };
  uint32_t dimensions_softmax_3[] = {64, 64};
  Qnn_Tensor_t outputs_node_softmax_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "softmax_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_softmax_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_softmax_3", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params_node_softmax_3, // Node Params
                         2, // Num Node Params
                         inputs_node_softmax_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_softmax_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_17(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_17 */
  Qnn_Param_t params_node_matmul_17[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_17[] = {
    "softmax_3",
    "matmul_15"
  };
  uint32_t dimensions_matmul_17[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_17[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_17",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_17,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_17", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_17, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_17, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_17, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_18(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_18 */
  Qnn_Param_t params_node_matmul_18[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_18[] = {
    "matmul_17",
    "t_2"
  };
  uint32_t dimensions_matmul_18[] = {64, 2048};
  Qnn_Tensor_t outputs_node_matmul_18[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "matmul_18",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_matmul_18,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_18", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_18, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_18, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_18, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_relu_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_relu_3 */
  Qnn_Param_t params_node_relu_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_node_relu_3[] = {
    "matmul_18"
  };
  uint32_t dimensions_relu_3[] = {64, 2048};
  Qnn_Tensor_t outputs_node_relu_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "relu_3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_relu_3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_relu_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_node_relu_3, // Node Params
                         1, // Num Node Params
                         inputs_node_relu_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_relu_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_19(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_19 */
  Qnn_Param_t params_node_matmul_19[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_19[] = {
    "relu_3",
    "t_3"
  };
  uint32_t dimensions_prediction[] = {64, 512};
  Qnn_Tensor_t outputs_node_matmul_19[] = {
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
                         "node_matmul_19", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_19, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_19, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_19, // Output Tensors 
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
    "tgt"
  };
  uint32_t dimensions_sub[] = {64, 512};
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

static ModelError_t addNode_node_t_16(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_16 */
  uint32_t dimensions_node_t_16_perm[] = {2};
  uint32_t node_t_16_perm[] = {1, 0};
  Qnn_Param_t params_node_t_16[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_16_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_16_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_16_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_16[] = {
    "sub"
  };
  uint32_t dimensions_t_16[] = {512, 64};
  Qnn_Tensor_t outputs_node_t_16[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_16,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_16", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_16, // Node Params
                         1, // Num Node Params
                         inputs_node_t_16, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_16, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_20(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_20 */
  Qnn_Param_t params_node_matmul_20[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_20[] = {
    "t_16",
    "relu_3"
  };
  uint32_t dimensions_grad_ffn2[] = {512, 2048};
  Qnn_Tensor_t outputs_node_matmul_20[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "grad_ffn2",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_grad_ffn2,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_20", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_20, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_20, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_20, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_t_17(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_t_17 */
  uint32_t dimensions_node_t_17_perm[] = {2};
  uint32_t node_t_17_perm[] = {1, 0};
  Qnn_Param_t params_node_t_17[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "node_t_17_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_node_t_17_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)node_t_17_perm,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_node_t_17[] = {
    "relu_3"
  };
  uint32_t dimensions_t_17[] = {2048, 64};
  Qnn_Tensor_t outputs_node_t_17[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "t_17",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_t_17,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_t_17", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_node_t_17, // Node Params
                         1, // Num Node Params
                         inputs_node_t_17, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_node_t_17, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_21(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_21 */
  Qnn_Param_t params_node_matmul_21[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_21[] = {
    "t_17",
    "matmul_17"
  };
  uint32_t dimensions_grad_ffn1[] = {2048, 512};
  Qnn_Tensor_t outputs_node_matmul_21[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "grad_ffn1",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_grad_ffn1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_21", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_21, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_21, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_21, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_node_matmul_22(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR node_matmul_22 */
  Qnn_Param_t params_node_matmul_22[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_node_matmul_22[] = {
    "t_16",
    "src"
  };
  uint32_t dimensions_grad_qkv[] = {512, 512};
  Qnn_Tensor_t outputs_node_matmul_22[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "grad_qkv",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_grad_qkv,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "node_matmul_22", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params_node_matmul_22, // Node Params
                         2, // Num Node Params
                         inputs_node_matmul_22, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_node_matmul_22, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

QNN_API
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

  /* model/graph for qnn_monster_model*/
  QnnModel qnn_monster_model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("qnn_monster_model", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(qnn_monster_model.initialize(backendHandle, interface, contextHandle, "qnn_monster_model", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_src(qnn_monster_model), err);
  VALIDATE(addTensor_tgt(qnn_monster_model), err);
  VALIDATE(addTensor_w_qkv(qnn_monster_model), err);
  VALIDATE(addTensor_w_ffn1(qnn_monster_model), err);
  VALIDATE(addTensor_w_ffn2(qnn_monster_model), err);
  VALIDATE(addNode_node_t(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul(qnn_monster_model), err);
  VALIDATE(addNode_node_t_1(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_1(qnn_monster_model), err);
  VALIDATE(addTensor_val_0(qnn_monster_model), err);
  VALIDATE(addNode_node_div(qnn_monster_model), err);
  VALIDATE(addNode_node_softmax(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_2(qnn_monster_model), err);
  VALIDATE(addNode_node_t_2(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_3(qnn_monster_model), err);
  VALIDATE(addNode_node_relu(qnn_monster_model), err);
  VALIDATE(addNode_node_t_3(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_4(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_5(qnn_monster_model), err);
  VALIDATE(addNode_node_t_5(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_6(qnn_monster_model), err);
  VALIDATE(addNode_node_div_1(qnn_monster_model), err);
  VALIDATE(addNode_node_softmax_1(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_7(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_8(qnn_monster_model), err);
  VALIDATE(addNode_node_relu_1(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_9(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_10(qnn_monster_model), err);
  VALIDATE(addNode_node_t_9(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_11(qnn_monster_model), err);
  VALIDATE(addNode_node_div_2(qnn_monster_model), err);
  VALIDATE(addNode_node_softmax_2(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_12(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_13(qnn_monster_model), err);
  VALIDATE(addNode_node_relu_2(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_14(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_15(qnn_monster_model), err);
  VALIDATE(addNode_node_t_13(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_16(qnn_monster_model), err);
  VALIDATE(addNode_node_div_3(qnn_monster_model), err);
  VALIDATE(addNode_node_softmax_3(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_17(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_18(qnn_monster_model), err);
  VALIDATE(addNode_node_relu_3(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_19(qnn_monster_model), err);
  VALIDATE(addNode_node_sub(qnn_monster_model), err);
  VALIDATE(addNode_node_t_16(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_20(qnn_monster_model), err);
  VALIDATE(addNode_node_t_17(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_21(qnn_monster_model), err);
  VALIDATE(addNode_node_matmul_22(qnn_monster_model), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&qnn_monster_model};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}