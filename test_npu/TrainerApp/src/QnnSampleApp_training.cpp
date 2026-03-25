//==============================================================================
//
// Copyright (c) 2022-2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <cstring>

#include "QnnSampleApp.hpp"
#include "PAL/include/PAL/Directory.hpp"
#include "PAL/include/PAL/Path.hpp"
#include "PAL/include/PAL/FileOp.hpp"
#include "QnnWrapperUtils.hpp"

namespace sample_app {

// Constructeur et autres méthodes omises pour la clarté (on garde l'original)
// ... (mais dans le fichier réel on garde tout) ...

sample_app::StatusCode sample_app::QnnSampleApp::executeGraphs() {
  auto returnStatus = StatusCode::SUCCESS;
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    QNN_DEBUG("Starting execution for graphIdx: %d", graphIdx);
    
    Qnn_Tensor_t* inputs  = nullptr;
    Qnn_Tensor_t* outputs = nullptr;
    if (iotensor::StatusCode::SUCCESS != m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx])) {
      return StatusCode::FAILURE;
    }

    auto inputFileList = m_inputFileLists[graphIdx];
    auto graphInfo     = (*m_graphsInfo)[graphIdx];
    
    // On charge les données initiales
    m_ioTensor.populateInputTensors(graphIdx, inputFileList, 0, false, m_inputNameToIndex[graphIdx], inputs, 1);

    // --- HACK : SNAPDRAGON TIGHT TRAINING LOOP ---
    int num_steps = 10000;
    printf("\n>>> STARTING HIGH-SPEED NPU TRAINING LOOP (%d steps) <<<\n", num_steps);
    
    Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
    for (int step = 0; step < num_steps; step++) {
        executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
            graphInfo.graph, inputs, graphInfo.numInputTensors,
            outputs, graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);
        
        if (executeStatus != QNN_GRAPH_NO_ERROR) {
            printf("NPU Fatal Error at step %d\n", step);
            break;
        }

        // FEEDBACK LOOP : On ré-injecte les poids (outputs -> inputs)
        // On suppose : Input 2,3,4 = Weights, Output 1,2,3 = Updated Weights
        if (graphInfo.numInputTensors >= 5 && graphInfo.numOutputTensors >= 4) {
            for (int j = 0; j < 3; j++) {
                memcpy(inputs[j+2].v2.clientBuf.data, outputs[j+1].v2.clientBuf.data, outputs[j+1].v2.clientBuf.dataSize);
            }
        }

        if (step % 1000 == 0) printf("  [NPU] Iteration %d processed...\n", step);
    }
    printf(">>> NPU TRAINING LOOP COMPLETED! <<<\n\n");

    // On écrit le résultat final
    m_ioTensor.writeOutputTensors(graphIdx, 0, outputs, graphInfo.numOutputTensors, m_outputPath, 1);
    
    m_ioTensor.tearDownInputAndOutputTensors(inputs, outputs, graphInfo.numInputTensors, graphInfo.numOutputTensors);
  }
  return returnStatus;
}

} // namespace sample_app
