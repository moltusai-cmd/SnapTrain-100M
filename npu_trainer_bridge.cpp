#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstring>
#include <windows.h>

#ifdef interface
#undef interface
#endif

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t***, uint32_t*);

extern "C" __declspec(dllexport) void run_npu_training(const char* htp_path, const char* sys_path, const char* bin_path, int steps) {
    std::cout << "[DLL] In-Memory Trainer Start" << std::endl;

    HMODULE h_sys = LoadLibraryA(sys_path);
    auto getSys = (QnnSystemInterfaceGetProvidersFn_t)GetProcAddress(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sysProviders = nullptr;
    uint32_t nSys = 0; getSys(&sysProviders, &nSys);
    auto sysQnn = sysProviders[0]->v1_1;

    HMODULE h_htp = LoadLibraryExA(htp_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    auto getHtp = (QnnInterfaceGetProvidersFn_t)GetProcAddress(h_htp, "QnnInterface_getProviders");
    const QnnInterface_t** htpProviders = nullptr;
    uint32_t nHtp = 0; getHtp(&htpProviders, &nHtp);
    auto qnn = htpProviders[0]->v2_19;

    std::ifstream file(bin_path, std::ios::binary | std::ios::ate);
    size_t bin_size = file.tellg();
    std::vector<char> bin_buffer(bin_size);
    file.seekg(0, std::ios::beg); file.read(bin_buffer.data(), bin_size); file.close();

    QnnSystemContext_Handle_t sysCtx = nullptr;
    sysQnn.systemContextCreate(&sysCtx);
    const QnnSystemContext_BinaryInfo_t* binInfo = nullptr;
    Qnn_ContextBinarySize_t binInfoSize = 0;
    sysQnn.systemContextGetBinaryInfo(sysCtx, bin_buffer.data(), (uint64_t)bin_size, &binInfo, &binInfoSize);

    auto& g = binInfo->contextBinaryInfoV1.graphs[0].graphInfoV1;
    std::cout << "[DLL] Graph: " << g.graphName << std::endl;

    Qnn_BackendHandle_t backend = nullptr; qnn.backendCreate(nullptr, nullptr, &backend);
    Qnn_DeviceHandle_t device = nullptr; qnn.deviceCreate(nullptr, nullptr, &device);
    Qnn_ContextHandle_t context = nullptr;
    qnn.contextCreateFromBinary(backend, device, nullptr, bin_buffer.data(), (uint32_t)bin_size, &context, nullptr);
    Qnn_GraphHandle_t graph = nullptr;
    qnn.graphRetrieve(context, g.graphName, &graph);

    // Setup tensors by copying pointers from binInfo
    std::vector<Qnn_Tensor_t> inputs(g.numGraphInputs);
    std::vector<Qnn_Tensor_t> outputs(g.numGraphOutputs);

    for (uint32_t i=0; i<g.numGraphInputs; i++) {
        inputs[i] = g.graphInputs[i]; // Bitwise copy metadata
        inputs[i].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        inputs[i].v1.memType = QNN_TENSORMEMTYPE_RAW;
        inputs[i].v1.clientBuf.data = malloc(inputs[i].v1.clientBuf.dataSize);
        memset(inputs[i].v1.clientBuf.data, 0, inputs[i].v1.clientBuf.dataSize);
    }

    for (uint32_t i=0; i<g.numGraphOutputs; i++) {
        outputs[i] = g.graphOutputs[i];
        outputs[i].v1.type = QNN_TENSOR_TYPE_APP_READ;
        outputs[i].v1.memType = QNN_TENSORMEMTYPE_RAW;
        outputs[i].v1.clientBuf.data = malloc(outputs[i].v1.clientBuf.dataSize);
    }

    std::cout << "[DLL] STARTING LOOP (" << steps << " steps)" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    int count = 0;
    for (int s = 0; s < steps; s++) {
        if (qnn.graphExecute(graph, inputs.data(), (uint32_t)inputs.size(), outputs.data(), (uint32_t)outputs.size(), nullptr, nullptr) != QNN_SUCCESS) break;
        count++;
        // Feedback
        for (auto& in : inputs) {
            for (auto& out : outputs) {
                if (strstr(out.v1.name, "new_") && strstr(out.v1.name, in.v1.name)) {
                    memcpy(in.v1.clientBuf.data, out.v1.clientBuf.data, in.v1.clientBuf.dataSize);
                }
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    if (ms == 0) ms = 1;
    std::cout << "[DLL] Executed " << count << " steps in " << ms << "ms" << std::endl;
    std::cout << "[DLL] Performance: " << (float)count * 1000.0f / ms << " it/s" << std::endl;
}
