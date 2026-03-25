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

size_t get_dtype_size(Qnn_DataType_t dt) {
    switch(dt) {
        case QNN_DATATYPE_FLOAT_32: return 4;
        case QNN_DATATYPE_FLOAT_16: return 2;
        case QNN_DATATYPE_INT_32: return 4;
        case QNN_DATATYPE_INT_16: return 2;
        case QNN_DATATYPE_INT_8: return 1;
        case QNN_DATATYPE_UINT_32: return 4;
        case QNN_DATATYPE_UINT_16: return 2;
        case QNN_DATATYPE_UINT_8: return 1;
        default: return 4;
    }
}

struct TensorPair {
    void* src;
    void* dst;
    size_t size;
};

int main(int argc, char** argv) {
    if (argc < 5) return 1;
    HMODULE h_sys = LoadLibraryA(argv[2]);
    auto getSys = (QnnSystemInterfaceGetProvidersFn_t)GetProcAddress(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sysProviders = nullptr;
    uint32_t nSys = 0; getSys(&sysProviders, &nSys);
    auto sysQnn = sysProviders[0]->v1_1;

    HMODULE h_htp = LoadLibraryExA(argv[1], NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    auto getHtp = (QnnInterfaceGetProvidersFn_t)GetProcAddress(h_htp, "QnnInterface_getProviders");
    const QnnInterface_t** providers = nullptr;
    uint32_t nProv = 0; getHtp(&providers, &nProv);
    auto qnn = providers[0]->v2_19;

    std::ifstream file(argv[3], std::ios::binary | std::ios::ate);
    size_t bin_size = file.tellg();
    std::vector<char> bin_buffer(bin_size);
    file.seekg(0, std::ios::beg); file.read(bin_buffer.data(), bin_size); file.close();

    QnnSystemContext_Handle_t sysCtx = nullptr;
    sysQnn.systemContextCreate(&sysCtx);
    const QnnSystemContext_BinaryInfo_t* binInfo = nullptr;
    Qnn_ContextBinarySize_t binInfoSize = 0;
    sysQnn.systemContextGetBinaryInfo(sysCtx, bin_buffer.data(), (uint64_t)bin_size, &binInfo, &binInfoSize);

    auto& g = binInfo->contextBinaryInfoV1.graphs[0].graphInfoV1;
    Qnn_BackendHandle_t backend = nullptr; qnn.backendCreate(nullptr, nullptr, &backend);
    Qnn_DeviceHandle_t device = nullptr; qnn.deviceCreate(nullptr, nullptr, &device);
    Qnn_ContextHandle_t context = nullptr;
    qnn.contextCreateFromBinary(backend, device, nullptr, bin_buffer.data(), (uint32_t)bin_size, &context, nullptr);
    Qnn_GraphHandle_t graph = nullptr;
    qnn.graphRetrieve(context, g.graphName, &graph);

    auto setup_tensors = [&](uint32_t num, Qnn_Tensor_t* src, Qnn_TensorType_t type) {
        std::vector<Qnn_Tensor_t> vt(num, QNN_TENSOR_INIT);
        for (uint32_t i=0; i<num; i++) {
            auto& s = src[i].v1;
            vt[i].version = QNN_TENSOR_VERSION_1;
            vt[i].v1.id = s.id;
            vt[i].v1.name = s.name;
            vt[i].v1.type = type;
            vt[i].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
            vt[i].v1.dataType = s.dataType;
            vt[i].v1.quantizeParams = s.quantizeParams;
            vt[i].v1.rank = s.rank;
            vt[i].v1.dimensions = (uint32_t*)malloc(s.rank * sizeof(uint32_t));
            size_t elements = 1;
            for(uint32_t r=0; r<s.rank; r++) {
                vt[i].v1.dimensions[r] = s.dimensions[r];
                elements *= s.dimensions[r];
            }
            vt[i].v1.memType = QNN_TENSORMEMTYPE_RAW;
            size_t bytes = elements * get_dtype_size(s.dataType);
            vt[i].v1.clientBuf.dataSize = (uint32_t)bytes;
            vt[i].v1.clientBuf.data = malloc(bytes);
            memset(vt[i].v1.clientBuf.data, 0, bytes);
        }
        return vt;
    };

    auto inputs = setup_tensors(g.numGraphInputs, g.graphInputs, QNN_TENSOR_TYPE_APP_WRITE);
    auto outputs = setup_tensors(g.numGraphOutputs, g.graphOutputs, QNN_TENSOR_TYPE_APP_READ);

    // PRE-MAP WEIGHT UPDATES (O(N) instead of O(N^2) strings)
    std::vector<TensorPair> feedback_map;
    for (auto& in : inputs) {
        for (auto& out : outputs) {
            if (strstr(out.v1.name, "new_") && strstr(out.v1.name, in.v1.name)) {
                feedback_map.push_back({out.v1.clientBuf.data, in.v1.clientBuf.data, in.v1.clientBuf.dataSize});
            }
        }
    }

    std::cout << "Starting Optimized Loop..." << std::endl;
    int num_steps = std::stoi(argv[4]);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < num_steps; s++) {
        if (qnn.graphExecute(graph, inputs.data(), (uint32_t)inputs.size(), outputs.data(), (uint32_t)outputs.size(), nullptr, nullptr) != QNN_SUCCESS) break;
        for (auto& pair : feedback_map) {
            memcpy(pair.dst, pair.src, pair.size);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    if (ms == 0) ms = 1;
    std::cout << "RESULT: " << (float)num_steps*1000.0f/ms << " it/s" << std::endl;

    return 0;
}
