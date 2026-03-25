#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <windows.h>

#ifdef interface
#undef interface
#endif

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);

extern "C" __declspec(dllexport) void run_npu_training_no_sys(const char* htp_path, const char* bin_path, int steps) {
    std::cout << "[DLL] In-Memory Trainer Start (No-Sys Mode)" << std::endl;

    HMODULE h_htp = LoadLibraryExA(htp_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if(!h_htp) { std::cout << "Fail HTP " << GetLastError() << std::endl; return; }
    auto getHtp = (QnnInterfaceGetProvidersFn_t)GetProcAddress(h_htp, "QnnInterface_getProviders");
    const QnnInterface_t** htpProviders = nullptr;
    uint32_t nHtp = 0;
    getHtp(&htpProviders, &nHtp);
    auto qnn = htpProviders[0]->v2_19;

    std::ifstream file(bin_path, std::ios::binary | std::ios::ate);
    size_t bin_size = file.tellg();
    std::vector<char> bin_buffer(bin_size);
    file.seekg(0, std::ios::beg);
    file.read(bin_buffer.data(), bin_size);
    file.close();

    Qnn_BackendHandle_t backend = nullptr; qnn.backendCreate(nullptr, nullptr, &backend);
    Qnn_DeviceHandle_t device = nullptr; qnn.deviceCreate(nullptr, nullptr, &device);
    Qnn_ContextHandle_t context = nullptr;
    qnn.contextCreateFromBinary(backend, device, nullptr, bin_buffer.data(), (uint32_t)bin_size, &context, nullptr);
    
    // On assume le nom du graphe car on n'a pas QnnSystem pour le parser
    const char* graph_name = "qnn_bitnet_choc";
    Qnn_GraphHandle_t graph = nullptr;
    if (qnn.graphRetrieve(context, graph_name, &graph) != QNN_SUCCESS) {
        std::cout << "Fail Graph Retrieve: " << graph_name << std::endl;
        return;
    }

    // On setup manuellement quelques tenseurs pour le speed test
    // Note: Dans un vrai train, on lirait les dimensions du bin, ici on simule la charge de travail
    // Pour bitnet_choc: ~6 inputs, ~4 outputs
    std::vector<Qnn_Tensor_t> inputs(6, QNN_TENSOR_INIT);
    std::vector<Qnn_Tensor_t> outputs(4, QNN_TENSOR_INIT);
    // ... setup minimal ... (On va juste mesurer graphExecute si possible)

    std::cout << "[DLL] LOOPING..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        // executeRes sera probablement une erreur car les tenseurs sont mal mappés
        // mais on veut juste voir si on peut appeler la fonction sans crash
        qnn.graphExecute(graph, inputs.data(), 6, outputs.data(), 4, nullptr, nullptr);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "[DLL] DONE in " << ms << "ms" << std::endl;
}
