set(APP "npu-trainer")
set(APP_SOURCES "QnnSampleApp.cpp"
                "main.cpp"
                "Log/Logger.cpp"
                "Log/LogUtils.cpp"
                "PAL/src/windows/Common.cpp"
                "PAL/src/windows/Directory.cpp"
                "PAL/src/windows/DynamicLoading.cpp"
                "PAL/src/windows/FileOp.cpp"
                "PAL/src/windows/Path.cpp"
                "PAL/src/common/GetOpt.cpp"
                "PAL/src/common/StringOp.cpp"
                "Utils/DataUtil.cpp"
                "Utils/DynamicLoadUtil.cpp"
                "Utils/IOTensor.cpp"
                "Utils/QnnSampleAppUtils.cpp"
                "WrapperUtils/QnnWrapperUtils.cpp")

add_executable(${APP} ${APP_SOURCES})

target_compile_definitions(${APP} PUBLIC "-DNOMINMAX")
target_link_libraries(${APP} PRIVATE Shlwapi Shell32)

# On retire les flags MSVC /MT
# On ajoute les headers QNN
target_include_directories(${APP} PUBLIC 
    CachingUtil
    Log
    PAL/include
    Utils
    WrapperUtils
    ${CMAKE_BINARY_DIR}
    "C:/Users/ncouf/bitnet/qairt_sdk/qairt/2.26.2.240911/include/QNN"
    ./
)
