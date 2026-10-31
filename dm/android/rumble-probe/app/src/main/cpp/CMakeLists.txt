cmake_minimum_required(VERSION 3.22.1)
project(nxrumble_probe C)
set(HOST "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../host")   # dm/host
add_library(nxrumble_probe SHARED
    "${HOST}/hle_haptics.c"
    "${HOST}/haptics_android.c"
    "${HOST}/haptics_android_jni.c"
    probe_jni.c)
target_include_directories(nxrumble_probe PRIVATE "${HOST}")
target_link_libraries(nxrumble_probe m log)
