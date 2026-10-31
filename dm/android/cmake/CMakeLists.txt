# The host for Android arm64 (docs/android/PLAN.md 1.1): dmhost, headless (the adb loop), and libdmhost.so, the NativeActivity of dm/android/app. GPU_BACKEND picks the GPU backend: null
# (dm/gfx/null, the default, Phase 1) or vkn (dm/gfx/vkn, the Vulkan backend of Phase 2, which loads Vulkan at run time: no libvulkan link).
# Driven by build.sh beside this file; build outputs stay under build/ (.gitignore). Source lists follow dm/host/Makefile (SRC, GFX) minus its Objective-C files (nvn_mtl.m, nvn_refl.m,
# hle_vk.m, hle_gl.m, tex/nvn_deswizzle_gpu.m, memstat.m, haptics_apple.m, front_mac.m), plus the platform layer (dm/host/plat), the GPU backend (GPU_BACKEND) and a front end
# (dm/android/front). New hle_*.c / dm/gfx sources are picked up by the same wildcards as the Makefile's
cmake_minimum_required(VERSION 3.22)
project(nxhost C ASM)

set(GAME "" CACHE STRING "base game title ID: the host is built for one game profile (dm/games/<id>/), as make GAME=")
if(NOT GAME)
    message(FATAL_ERROR "GAME is not set: -DGAME=<base game title ID> (dm/games/<id>/game.tsv)")
endif()
get_filename_component(DM "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)   # dm/
set(GPU_BACKEND null CACHE STRING "GPU backend: null (dm/gfx/null) or vkn (dm/gfx/vkn, Vulkan)")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS null vkn)
if(GPU_BACKEND STREQUAL "null")
    set(GPU_SRC "${DM}/gfx/null/nvn_null.c")
elseif(GPU_BACKEND STREQUAL "vkn")
    file(GLOB GPU_SRC "${DM}/gfx/vkn/*.c")   # as the Makefile's GFX_VKN: what make GFX=vkn builds on the Mac
    list(SORT GPU_SRC)
else()
    message(FATAL_ERROR "GPU_BACKEND=${GPU_BACKEND}: null or vkn")
endif()
if(NOT EXISTS "${DM}/games/${GAME}/game.tsv")
    message(FATAL_ERROR "no game profile ${GAME} in dm/games (a provisional profile outside the repository is a Makefile-only path for now)")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(GEN "${CMAKE_BINARY_DIR}/gen")

# Game profile header: regenerated on every build, as the Makefile's FORCE rule (game.py writes it only when it changes)
add_custom_target(game_profile ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN}/${GAME}"
    COMMAND ${Python3_EXECUTABLE} "${DM}/games/game.py" header ${GAME} "${GEN}/${GAME}/game_profile.h"
    BYPRODUCTS "${GEN}/${GAME}/game_profile.h"
    COMMENT "game profile ${GAME}")
# The NVN counting trampolines in ELF spelling (the committed dm/gfx/nvn/nvn_thunks.S is the Mach-O one)
add_custom_command(OUTPUT "${GEN}/nvn_thunks_elf.S"
    COMMAND ${Python3_EXECUTABLE} "${DM}/gfx/tools/gen_protos.py" --elf "${DM}/gfx/nvn/names.txt" "${GEN}/nvn_thunks_elf.S"
    DEPENDS "${DM}/gfx/tools/gen_protos.py" "${DM}/gfx/nvn/names.txt"
    COMMENT "NVN trampolines (ELF)")

# Compiler flags as the Makefile's CFLAGS. _GNU_SOURCE: bionic declares renameat2, process_vm_readv, pthread_getattr_np and the like under it.
# Frame pointers kept (the crash and probe backtraces walk the x29 chain, as on Apple arm64, whose ABI always keeps it); no branch protection (no -mbranch-protection: the guest's code has no BTI landing pads)
set(DM_CFLAGS -O1 -g -Wall -Wno-unused-function -D_GNU_SOURCE -fno-omit-frame-pointer)
set(DM_INC "${DM}/gfx/nvn/include" "${DM}/third_party/vulkan-headers/include" "${GEN}/${GAME}" "${DM}/third_party/libvpx" "${DM}/third_party/stb_vorbis")

# libvpx (the VP9 decoder sources only; config/ is the shared arm64 configuration), a static library as in the Makefile
file(GLOB_RECURSE VPX_SRC "${DM}/third_party/libvpx/*.c")
add_library(vpx STATIC ${VPX_SRC})
target_compile_options(vpx PRIVATE -O2 -g)
target_include_directories(vpx PRIVATE "${DM}/third_party/libvpx" "${DM}/third_party/libvpx/config")

file(GLOB HLE_SRC "${DM}/host/hle_*.c")
file(GLOB GFX_SRC "${DM}/gfx/*.c" "${DM}/gfx/nvn/*.c" "${DM}/gfx/tex/*.c")
list(SORT HLE_SRC)
list(SORT GFX_SRC)
set(HOST_SRC
    "${DM}/host/loader.c" "${DM}/host/hle.c" "${DM}/host/crash.c" "${DM}/host/probe.c" "${DM}/host/thread.c" ${HLE_SRC} "${DM}/host/entry.S"
    "${DM}/host/plat/plat_linux.c"
    ${GFX_SRC} "${DM}/gfx/nvn/nvn_entry.S" "${GEN}/nvn_thunks_elf.S"
    ${GPU_SRC})
add_executable(dmhost ${HOST_SRC} "${DM}/host/main.c" "${DM}/android/front/front_headless.c")
add_dependencies(dmhost game_profile)
target_compile_options(dmhost PRIVATE ${DM_CFLAGS})
target_include_directories(dmhost PRIVATE ${DM_INC})
# --export-dynamic: dladdr names host functions in crash dumps and stack scans (Mach-O executables export theirs). 16 KB segment alignment: runs on 4 KB and 16 KB page kernels alike
target_link_options(dmhost PRIVATE -Wl,--export-dynamic -Wl,-z,max-page-size=16384)
target_link_libraries(dmhost PRIVATE vpx aaudio log m dl)
if(GPU_BACKEND STREQUAL "vkn")
    target_link_libraries(dmhost PRIVATE android)   # ANativeWindow_acquire / _release in vk_present.c (the adb route has no window, the calls are linked all the same)
endif()

# libdmhost.so: the same sources compiled as PIC, main.c's main as dm_main (front/host_main.c, as the iOS app builds it) and front_android.c (ANativeActivity_onCreate) in place of the headless front.
# Nothing may use initial-exec TLS (bionic refuses to dlopen such a library, docs/android/probes-cpu.md P0.2): the __thread variables stay general-dynamic (TLSDESC) under -fPIC.
# -Bsymbolic: the host's own names bind inside the library, as in the executable (a dlopen'd library is searched after the app's global group, where another library's die or module_of could win);
# the C library's (sigaction, sigprocmask) stay undefined, so ART's libsigchain still stands in front of them. No text relocations (-z text, lld's default, makes one a link error)
# haptics_android.c / haptics_android_jni.c: rumble through the app's HapticsRouter (docs/android/rumble.md); the headless dmhost has no activity, so no output, as before
add_library(dmhost_lib SHARED ${HOST_SRC} "${DM}/android/front/host_main.c" "${DM}/android/front/front_android.c" "${DM}/android/front/hotkey.c"
    "${DM}/host/haptics_android.c" "${DM}/host/haptics_android_jni.c")
set_target_properties(dmhost_lib PROPERTIES OUTPUT_NAME dmhost)
add_dependencies(dmhost_lib game_profile)
target_compile_options(dmhost_lib PRIVATE ${DM_CFLAGS})
target_include_directories(dmhost_lib PRIVATE ${DM_INC} "${DM}/host")
target_link_options(dmhost_lib PRIVATE -Wl,-soname,libdmhost.so -Wl,-Bsymbolic -Wl,-z,text -Wl,-z,max-page-size=16384)
target_link_libraries(dmhost_lib PRIVATE vpx aaudio android log m dl)

# Unit tests that run on the phone (build.sh --tests): the portable ones of dm/gfx/tests and dm/gfx/tex/tests, built as their Makefiles build them
set(T "${DM}/gfx/tests")
foreach(t t_hmap t_cover t_pace t_fstep)
    add_executable(${t} EXCLUDE_FROM_ALL "${T}/${t}.c")
    target_compile_options(${t} PRIVATE -O1 -Wall)
endforeach()
add_executable(t_dirty_slots EXCLUDE_FROM_ALL "${T}/t_dirty_slots.c" "${DM}/host/crash.c" "${DM}/gfx/nvn/nvn_dirty.c" "${DM}/host/plat/plat_linux.c")
add_executable(t_dirty_race EXCLUDE_FROM_ALL "${T}/t_dirty_race.c" "${DM}/host/crash.c" "${DM}/host/plat/plat_linux.c")
add_executable(t_dirty_fail EXCLUDE_FROM_ALL "${T}/t_dirty_fail.c" "${DM}/host/crash.c" "${DM}/host/plat/plat_linux.c")
add_executable(t_dirty_wait EXCLUDE_FROM_ALL "${T}/t_dirty_wait.c" "${DM}/host/crash.c" "${DM}/host/plat/plat_linux.c")
foreach(t t_dirty_slots t_dirty_race t_dirty_fail t_dirty_wait)
    add_dependencies(${t} game_profile)
    target_compile_options(${t} PRIVATE ${DM_CFLAGS})
    target_include_directories(${t} PRIVATE "${DM}/gfx/nvn/include" "${GEN}/${GAME}")
endforeach()
add_executable(t_tex EXCLUDE_FROM_ALL "${DM}/gfx/tex/tests/t_tex.c" "${DM}/gfx/tex/nvn_deswizzle.c" "${DM}/gfx/nvn/nvn_tex.c")
target_compile_options(t_tex PRIVATE -O2 -g -Wall -Wextra)
target_link_libraries(t_tex PRIVATE m)
add_executable(t_sha256 EXCLUDE_FROM_ALL "${DM}/host/plat/tests/t_sha256.c")   # the vendored SHA-256 that stands in for CommonCrypto (known digests; on the Mac it is checked against CommonCrypto too)
target_compile_options(t_sha256 PRIVATE -O1 -Wall)
# The signal guard of crash.c (Linux only): rescans at each thread start, the handshake signals left unblocked, android_run_on_all_threads returning
add_executable(t_sig_guard EXCLUDE_FROM_ALL "${DM}/host/tests/t_sig_guard.c" "${DM}/host/crash.c" "${DM}/host/plat/plat_linux.c")
add_dependencies(t_sig_guard game_profile)
target_compile_options(t_sig_guard PRIVATE ${DM_CFLAGS})
target_include_directories(t_sig_guard PRIVATE ${DM_INC})
target_link_libraries(t_sig_guard PRIVATE dl)
add_custom_target(tests DEPENDS t_hmap t_cover t_pace t_fstep t_dirty_slots t_dirty_race t_dirty_fail t_dirty_wait t_sig_guard t_tex t_sha256)
