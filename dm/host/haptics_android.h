// Portable haptic dispatch for the Android backend. This is linked only in Android builds.
// All callbacks run on a worker, never under the NXNative HID/haptics lock.
#pragma once
#include "hle_haptics.h"
#include <stdint.h>

typedef void (*dm_android_haptics_output)(void *opaque, int player, int route_token,
                                           uint64_t epoch, int left, int right, int mono);

// Start before launching guest code. No route is advertised until set_route() succeeds.
// Only one backend instance may exist. Output owns no pointer into a snapshot.
int dm_haptics_android_start(dm_android_haptics_output output, void *opaque, int enabled);
// Route token is an opaque generation ID owned by the Android router, not an InputDevice ID.
// Switching routes clears the old game state and stops the previous output.
void dm_haptics_android_set_route(int player, int route_token, int available);
void dm_haptics_android_set_gain(float gain);  // 0..3, applied exactly once
void dm_haptics_android_suspend(unsigned reason, int on);
void dm_haptics_android_stop(void);            // synchronous STOP then join; no callbacks afterward
uint64_t dm_haptics_android_overflows(void);
#ifdef __ANDROID__
#include <jni.h>
// haptics_android_jni.c: HapticsRouter's natives bound to its class (a host loaded by NativeActivity, whose exported JNI names ART does not search). 1 when bound
int dm_haptics_android_register(JNIEnv *env, jclass router_class);
#endif
