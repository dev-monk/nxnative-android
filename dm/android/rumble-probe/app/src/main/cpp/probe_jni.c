// Debug-only: synthesize game-side nn::hid commands through the real haptics engine.
#include <jni.h>
#include "hle_haptics.h"
JNIEXPORT void JNICALL Java_dev_nxnative_probe_MainActivity_nativeSend(
        JNIEnv *env, jclass cls, jint motor, jfloat amplitude) {
    (void)env; (void)cls;
    if (motor < 0 || motor >= 2) return;
    dm_vibration_update u = { 0, motor, { amplitude, 160, amplitude * .3f, 320 } };
    dm_haptics_send(&u, 1);
}
JNIEXPORT void JNICALL Java_dev_nxnative_probe_MainActivity_nativeBoth(
        JNIEnv *env, jclass cls, jfloat amplitude) {
    (void)env; (void)cls;
    dm_vibration_update u[2] = {
        { 0, 0, { amplitude, 160, 0, 320 } },
        { 0, 1, { amplitude, 160, 0, 320 } },
    };
    dm_haptics_send(u, 2);
}
