// JNI entry points for dev.nxnative.haptics.HapticsRouter.
// Compile this source and haptics_android.c into the Android host's native library;
// do not compile these sources in the Apple-only Makefile.
// A library loaded by System.loadLibrary (the rumble probe) is found by these exported names; libdmhost.so is loaded by NativeActivity, which ART does not search for
// them, so the host binds them with dm_haptics_android_register (front_android.c, onCreate).
#include "haptics_android.h"
#include <jni.h>
#include <pthread.h>
#include <stdint.h>

static JavaVM *vm;
static jobject router;
static jmethodID deliver;
static pthread_key_t attachment;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static unsigned exception_count;

static void detach(void *attached) {
    if (attached && vm) (*vm)->DetachCurrentThread(vm);
}
static void make_key(void) { pthread_key_create(&attachment, detach); }

static void output(void *unused, int player, int token, uint64_t epoch,
                   int left, int right, int mono) {
    (void)unused;
    if (!vm || !router || !deliver) return;
    JNIEnv *env = NULL;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) == JNI_EDETACHED) {
        if ((*vm)->AttachCurrentThread(vm, (void **)&env, NULL) != JNI_OK) return;
        pthread_once(&once, make_key);
        pthread_setspecific(attachment, (void *)1);
    } else if (!env) return;
    (*env)->CallVoidMethod(env, router, deliver, (jint)player, (jint)token,
                           (jlong)epoch, (jint)left, (jint)right, (jint)mono);
    // Never leave a pending exception on an attached native thread.
    if ((*env)->ExceptionCheck(env)) {
        if (++exception_count <= 8 || exception_count % 64 == 0)
            (*env)->ExceptionDescribe(env);
        (*env)->ExceptionClear(env);
    }
}

JNIEXPORT jboolean JNICALL Java_dev_nxnative_haptics_HapticsRouter_nativeStart(
    JNIEnv *env, jclass cls, jobject instance, jboolean enabled) {
    (void)cls;
    if (router) return JNI_FALSE;
    if ((*env)->GetJavaVM(env, &vm) != JNI_OK) return JNI_FALSE;
    exception_count = 0;
    jclass klass = (*env)->GetObjectClass(env, instance);
    if (!klass) return JNI_FALSE;
    deliver = (*env)->GetMethodID(env, klass, "applyFromNative", "(IIJIII)V");
    (*env)->DeleteLocalRef(env, klass);
    if (!deliver) return JNI_FALSE;
    router = (*env)->NewGlobalRef(env, instance);
    if (!router) return JNI_FALSE;
    if (!dm_haptics_android_start(output, NULL, enabled)) {
        (*env)->DeleteGlobalRef(env, router); router = NULL; deliver = NULL;
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_dev_nxnative_haptics_HapticsRouter_nativeRoute(
    JNIEnv *env, jclass cls, jint player, jint token, jboolean available) {
    (void)env; (void)cls;
    dm_haptics_android_set_route(player, token, available);
}
JNIEXPORT void JNICALL Java_dev_nxnative_haptics_HapticsRouter_nativeGain(
    JNIEnv *env, jclass cls, jfloat gain) {
    (void)env; (void)cls;
    dm_haptics_android_set_gain(gain);
}
JNIEXPORT void JNICALL Java_dev_nxnative_haptics_HapticsRouter_nativeSuspend(
    JNIEnv *env, jclass cls, jint reason, jboolean suspended) {
    (void)env; (void)cls;
    dm_haptics_android_suspend((unsigned)reason, suspended);
}
JNIEXPORT void JNICALL Java_dev_nxnative_haptics_HapticsRouter_nativeStop(
    JNIEnv *env, jclass cls) {
    (void)cls;
    dm_haptics_android_stop();
    if (router) (*env)->DeleteGlobalRef(env, router);
    router = NULL; deliver = NULL;
    vm = NULL;
}
JNIEXPORT jlong JNICALL Java_dev_nxnative_haptics_HapticsRouter_nativeOverflows(
    JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    return (jlong)dm_haptics_android_overflows();
}

// The natives above bound to router (HapticsRouter's class, from the app's class loader): 1 when bound
int dm_haptics_android_register(JNIEnv *env, jclass router_class) {
    JNINativeMethod m[] = {
        { "nativeStart", "(Ldev/nxnative/haptics/HapticsRouter;Z)Z", (void *)Java_dev_nxnative_haptics_HapticsRouter_nativeStart },
        { "nativeRoute", "(IIZ)V", (void *)Java_dev_nxnative_haptics_HapticsRouter_nativeRoute },
        { "nativeGain", "(F)V", (void *)Java_dev_nxnative_haptics_HapticsRouter_nativeGain },
        { "nativeSuspend", "(IZ)V", (void *)Java_dev_nxnative_haptics_HapticsRouter_nativeSuspend },
        { "nativeStop", "()V", (void *)Java_dev_nxnative_haptics_HapticsRouter_nativeStop },
        { "nativeOverflows", "()J", (void *)Java_dev_nxnative_haptics_HapticsRouter_nativeOverflows },
    };
    int ok = router_class && !(*env)->RegisterNatives(env, router_class, m, (jint)(sizeof m / sizeof *m));
    if ((*env)->ExceptionCheck(env)) { (*env)->ExceptionClear(env); ok = 0; }
    return ok;
}
