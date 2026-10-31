// Portable integration tests for Android queue, JNI-free worker and NXNative vibration state.
#include "../haptics_android.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    int player, token, left, right, mono;
    uint64_t epoch;
} observed;
static observed seen[2048];
static int count;
static pthread_mutex_t captured = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t got = PTHREAD_COND_INITIALIZER;

static void receive(void *opaque, int player, int token, uint64_t epoch,
                    int left, int right, int mono) {
    (void)opaque;
    pthread_mutex_lock(&captured);
    if (count < (int)(sizeof seen / sizeof *seen))
        seen[count++] = (observed){ player, token, left, right, mono, epoch };
    pthread_cond_broadcast(&got);
    pthread_mutex_unlock(&captured);
}
static void until_count(int goal) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 2;
    pthread_mutex_lock(&captured);
    while (count < goal) {
        int rc = pthread_cond_timedwait(&got, &captured, &deadline);
        assert(rc == 0);
    }
    pthread_mutex_unlock(&captured);
}
static observed latest(void) {
    pthread_mutex_lock(&captured);
    assert(count > 0);
    observed o = seen[count - 1];
    pthread_mutex_unlock(&captured);
    return o;
}
static int total(void) {
    pthread_mutex_lock(&captured);
    int n = count;
    pthread_mutex_unlock(&captured);
    return n;
}
static void send(int player, int motor, float strength) {
    dm_vibration_update u = {player, motor, {strength, 160, 0, 320}};
    dm_haptics_send(&u, 1);
}
static void delay_ms(int n) {
    struct timespec t = {n / 1000, (n % 1000) * 1000000L};
    nanosleep(&t, NULL);
}
int main(void) {
    assert(dm_haptics_android_start(receive, NULL, 1));
    assert(!dm_haptics_android_start(receive, NULL, 1));
    assert(!dm_haptics_available());
    dm_haptics_android_set_route(0, 11, 1);
    assert(dm_haptics_available() == 1);
    int n = total();
    send(0, 0, .35f);
    until_count(n + 1);
    observed a = latest();
    assert(a.token == 11 && a.player == 0 && a.left > 0 && a.right == 0 && a.mono > 0);

    // State persists without another game HID command.
    delay_ms(100);
    assert(dm_haptics_actual(0, 0).amplitude_low == .35f);
    n = total();
    dm_haptics_android_set_gain(.5f);
    until_count(n + 1);
    observed b = latest();
    assert(b.left > 0 && b.left < a.left);
    dm_haptics_android_set_gain(1.0f);
    n = total();
    send(0, 1, .7f);
    until_count(n + 1);
    observed both = latest();
    assert(both.left > 0 && both.right > 0 && both.mono >= both.right);

    // Explicit stop of one motor keeps the second motor untouched.
    n = total();
    send(0, 0, 0);
    until_count(n + 1);
    observed right = latest();
    assert(right.left == 0 && right.right > 0);

    // Reassigning controllers stops old route and invalidates the original HID state.
    n = total();
    dm_haptics_android_set_route(0, 22, 1);
    until_count(n + 1);
    pthread_mutex_lock(&captured);
    int did_stop11 = 0;
    for (int i = n; i < count; i++) if (seen[i].token == 11 && seen[i].mono == 0) did_stop11 = 1;
    pthread_mutex_unlock(&captured);
    assert(did_stop11);
    assert(dm_haptics_actual(0, 1).amplitude_low == 0);
    n = total();
    send(0, 0, .4f);
    until_count(n + 1);
    assert(latest().token == 22 && latest().mono > 0);

    // Suspend cannot restart rumble on resume; the game must send a fresh command.
    n = total();
    dm_haptics_android_suspend(DM_HAPTICS_INACTIVE, 1);
    until_count(n + 1);
    delay_ms(30);
    assert(dm_haptics_actual(0, 0).amplitude_low == 0);
    dm_haptics_android_suspend(DM_HAPTICS_INACTIVE, 0);
    n = total();
    send(0, 0, .6f);
    until_count(n + 1);
    assert(latest().mono > 0);

    // Other players have no implicit output until routed.
    send(1, 0, 1);
    assert(!(dm_haptics_available() & 2));
    dm_haptics_android_set_route(1, 33, 1);
    n = total();
    send(1, 0, .25f);
    until_count(n + 1);
    assert(latest().player == 1 && latest().token == 33);

    // After shutdown the final operation for each token must be STOP.
    dm_haptics_android_stop();
    assert(!dm_haptics_available());
    pthread_mutex_lock(&captured);
    int stop22 = 0, stop33 = 0;
    for (int i = 0; i < count; i++) {
        if (seen[i].token == 22) stop22 = seen[i].mono == 0;
        if (seen[i].token == 33) stop33 = seen[i].mono == 0;
    }
    pthread_mutex_unlock(&captured);
    assert(stop22 && stop33);
    assert(dm_haptics_android_start(receive, NULL, 1));
    dm_haptics_android_stop();
    puts("android haptics dispatcher: ok");
}
