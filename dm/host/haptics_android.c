// Android adaptation of the existing nn::hid persistent vibration state.
// The guest-facing sink only enqueues fixed-size snapshots. Binder/JNI work is performed
// exclusively by the worker. Separate route tokens reject stale controller assignments.
#include "haptics_android.h"
#include <math.h>
#include <pthread.h>
#include <string.h>

enum { QUEUE_CAP = 256 };
typedef struct {
    dm_haptic_state state;
    uint64_t route_generation;
    int route_token;
    int stop;
} event;
typedef struct {
    int token, available;
    uint64_t generation;
} route;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready = PTHREAD_COND_INITIALIZER;
static pthread_t thread;
static event queue[QUEUE_CAP];
static unsigned head, length;
static route routes[DM_PLAYERS];
static dm_android_haptics_output emit;
static void *emit_context;
static float gain = 1;
static uint64_t overflows;
static int active, closing, worker_running;

static int good_player(int p) { return p >= 0 && p < DM_PLAYERS; }

// Called only with mu held. Stops are never dropped when the queue is congested:
// evict a non-stop snapshot, otherwise compress adjacent stops of the same target.
static void queue_event(event e) {
    if (length == QUEUE_CAP) {
        unsigned victim = QUEUE_CAP;
        for (unsigned i = 0; i < length; i++) {
            if (!queue[(head + i) % QUEUE_CAP].stop) { victim = i; break; }
        }
        if (victim == QUEUE_CAP) {
            // These are all route STOPs: retain the newest stop for each route.
            // A matching later STOP has equivalent effect; don't discard an unrelated STOP.
            for (unsigned i = 0; i < length; i++) {
                event *old = &queue[(head + i) % QUEUE_CAP];
                if (old->state.player == e.state.player && old->route_token == e.route_token) return;
            }
            // Extremely pathological: 256 distinct outstanding route changes. Keep the
            // latest STOP; Android synchronously cancels routes on reassignment as well.
            victim = 0;
        }
        for (unsigned i = victim; i + 1 < length; i++)
            queue[(head + i) % QUEUE_CAP] = queue[(head + i + 1) % QUEUE_CAP];
        length--;
        overflows++;
    }
    queue[(head + length) % QUEUE_CAP] = e;
    length++;
    pthread_cond_signal(&ready);
}

static void stop_route_locked(int p, int token) {
    event e = { .state = { .player = p }, .route_token = token, .stop = 1 };
    queue_event(e);
}

static void sink(dm_haptic_state state) {
    if (!good_player(state.player)) return;
    pthread_mutex_lock(&mu);
    if (active && !closing) {
        route r = routes[state.player];
        if (r.available) {
            event e = { .state = state, .route_generation = r.generation,
                        .route_token = r.token };
            queue_event(e);
        }
    }
    pthread_mutex_unlock(&mu);
}

static int amount(float x) {
    if (!isfinite(x) || x <= 0) return 0;
    int n = (int)lroundf(fminf(1, x) * 255);
    return n < 1 ? 1 : n;
}

// An Android motor's strength: the largest amplitude of the motors' two bands, times gain. dm_haptics_level adds the bands' and motors' power (Core Haptics' phone mix), which
// reaches full strength at amplitude 0.5 on both bands: TTYD's weak, normal and strong rumble settings (peaks 0.5, 1, 1) all came out at full strength on an Xbox controller
static float strength(const dm_vibration *v, int n, float gain) {
    float a = 0;
    for (int i = 0; i < n; i++) {
        float lo = isfinite(v[i].amplitude_low) ? v[i].amplitude_low : 0, hi = isfinite(v[i].amplitude_high) ? v[i].amplitude_high : 0;
        a = fmaxf(a, fmaxf(lo, hi));
    }
    a = fminf(1, fmaxf(0, a));
    return isfinite(gain) && gain > 0 ? fminf(1, a * gain) : 0;
}

static void *run(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (!length && !closing) pthread_cond_wait(&ready, &mu);
        if (!length && closing) { pthread_mutex_unlock(&mu); break; }
        event e = queue[head]; head = (head + 1) % QUEUE_CAP; length--;
        route r = routes[e.state.player];
        float g = gain;
        dm_android_haptics_output output = emit;
        void *opaque = emit_context;
        pthread_mutex_unlock(&mu);

        if (e.stop) {
            output(opaque, e.state.player, e.route_token, 0, 0, 0, 0);
            continue;
        }
        // Do not hold mu while acquiring the haptics engine mutex: sink uses the opposite order.
        if (!r.available || e.route_generation != r.generation ||
            e.state.epoch != dm_haptics_epoch(e.state.player)) continue;
        float left = strength(e.state.motor, 1, g), right = strength(e.state.motor + 1, 1, g);
        // Serialized callback. Android checks the route token again under its routing lock.
        output(opaque, e.state.player, e.route_token, e.state.epoch,
               amount(left), amount(right), amount(fmaxf(left, right)));
    }
    return NULL;
}

int dm_haptics_android_start(dm_android_haptics_output output, void *opaque, int enabled) {
    if (!output) return 0;
    pthread_mutex_lock(&mu);
    if (active || worker_running) { pthread_mutex_unlock(&mu); return 0; }
    emit = output; emit_context = opaque;
    head = length = 0; overflows = 0; gain = 1;
    memset(routes, 0, sizeof routes);
    closing = 0; active = 1;
    worker_running = pthread_create(&thread, NULL, run, NULL) == 0;
    if (!worker_running) { active = 0; emit = NULL; emit_context = NULL; pthread_mutex_unlock(&mu); return 0; }
    pthread_mutex_unlock(&mu);
    dm_haptics_init(sink, enabled);
    return 1;
}

void dm_haptics_android_set_route(int player, int token, int available) {
    if (!good_player(player)) return;
    pthread_mutex_lock(&mu);
    if (!active || closing) { pthread_mutex_unlock(&mu); return; }
    route *r = &routes[player];
    if (r->available == !!available && (!available || r->token == token)) {
        pthread_mutex_unlock(&mu); return;
    }
    if (r->available) stop_route_locked(player, r->token);
    r->generation++;
    r->token = token;
    r->available = !!available;
    pthread_mutex_unlock(&mu);
    // Clears old persistent motor values even if a new controller supports vibration.
    dm_haptics_route(player, !!available);
}

void dm_haptics_android_set_gain(float value) {
    if (!isfinite(value) || value < 0) value = 0;
    if (value > 3) value = 3;
    pthread_mutex_lock(&mu);
    if (!active || closing || gain == value) { pthread_mutex_unlock(&mu); return; }
    gain = value;
    pthread_mutex_unlock(&mu);
    // Refresh current continuous state even if the game sends no new vibration command.
    for (int p = 0; p < DM_PLAYERS; p++) {
        dm_haptic_state s = { .player = p, .epoch = dm_haptics_epoch(p) };
        s.motor[0] = dm_haptics_actual(p, 0);
        s.motor[1] = dm_haptics_actual(p, 1);
        sink(s);
    }
}

void dm_haptics_android_suspend(unsigned reason, int on) {
    dm_haptics_suspend(reason, on);
    if (!on) return;
    pthread_mutex_lock(&mu);
    if (active && !closing) {
        for (int p = 0; p < DM_PLAYERS; p++) if (routes[p].available)
            stop_route_locked(p, routes[p].token);
    }
    pthread_mutex_unlock(&mu);
}

void dm_haptics_android_stop(void) {
    pthread_mutex_lock(&mu);
    if (!active || closing) { pthread_mutex_unlock(&mu); return; }
    // Block future snapshots before unregistering the sink; don't hold mu across haptics_init.
    active = 0;
    pthread_mutex_unlock(&mu);
    dm_haptics_init(NULL, 0);
    pthread_mutex_lock(&mu);
    head = length = 0;
    for (int p = 0; p < DM_PLAYERS; p++) if (routes[p].available)
        stop_route_locked(p, routes[p].token);
    closing = 1;
    pthread_cond_signal(&ready);
    pthread_mutex_unlock(&mu);
    pthread_join(thread, NULL);
    pthread_mutex_lock(&mu);
    worker_running = 0;
    emit = NULL; emit_context = NULL;
    closing = 0; length = 0;
    pthread_mutex_unlock(&mu);
}

uint64_t dm_haptics_android_overflows(void) {
    pthread_mutex_lock(&mu);
    uint64_t n = overflows;
    pthread_mutex_unlock(&mu);
    return n;
}
