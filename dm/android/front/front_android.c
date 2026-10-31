// Android app front end (front.h; docs/android/PLAN.md 1.6, 1.7): libdmhost.so in a NativeActivity (dm/android/app; its Java class HostActivity adds the software keyboard's input box and,
// with TouchPad, the touch gamepad).
// It does what the iOS app's main.m and front_ios.m do:
// the environment, the directories and the log are set before the host starts, dm_main runs on a big-stack thread, the activity's lifecycle drives focus, audio, the guest clock and graphics,
// and the activity's input queue feeds player 1's controller path (dm_pad_set) and the keyboard / touch path (dm_input_set), where the touch gamepad's controls go too; its other touches
// feed the game's touch screen (dm_touch_set). Every callback here runs on the activity's main (UI) thread; the host calls
// it makes are lock-free (hle_hid.c) or take short host locks (hle_oe.c, hle_audio.c, hle_os.c, hle_touch.c, the input box's), and no Java is called from them (JNI calls stay on host-owned threads, PLAN.md 1.6).
// The window is recorded for the Vulkan backend (front_native_window, Phase 2), whose swapchain draws it; the null backend's build paints it black, as nothing else draws to it
#include "dm.h"
#include "hle.h"
#include "front.h"
#include "dm_input.h"
#include "game_profile.h"   // DM_GAME_ROMFS_MARKER
#include "haptics_android.h"
#include <android/asset_manager.h>
#include <android/input.h>
#include <android/keycodes.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_activity.h>
#include <android/native_window.h>
#include <android/window.h>
#include <ctype.h>
#include <fcntl.h>
#include <ftw.h>
#include <jni.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/stat.h>

int dm_main(int argc, char **argv);   // dm/host/main.c built as dm_main (host_main.c)

static ANativeActivity *g_act;
static char g_files[512], g_ext[512], g_log[600];
static char *g_argv[3];
static _Atomic int g_host_on;          // the host thread was started (game files found): the lifecycle and input reach the host only then
static _Atomic(ANativeWindow *) g_win;

// ---------- front.h ----------
void front_run(void (*guest)(void)) { guest(); }   // already on the host thread (host_thread below), as on iOS
void *front_metal_layer(void) { return NULL; }
void *front_native_window(void) { return atomic_load(&g_win); }   // the activity's window, NULL while it has none (the Vulkan backend's surface, Phase 2)
int front_screen_max_fps(void) { return 0; }        // not known yet (Phase 2: the display modes)
void front_screen_match(int fps) { (void)fps; }
// ---------- Text input box (front.h): HostActivity.showTextInput (dm/android/app/java), an AlertDialog the system keyboard types into ----------
// The calling (guest) thread starts a host-owned thread for the JNI call (PLAN.md 1.6: ART cannot run on guest stacks), which attaches, asks for the box and detaches; then it waits for
// nativeTextDone (tx_done), which the UI thread calls once the box is answered. One box at a time. -1 (taken as a cancel) only when HostActivity's method or its native could not be bound
// in onCreate: not expected, since the manifest names HostActivity and an APK without its class does not start
extern int g_swkbd_min_chars;   // hle_swkbd.c: this input's minimum length (the applet keeps OK disabled until the text is long enough), -1 for none
static void input_release(void);
static jmethodID g_tx_show;    // HostActivity.showTextInput, set in onCreate (text_box_init)
static pthread_mutex_t g_tx_one = PTHREAD_MUTEX_INITIALIZER, g_tx_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tx_cv = PTHREAD_COND_INITIALIZER;
static int64_t g_tx_token; static int g_tx_state;   // the box asked for, and its answer under g_tx_mu: 0 none yet, 1 OK, 2 cancel
static char *g_tx_out; static int g_tx_cap;
typedef struct { const char *title, *sub, *guide, *initial; int max, min; int64_t token; int ok; } tx_req;
static void *tx_call(void *p) {   // its own stack, not the guest's: the message is built here
    tx_req *q = p; JNIEnv *env; JavaVM *vm = g_act->vm;
    char msg[1280]; snprintf(msg, sizeof msg, "%s%s%s", q->sub, *q->sub && *q->guide ? "\n" : "", q->guide);   // sub (129 UTF-16 units) and guide (257) are at most 1159 bytes of UTF-8: never cut
    const char *str[4] = { q->title, msg, q->guide, q->initial };
    if ((*vm)->AttachCurrentThread(vm, &env, NULL) != JNI_OK) return NULL;
    jbyteArray b[4] = { 0 }; int nb = 0;
    for (; nb < 4; nb++) {   // a failed allocation leaves an exception pending: no other JNI call until it is cleared below
        jsize n = (jsize)strlen(str[nb]);
        if (!(b[nb] = (*env)->NewByteArray(env, n))) break;
        (*env)->SetByteArrayRegion(env, b[nb], 0, n, (const jbyte *)str[nb]);
    }
    if (nb == 4) (*env)->CallVoidMethod(env, g_act->clazz, g_tx_show, (jlong)q->token, b[0], b[1], b[2], b[3], (jint)q->max, (jint)q->min);
    q->ok = nb == 4 && !(*env)->ExceptionCheck(env);
    if ((*env)->ExceptionCheck(env)) { (*env)->ExceptionDescribe(env); (*env)->ExceptionClear(env); }
    for (int i = 0; i < 4; i++) if (b[i]) (*env)->DeleteLocalRef(env, b[i]);
    (*vm)->DetachCurrentThread(vm);
    return NULL;
}
static void JNICALL tx_done(JNIEnv *env, jclass c, jlong token, jboolean ok, jbyteArray text) {   // the UI thread
    (void)c;
    pthread_mutex_lock(&g_tx_mu);
    if (token == g_tx_token && !g_tx_state) {
        int len = ok && text ? (int)(*env)->GetArrayLength(env, text) : 0, n = len < g_tx_cap - 1 ? len : g_tx_cap - 1;
        if (n > 0) (*env)->GetByteArrayRegion(env, text, 0, n + (n < len), (jbyte *)g_tx_out);   // one byte past a cut, to see where the cut falls
        while (n > 0 && n < len && ((unsigned char)g_tx_out[n] & 0xC0) == 0x80) n--;   // cut short: at a character's start
        g_tx_out[n] = 0;
        g_tx_state = ok ? 1 : 2; pthread_cond_signal(&g_tx_cv);
    }
    pthread_mutex_unlock(&g_tx_mu);
    // As the Mac box: the box took the keys and touches, so the game never saw the release of the press that opened it; nothing may stay pressed when it closes (onWindowFocusChanged also
    // releases them as the box takes focus)
    input_release();
}
static void text_box_init(ANativeActivity *act) {   // onCreate, the UI thread
    JNIEnv *env = act->env; jclass c = (*env)->GetObjectClass(env, act->clazz);
    jmethodID show = c ? (*env)->GetMethodID(env, c, "showTextInput", "(J[B[B[B[BII)V") : NULL;
    JNINativeMethod m = { "nativeTextDone", "(JZ[B)V", (void *)tx_done };
    if (show && !(*env)->ExceptionCheck(env) && !(*env)->RegisterNatives(env, c, &m, 1)) g_tx_show = show;
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    if (c) (*env)->DeleteLocalRef(env, c);
    if (!g_tx_show) fprintf(stderr, "boot: [RED] HostActivity.showTextInput or its native nativeTextDone could not be bound: software keyboards are cancelled, no box is shown\n");
}
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    if (!g_tx_show || !out || out_cap < 1) return -1;
    pthread_mutex_lock(&g_tx_one);
    pthread_mutex_lock(&g_tx_mu);
    tx_req q = { title ? title : "", sub ? sub : "", guide ? guide : "", initial ? initial : "", max_chars > 0 ? max_chars : 0, g_swkbd_min_chars > 0 ? g_swkbd_min_chars : 0, ++g_tx_token, 0 };
    g_tx_state = 0; g_tx_out = out; g_tx_cap = out_cap;
    pthread_mutex_unlock(&g_tx_mu);
    pthread_t t; int r = -1;
    LOG("front end: input box %lld asked for (title '%s', at most %d and at least %d characters; 0: no limit)", (long long)q.token, q.title, q.max, q.min);
    if (!pthread_create(&t, NULL, tx_call, &q)) { pthread_join(t, NULL); r = q.ok ? 0 : -1; }
    pthread_mutex_lock(&g_tx_mu);
    if (r == 0) { while (!g_tx_state) pthread_cond_wait(&g_tx_cv, &g_tx_mu); r = g_tx_state == 1; }
    else LOG("front end: [RED] the input box could not be shown (the JNI call failed): taken as a cancel");
    g_tx_out = NULL; g_tx_cap = 0; g_tx_state = 2;   // a late answer to this box is ignored (tx_done checks the state)
    pthread_mutex_unlock(&g_tx_mu);
    pthread_mutex_unlock(&g_tx_one);
    return r;
}

// ---------- Log: <files>/log.txt (the previous run's kept as log.prev.txt, as on iOS) and logcat, tag "dmhost" ----------
// The host writes only to stderr: fd 2 itself becomes the file, so the host, the crash handler's raw write(2, …) and any library land there synchronously and a crash loses nothing.
// A thread follows the file and copies its lines to logcat (the end of a crash may miss logcat; the file has it).
// Constructors run while the activity dlopens this library, before onCreate, and some log (hle_nvn.c: the null-device line); fd 2 is the zygote's /dev/null then. Priority 101 runs early_log
// before every default-priority constructor of the library: it opens log.txt as fd 2 at once in the files directory derived the way Android lays it out (files_derive; NativeActivity's
// onCreate makes it, as getFilesDir, in the arguments of the call that dlopens this library), so every line lands in order and nothing is copied, then reads env.txt (env_early, below) for
// the constructors that read the environment. Only if that fails does fd 2 become a memory file that log_open copies to the head of log.txt: best effort (a line written during the copy may
// be lost or split), and said so in red
static char g_files_early[512];   // the derived files directory ("" = none)
static int g_log_early, g_early_err;   // early_log made fd 2 log.txt; else why not (errno, 0 = no directory derived)
static int g_early_fd = -1;   // the memory file fd 2 is instead
static off_t g_early_at;   // how much of the memory file has been copied out
static void env_early(void);
static int files_derive(char *dir, size_t cap) {   // /data/user/<uid / 100000>/<package, the process name in /proc/self/cmdline>/files; 0 = not an app's process
    char pkg[256]; int fd = open("/proc/self/cmdline", O_RDONLY | O_CLOEXEC); ssize_t n = fd >= 0 ? read(fd, pkg, sizeof pkg - 1) : -1;
    if (fd >= 0) close(fd);
    if (n <= 0) return 0;
    pkg[n] = 0; char *c = strchr(pkg, ':'); if (c) *c = 0;   // "<package>:<name>" for a component's own process
    if (!*pkg || strchr(pkg, '/')) return 0;   // not an app's process name
    snprintf(dir, cap, "/data/user/%u/%s/files", (unsigned)(getuid() / 100000), pkg);
    return 1;
}
static int g_keep_err;   // log_create could not keep the last log as log.prev.txt (errno): this run's is appended to it, said in red once the log is up
static off_t g_log_from;   // where this run's log starts in log.txt (the follower starts there)
static int log_create(const char *dir) {   // a fresh dir/log.txt (the last one kept as log.prev.txt), -1 = it cannot be opened (errno)
    char prev[600]; snprintf(g_log, sizeof g_log, "%s/log.txt", dir); snprintf(prev, sizeof prev, "%s/log.prev.txt", dir);
    int keep = rename(g_log, prev) && errno != ENOENT;   // ENOENT: no last log. Else (log.prev.txt a directory, say) emptying log.txt would lose the last log unseen: append to it
    g_keep_err = keep ? errno : 0;
    int fd = open(g_log, O_WRONLY | O_CREAT | (keep ? 0 : O_TRUNC) | O_APPEND | O_CLOEXEC, 0644);   // appends: a copied early log and the host's lines never overwrite each other
    struct stat sb; g_log_from = fd >= 0 && keep && !fstat(fd, &sb) ? sb.st_size : 0;
    return fd;
}
__attribute__((constructor(101))) static void early_log(void) {
    if (files_derive(g_files_early, sizeof g_files_early)) {
        int fd = log_create(g_files_early);
        if (fd < 0) g_early_err = errno; else { g_log_early = dup2(fd, 2) >= 0; if (!g_log_early) g_early_err = errno; if (fd != 2) close(fd); }
    }
    if (g_log_early) setvbuf(stderr, NULL, _IOLBF, 0);   // LOG's text and its newline in one write
    else { int fd = memfd_create("dmhost-early-log", MFD_CLOEXEC); if (fd >= 0 && dup2(fd, 2) >= 0) g_early_fd = fd; else if (fd >= 0) close(fd); }
    env_early();
}
// The memory file past g_early_at, to log.txt's descriptor or (-1) to logcat. Whole lines only unless all: a write still being copied in shows its first page's part first (seen on the phone),
// and a line cut there would have a line written straight to the log in its middle
static void early_drain(int to, int all) {
    if (g_early_fd < 0) return;
    static char b[16384]; ssize_t n;
    while ((n = pread(g_early_fd, b, sizeof b - 1, g_early_at)) > 0) {
        if (!all) { while (n > 0 && b[n - 1] != '\n') n--; if (!n) break; }
        g_early_at += n;
        if (to >= 0) { if (write(to, b, (size_t)n) != n) break; continue; }
        b[n] = 0; for (char *s = b, *nl; *s; s = nl ? nl + 1 : s + strlen(s)) { if ((nl = strchr(s, '\n'))) *nl = 0; __android_log_write(ANDROID_LOG_INFO, "dmhost", s); if (!nl) break; }
    }
}
// The fallback's handoff: fd 2 goes to fd2 (log.txt, or /dev/null; -1 = left as it is) between drains: a line another thread writes after the first drain still lands in the memory file, and after
// the dup2 only a write that had already entered the kernel with it does (it holds the file; on the phone one allocating a page outlasted the second drain), so a last drain 10 ms on takes it. Then
// the file is closed. A LOG cut between its text and its newline by the dup2 is split
static void early_log_end(int to, int fd2) {
    early_drain(to, 0); if (fd2 >= 0) dup2(fd2, 2); early_drain(to, 0);
    if (g_early_fd >= 0) { usleep(10000); early_drain(to, 1); close(g_early_fd); g_early_fd = -1; }
}
static void log_nowhere(void) {   // no log.txt: what the constructors wrote goes to logcat, and fd 2 to /dev/null (not on into memory); one opened at load stays
    if (g_log_early) return;
    int fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
    early_log_end(-1, fd); if (fd >= 0) close(fd);
}
static void *log_follow(void *p) {
    (void)p;
    int fd = open(g_log, O_RDONLY | O_CLOEXEC); if (fd < 0) return NULL;
    if (g_log_from) lseek(fd, g_log_from, SEEK_SET);   // not the last run's log again
    static char b[4096]; size_t have = 0;
    for (;;) {
        ssize_t n = read(fd, b + have, sizeof b - 1 - have);
        if (n <= 0) { usleep(100000); continue; }   // at the end for now: the host writes more
        have += (size_t)n;
        char *s = b, *e = b + have, *nl;
        while ((nl = memchr(s, '\n', (size_t)(e - s)))) { *nl = 0; __android_log_write(ANDROID_LOG_INFO, "dmhost", s); s = nl + 1; }
        have = (size_t)(e - s); memmove(b, s, have);
        if (have == sizeof b - 1) { b[have] = 0; __android_log_write(ANDROID_LOG_INFO, "dmhost", b); have = 0; }   // a line longer than the buffer goes out in pieces
    }
}
static void log_open(void) {
    if (g_log_early) {   // fd 2 has been log.txt since the first constructor: no rename, no reopen, only the follower (from where this run's log starts)
        if (strcmp(g_files, g_files_early)) fprintf(stderr, "boot: [RED] the log is %s, derived at load, not in the activity's files directory\n", g_log);
    } else {
        int fd = log_create(g_files), mem = g_early_fd >= 0;
        if (fd < 0) { __android_log_print(ANDROID_LOG_ERROR, "dmhost", "boot: [RED] %s cannot be written (errno %d): the log goes nowhere (logcat gets only the constructors' lines kept in memory)", g_log, errno); log_nowhere(); return; }
        early_log_end(fd, fd); if (fd != 2) close(fd);   // the constructors' lines first, then fd 2 is log.txt
        setvbuf(stderr, NULL, _IOLBF, 0);
        char why[600]; if (*g_files_early) snprintf(why, sizeof why, "%s/log.txt: errno %d", g_files_early, g_early_err); else snprintf(why, sizeof why, "no files directory derived");
        fprintf(stderr, "boot: [RED] log.txt was not opened at load (%s): %s\n", why,
                mem ? "the constructors' lines were kept in memory and copied above, and a line written during the copy may be lost or split" : "the constructors' lines went nowhere");
    }
    if (g_keep_err) fprintf(stderr, "boot: [RED] log.txt could not be kept as log.prev.txt (errno %d): this run's log is appended to the previous one\n", g_keep_err);
    pthread_t t; if (!pthread_create(&t, NULL, log_follow, NULL)) pthread_detach(t);
}

// ---------- Environment, before the host starts (iOS main.m boot_env) ----------
// <files>/env.txt: one KEY=VALUE per line, # starts a comment; a variable already set wins (a sideloaded app gets no environment from outside: adb pushes this file, run_app.sh).
// Read in early_log, before the library's other constructors, as three of those read the environment once; the activity's internalDataPath is not known then, so the directory is the derived one
// (files_derive). boot_env checks it against internalDataPath and reads a file missed there late
static const char *const g_ctor_env[] = {"DM_NVN_LOG", "DM_NVN_PROF", "DM_FD_LOG"};   // read by constructors: hle_nvn.c nvn_init_slots and nvn_nops_register, hle_fdstat.c fdstat_init
static char g_env_keys[2048]; static size_t g_env_kl; static int g_env_early_read;
// The rule of a line (scripts/build_apk_from_nsp.py env_pairs reads an --env-file by the same rule before baking it into the APK, and dm/tools/tests/test_nsp_apk.py runs this function on
// the same lines): it ends at \n; spaces and tabs are cut at its start, at its end (with \r) and before the first =; # starts a comment; a line without = or without a name is skipped; a
// name already set is skipped (the first wins). A line over ENV_LINE_MAX bytes (the \n not counted) or with a NUL byte is ignored whole, in red, never read in pieces
#define ENV_LINE_MAX 4095
static void env_lines(FILE *f, const char *src, char *ctor, size_t ctor_cap) {   // ctor gets the constructor-read variables the lines set
    char *ln = NULL; size_t cap = 0, cl = 0; ssize_t len; int n = 0;
    while ((len = getline(&ln, &cap, f)) != -1) {
        n++;
        size_t body = (size_t)len - (len > 0 && ln[len - 1] == '\n');
        if (body > ENV_LINE_MAX || memchr(ln, 0, (size_t)len)) {
            fprintf(stderr, "boot: [RED] %s line %d %s: ignored\n", src, n, body > ENV_LINE_MAX ? "is longer than 4095 bytes" : "has a NUL byte"); continue;
        }
        char *s = ln, *e = ln + len;
        while (*s == ' ' || *s == '\t') s++;
        while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = 0;
        char *eq = strchr(s, '='), *ke = eq;
        if (!*s || *s == '#' || !eq) continue;
        while (ke > s && (ke[-1] == ' ' || ke[-1] == '\t')) ke--;
        if (ke == s) continue;
        *ke = 0; *eq = 0;
        if (getenv(s)) continue;
        setenv(s, eq + 1, 0);
        int w = snprintf(g_env_keys + g_env_kl, sizeof g_env_keys - g_env_kl, "%s%s", g_env_kl ? " " : "", s); g_env_kl = w > 0 && (size_t)w < sizeof g_env_keys - g_env_kl ? g_env_kl + (size_t)w : g_env_kl;
        for (size_t k = 0; ctor && k < sizeof g_ctor_env / sizeof *g_ctor_env; k++)
            if (!strcmp(s, g_ctor_env[k])) { int c = snprintf(ctor + cl, ctor_cap - cl, "%s%s", cl ? " " : "", s); cl = c > 0 && (size_t)c < ctor_cap - cl ? cl + (size_t)c : cl; }
        fprintf(stderr, "boot: env %s=%s (%s)\n", s, eq + 1, src);
    }
    free(ln);
    setenv("DM_ENV_FILE_KEYS", g_env_keys, 1);   // as iOS sets it for its launcher: what an env file set
}
static int env_file(const char *path, char *ctor, size_t ctor_cap) {   // 1 = the file was there
    FILE *f = fopen(path, "r"); if (!f) return 0;
    env_lines(f, "env.txt", ctor, ctor_cap); fclose(f);
    return 1;
}
static void env_early(void) {
    char ef[600]; if (!*g_files_early) return;
    snprintf(ef, sizeof ef, "%s/env.txt", g_files_early); g_env_early_read = env_file(ef, NULL, 0);
}
static int has_marker(const char *dir) { char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, DM_GAME_ROMFS_MARKER); return !access(p, R_OK); }
// The game files (RomFS): <external files>/romfs, the app's own external directory that adb can push to (the app cannot read /data/local/tmp), else <files>/romfs; only a directory holding
// the game profile's romfs_marker path counts (iOS romfs_locate). An env file's DM_ROMFS is taken as given
static const char *romfs_locate(char *buf, size_t n) {
    if (getenv("DM_ROMFS")) return getenv("DM_ROMFS");
    const char *base[2] = {g_ext, g_files};
    for (int i = 0; i < 2; i++) { if (!*base[i]) continue; snprintf(buf, n, "%s/romfs", base[i]); if (has_marker(buf)) return buf; }
    return NULL;
}
static void set_dir(const char *var, const char *sub) {   // a directory under <files>, made if missing, unless an env file set the variable
    char d[600]; snprintf(d, sizeof d, "%s/%s", g_files, sub); mkdir(d, 0700); setenv(var, d, 0);
}
// ---------- The game's package in the APK (scripts/build_apk_from_nsp.py; docs/android/PLAN.md Phase 3) ----------
// A game's APK carries assets/nx/: stamp (16 hex digits of the payload's content hash), index.txt (one "<size>\t<path>" line per file), env.txt (the build's defaults, below the user's env.txt),
// mods/ (patch_nso.py --target android), vksh/ (the Vulkan shader modules decompressed, their reflection, inputs.tsv and the shader map) and font/. Those files are unpacked into <files>/pkg/ on
// the host thread, before dm_main, whenever the APK's stamp is not the one unpacked there: into pkg.new, its stamp written last, then swapped in, so a start cut short leaves the last whole set
// (or none) and never a mix. The loader reads the modules into anonymous memory and makes that executable (loader.c; docs/android/probes.md P0.1), so no file has to be in nativeLibraryDir.
// The development APK (build_app.sh without --assets) has no assets/nx and runs as before: modules in <files>/mods, shaders from DM_VK_SHADERS in env.txt
static char g_pkg[32];   // the APK's payload stamp, "" = no payload
static char *asset_read(const char *name, size_t *n) {   // a small asset whole, NUL-terminated (NULL: none)
    AAsset *a = AAssetManager_open(g_act->assetManager, name, AASSET_MODE_BUFFER); if (!a) return NULL;
    off_t z = AAsset_getLength(a); char *b = z >= 0 && z < (1 << 24) ? malloc((size_t)z + 1) : NULL;
    if (b && AAsset_read(a, b, (size_t)z) != (int)z) { free(b); b = NULL; }
    AAsset_close(a); if (b) { b[z] = 0; if (n) *n = (size_t)z; }
    return b;
}
static void pkg_probe(void) {   // onCreate: the APK's stamp, checked to be 16 lowercase hex digits
    char *s = asset_read("nx/stamp", NULL); if (!s) return;
    size_t n = strspn(s, "0123456789abcdef");
    if (n == 16 && (!s[n] || s[n] == '\n')) memcpy(g_pkg, s, 16), g_pkg[16] = 0;
    else fprintf(stderr, "boot: [RED] the APK's assets/nx/stamp is not 16 hex digits: its package is not used\n");
    free(s);
}
static int rm_one(const char *p, const struct stat *sb, int fl, struct FTW *fw) { (void)sb; (void)fl; (void)fw; return remove(p) && errno != ENOENT ? -1 : 0; }
static void rm_tree(const char *p) { if (!access(p, F_OK)) nftw(p, rm_one, 16, FTW_DEPTH | FTW_PHYS); }
static int path_ok(const char *s) {   // an index path: relative, [A-Za-z0-9._-] parts, none empty or starting with '.'
    if (!*s || *s == '/') return 0;
    for (const char *p = s; *p; p++) {
        if ((p == s || p[-1] == '/') && (*p == '.' || *p == '/')) return 0;
        if (!(isalnum((unsigned char)*p) || *p == '.' || *p == '_' || *p == '-' || *p == '/')) return 0;
    }
    return s[strlen(s) - 1] != '/';
}
static int pkg_unpack(void) {   // the host thread: 1 = <files>/pkg holds the APK's payload
    char cur[600], nw[600], old[600], p[1200], have[32] = "";
    snprintf(cur, sizeof cur, "%s/pkg", g_files); snprintf(nw, sizeof nw, "%s/pkg.new", g_files); snprintf(old, sizeof old, "%s/pkg.old", g_files);
    rm_tree(nw); rm_tree(old);   // what a start cut short left
    snprintf(p, sizeof p, "%s/stamp", cur);
    FILE *sf = fopen(p, "r"); if (sf) { if (!fgets(have, sizeof have, sf)) *have = 0; fclose(sf); have[strcspn(have, "\n")] = 0; }
    if (!strcmp(have, g_pkg)) { fprintf(stderr, "boot: package %s: unpacked in %s\n", g_pkg, cur); return 1; }
    uint64_t t0 = now_ns(); size_t ni = 0; char *idx = asset_read("nx/index.txt", &ni), *buf = malloc(1 << 20);
    int nf = 0, ok = idx && buf && !mkdir(nw, 0700); uint64_t nb = 0;
    if (!ok) fprintf(stderr, "boot: [RED] package %s: %s\n", g_pkg, !idx ? "the APK has no assets/nx/index.txt" : !buf ? "no memory" : "cannot make pkg.new");
    for (char *l = idx, *e; ok && l && *l; l = e ? e + 1 : NULL) {
        e = strchr(l, '\n'); if (e) *e = 0;
        if (!*l) continue;
        char *tab = strchr(l, '\t'), *end; unsigned long long want = strtoull(l, &end, 10);
        if (!tab || end != tab || !path_ok(tab + 1)) { fprintf(stderr, "boot: [RED] package %s: index line '%.200s' is not \"<size>\\t<path>\"\n", g_pkg, l); ok = 0; break; }
        const char *rel = tab + 1;
        snprintf(p, sizeof p, "%s/%s", nw, rel);
        for (char *s = p + strlen(nw) + 1; (s = strchr(s, '/')); s++) { *s = 0; if (mkdir(p, 0700) && errno != EEXIST) ok = 0; *s = '/'; }
        char an[700]; snprintf(an, sizeof an, "nx/%s", rel);
        AAsset *a = ok ? AAssetManager_open(g_act->assetManager, an, AASSET_MODE_STREAMING) : NULL;   // streaming: a 200 MB module is inflated piece by piece, never whole in memory
        int fd = a ? open(p, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600) : -1; unsigned long long got = 0; int r = 0;
        while (fd >= 0 && (r = AAsset_read(a, buf, 1 << 20)) > 0) {
            for (int w = 0, k; w < r; w += k) if ((k = (int)write(fd, buf + w, (size_t)(r - w))) <= 0) { r = -1; break; }
            if (r < 0) break;
            got += (unsigned)r;
        }
        if (fd >= 0 && close(fd)) r = -1;
        if (a) AAsset_close(a);
        if (!a || fd < 0 || r < 0 || got != want) { fprintf(stderr, "boot: [RED] package %s: %s %s (errno %d; %llu of %llu bytes)\n", g_pkg, !a ? "no asset" : "cannot write", rel, errno, got, want); ok = 0; break; }
        nf++; nb += got;
    }
    free(idx); free(buf);
    if (ok) { snprintf(p, sizeof p, "%s/stamp", nw); FILE *f = fopen(p, "w"); ok = f && fprintf(f, "%s\n", g_pkg) > 0; if (f && fclose(f)) ok = 0; }
    if (ok && !access(cur, F_OK) && rename(cur, old)) ok = 0;
    if (ok && rename(nw, cur)) { ok = 0; rename(old, cur); }
    if (!ok) { fprintf(stderr, "boot: [RED] package %s could not be unpacked into %s (errno %d): the host does not start\n", g_pkg, cur, errno); rm_tree(nw); return 0; }
    rm_tree(old);
    fprintf(stderr, "boot: package %s: unpacked into %s (%d files, %llu bytes, %.2f s)%s%s\n", g_pkg, cur, nf, (unsigned long long)nb, (now_ns() - t0) / 1e9, *have ? "; it held " : "", have);
    return 1;
}
static int boot_env(void) {
    log_open();
    fprintf(stderr, "boot: files %s, external files %s\n", g_files, *g_ext ? g_ext : "(none)");
    char ef[600]; snprintf(ef, sizeof ef, "%s/env.txt", g_files);
    if (!g_env_early_read || strcmp(g_files, g_files_early)) {   // missed at load (another path, or no file then): set now, too late for what the constructors read
        char late[256] = "";
        if (env_file(ef, late, sizeof late) && *late)
            fprintf(stderr, "boot: [RED] env.txt was read only after the library's constructors (they looked in %s): %s took no effect\n", *g_files_early ? g_files_early : "(no directory)", late);
    }
    pkg_probe();
    if (*g_pkg) {   // the build's defaults (build_apk_from_nsp.py: the game language, --env-file), below env.txt: the first line that sets a variable wins
        size_t n = 0; char *t = asset_read("nx/env.txt", &n), late[256] = "";
        FILE *f = t && n ? fmemopen(t, n, "r") : NULL;
        if (f) { env_lines(f, "the APK's env.txt", late, sizeof late); fclose(f); }
        if (*late) fprintf(stderr, "boot: [RED] the APK's env.txt is read after the library's constructors: %s took no effect (put it in %s instead)\n", late, ef);
        free(t);
    }
    setenv("DM_ROOT", g_files, 0);
    set_dir("DM_SAVE", "save"); set_dir("DM_TEMP", "temp");
    // Real audio output by default, as on iOS: the host's default elsewhere is the silent sink (hle_audio.c real_audio: adb-shell runs stay opt-in with DM_AUDIO=1); DM_AUDIO=0 in env.txt silences the app
    setenv("DM_AUDIO", "1", 0);
    char d[600];
    if (*g_ext) {   // the shared-font stand-ins and add-on content, pushed beside the RomFS when a game needs them (iOS: <App>/dm/font, <App>/aoc)
        snprintf(d, sizeof d, "%s/font", g_ext); if (!access(d, R_OK)) setenv("DM_FONT_DIR", d, 0);
        snprintf(d, sizeof d, "%s/aoc", g_ext); if (!access(d, R_OK)) setenv("DM_AOC_DIR", d, 0);
    }
    if (*g_pkg) {   // the APK's own: the Vulkan shader modules and the generated button-icon font, unpacked before dm_main (host_thread)
        snprintf(d, sizeof d, "%s/pkg/vksh", g_files); setenv("DM_VK_SHADERS", d, 0);
        snprintf(d, sizeof d, "%s/pkg/font", g_files); setenv("DM_FONT_DIR", d, 0);
    }
    fprintf(stderr, "boot: game language %s\n", getenv("DM_LANG") ?: "en-US");
    char rb[600]; const char *romfs = romfs_locate(rb, sizeof rb);
    if (!romfs) { fprintf(stderr, "boot: game files not found (%s/romfs or %s/romfs holding %s): the host does not start\n", *g_ext ? g_ext : "<external files>", g_files, DM_GAME_ROMFS_MARKER); return 0; }
    fprintf(stderr, "boot: game files %s\n", romfs);
    setenv("DM_ROMFS", romfs, 0);
    // The rewritten modules (patch_nso.py --target android output): the APK's package, else pushed by run_app.sh
    if (*g_pkg) snprintf(d, sizeof d, "%s/pkg/mods", g_files);
    else { snprintf(d, sizeof d, "%s/mods", g_files); mkdir(d, 0700); }
    fprintf(stderr, "boot: modules %s\n", d);
    if (*g_pkg) fprintf(stderr, "boot: package %s in the APK: modules, Vulkan shaders (DM_VK_SHADERS %s) and font (DM_FONT_DIR %s)\n", g_pkg, getenv("DM_VK_SHADERS"), getenv("DM_FONT_DIR"));
    if (chdir(g_files)) fprintf(stderr, "boot: cannot enter %s (errno %d)\n", g_files, errno);
    g_argv[0] = strdup("dmhost"); g_argv[1] = strdup(d);
    return 1;
}
static void *host_thread(void *p) {
    (void)p;
    if (*g_pkg && !pkg_unpack()) { ANativeActivity_finish(g_act); return NULL; }   // off the UI thread: a first start inflates some 200 MB
    int rc = dm_main(2, g_argv);
    fprintf(stderr, "boot: dm_main returned %d\n", rc);
    ANativeActivity_finish(g_act);   // any thread may ask; onDestroy then ends the process
    return NULL;
}

// ---------- Input: the activity's input queue on the main thread's looper ----------
// Controllers map by position, as on the Mac and iPhone (front_mac.m pad_read, DmGamepad.m read_pad): Android names the face buttons by position too (BUTTON_A south, B east, X west, Y north),
// so south is the Switch's B and east its A. Player 1 only for now (every controller merged; the Mac's per-controller players, pad_slots.h, come later)
static uint64_t g_pad_keys, g_pad_hat, g_pad_trig, g_kbd;   // main thread only
static int32_t g_lx, g_ly, g_rx, g_ry;
static int32_t g_pad_dev = -1;   // the input device of the last controller motion (its sticks, hat and triggers above), -1 none: its removal lets them go (pad_removed)
static int g_w, g_a, g_s, g_d, g_input_log;
static uint64_t pad_key_bit(int32_t c) {
    switch (c) {
        case AKEYCODE_BUTTON_A: return DM_BTN_B; case AKEYCODE_BUTTON_B: return DM_BTN_A; case AKEYCODE_BUTTON_X: return DM_BTN_Y; case AKEYCODE_BUTTON_Y: return DM_BTN_X;
        case AKEYCODE_BUTTON_L1: return DM_BTN_L; case AKEYCODE_BUTTON_R1: return DM_BTN_R; case AKEYCODE_BUTTON_L2: return DM_BTN_ZL; case AKEYCODE_BUTTON_R2: return DM_BTN_ZR;
        case AKEYCODE_BUTTON_THUMBL: return DM_BTN_LS; case AKEYCODE_BUTTON_THUMBR: return DM_BTN_RS; case AKEYCODE_BUTTON_START: return DM_BTN_PLUS; case AKEYCODE_BUTTON_SELECT: return DM_BTN_MINUS;
        case AKEYCODE_DPAD_LEFT: return DM_BTN_LEFT; case AKEYCODE_DPAD_UP: return DM_BTN_UP; case AKEYCODE_DPAD_RIGHT: return DM_BTN_RIGHT; case AKEYCODE_DPAD_DOWN: return DM_BTN_DOWN;
    }
    return 0;
}
// Keyboard fallbacks, the Mac's layout (front_mac.m key_bit): J K U I = A B X Y, F G = stick presses, Q E Z C = L R ZL ZR, Enter / Tab = + / −, WASD = left stick, Space = shake; the arrows arrive as DPAD keys
static uint64_t kbd_key_bit(int32_t c) {
    switch (c) {
        case AKEYCODE_J: return DM_BTN_A; case AKEYCODE_K: return DM_BTN_B; case AKEYCODE_U: return DM_BTN_X; case AKEYCODE_I: return DM_BTN_Y;
        case AKEYCODE_F: return DM_BTN_LS; case AKEYCODE_G: return DM_BTN_RS;
        case AKEYCODE_Q: return DM_BTN_L; case AKEYCODE_E: return DM_BTN_R; case AKEYCODE_Z: return DM_BTN_ZL; case AKEYCODE_C: return DM_BTN_ZR;
        case AKEYCODE_ENTER: return DM_BTN_PLUS; case AKEYCODE_TAB: return DM_BTN_MINUS; case AKEYCODE_SPACE: return DM_BTN_SHAKE;
    }
    return 0;
}
// The touch gamepad's state (HostActivity's TouchPad, dm/android/app/java): what its controls hold, and the fingers on no control (the game's touch screen). Main thread only, as the rest of
// this section. The controls go on the keyboard / touch path with the keyboard (kbd_push, dm_input_set), as iOS publishes its touch controls on that path (DmGamepad.m publish); the controller
// keeps player 1's controller path (pad_push, dm_pad_set), which DM_KEYS steps share (hle_hid.c keys_main: a step's press, then dm_pad_set(0, ...)). The host merges the two paths when the game
// reads them (hle_hid.c fill: buttons ORed, each stick axis the one deflected more), so a DM_KEYS step's release never lets go of a touch control, nor a touch of a controller's key
static uint64_t g_touch_btn;
static int32_t g_touch_lx, g_touch_ly, g_touch_log_lx, g_touch_log_ly;
static int g_touch_n, g_touch_epoch;
static dm_touch g_touch_game[16];
static uint64_t pad_bits(void) { return g_pad_keys | g_pad_hat | g_pad_trig; }   // what player 1's controller path holds
static uint64_t kbd_bits(void) { return g_kbd | g_touch_btn; }                   // what the keyboard / touch path holds
static void pad_push(void) { dm_pad_set(pad_bits(), g_lx, g_ly, g_rx, g_ry); }
// The left stick: WASD's while any of them is held, else the touch stick's. The right stick has no keyboard or touch control yet
static void kbd_push(void) {
    int wasd = g_w || g_a || g_s || g_d;
    dm_input_set(kbd_bits(), wasd ? (g_d - g_a) * DM_STICK_MAX : g_touch_lx, wasd ? (g_w - g_s) * DM_STICK_MAX : g_touch_ly, 0, 0);
}
// Focus lost, the queue gone, the activity paused: no key-up or stick release follows, so whatever the front end holds is let go (DM_KEYS steps hold on, as on the Mac). The touch gamepad's
// too: its fingers on the game are lifted, and the epoch moves on, which TouchPad reads before each touch event and then leaves every finger already down untracked until it lifts (a finger
// held across a release must not press again on its next move)
static void input_release(void) {
    if (!atomic_load(&g_host_on)) return;
    g_pad_keys = g_pad_hat = g_pad_trig = g_kbd = 0; g_lx = g_ly = g_rx = g_ry = 0; g_w = g_a = g_s = g_d = 0;
    g_touch_btn = 0; g_touch_lx = g_touch_ly = 0; g_touch_epoch++;
    if (g_touch_n) { g_touch_n = 0; dm_touch_set(NULL, 0); }
    pad_push(); kbd_push();
    if (g_input_log) LOG("front end: input released: the controller, the keyboard and the touch controls let go (%.2f s after launch)", (now_ns() - g_t0_ns) / 1e9);
}
// ---- The touch gamepad's natives (HostActivity: nativeTouchEpoch, nativeTouchSet, nativeTouchLog; nativeControllerRemoved, below), registered in onCreate by touch_init ----
// TouchPad, a view in a panel window over the activity's that never takes focus, takes every touch on the activity: it hit-tests them against its controls, draws them, and after each touch
// event hands the whole touch state here (one path: the input queue receives no touches while the panel covers the window). The UI thread calls these, the thread the input queue's callbacks
// run on, so the state needs no lock; they call nothing but lock-free host setters and dm_touch_set's short lock, and never Java. Unchanged state is not pushed again
static jint JNICALL touch_epoch(JNIEnv *env, jclass c) { (void)env; (void)c; return g_touch_epoch; }
// buttons: dm_input.h bits (the 16 NpadButtons); lx, ly: the left stick, up positive; game: n fingers on the game as (finger slot, x, y), normalized on the picture
static void JNICALL touch_set(JNIEnv *env, jclass c, jint buttons, jint lx, jint ly, jfloatArray game, jint n) {
    (void)c;
    if (!atomic_load(&g_host_on)) return;
    uint64_t b = (uint32_t)buttons & 0xffffu;
    lx = lx < -DM_STICK_MAX ? -DM_STICK_MAX : lx > DM_STICK_MAX ? DM_STICK_MAX : lx; ly = ly < -DM_STICK_MAX ? -DM_STICK_MAX : ly > DM_STICK_MAX ? DM_STICK_MAX : ly;
    if (b != g_touch_btn || lx != g_touch_lx || ly != g_touch_ly) {
        // logged: every button change, the stick's grip and release, and its moves past an eighth of its travel (not every move)
        int log = g_input_log && (b != g_touch_btn || (!lx && !ly) != (!g_touch_log_lx && !g_touch_log_ly) || abs(lx - g_touch_log_lx) > 8192 || abs(ly - g_touch_log_ly) > 8192);
        g_touch_btn = b; g_touch_lx = lx; g_touch_ly = ly;
        kbd_push();
        if (log) {   // the two paths the host merges: the keyboard / touch path's buttons (these controls' and the keyboard's) and the controller path's
            g_touch_log_lx = lx; g_touch_log_ly = ly;
            LOG("front end: touch controls %#llx stick %d,%d → keyboard and touch %#llx, controller %#llx (%.2f s after launch)", (unsigned long long)b, lx, ly, (unsigned long long)kbd_bits(),
                (unsigned long long)pad_bits(), (now_ns() - g_t0_ns) / 1e9);
        }
    }
    dm_touch t[16]; float v[48];
    if (n < 0 || !game) n = 0;
    if (n > 16) n = 16;
    if (n && (*env)->GetArrayLength(env, game) < n * 3) n = (*env)->GetArrayLength(env, game) / 3;
    if (n) (*env)->GetFloatArrayRegion(env, game, 0, n * 3, v);
    for (int i = 0; i < n; i++) t[i] = (dm_touch){ (uint32_t)v[i * 3], v[i * 3 + 1], v[i * 3 + 2] };
    if (n != g_touch_n || memcmp(t, g_touch_game, (size_t)n * sizeof *t)) {
        if (g_input_log && n != g_touch_n) {
            double s = (now_ns() - g_t0_ns) / 1e9;
            if (n) LOG("front end: touch screen %d finger%s, the first at %.3f,%.3f (%.2f s after launch)", n, n == 1 ? "" : "s", t[0].x, t[0].y, s);
            else LOG("front end: touch screen 0 fingers (%.2f s after launch)", s);
        }
        g_touch_n = n; memcpy(g_touch_game, t, (size_t)n * sizeof *t);
        dm_touch_set(t, n);
    }
}
static void JNICALL touch_log(JNIEnv *env, jclass c, jbyteArray msg) {   // TouchPad's own lines (its layout, shown or hidden), under DM_INPUT_LOG
    (void)c;
    if (!g_input_log || !msg) return;
    char s[1024]; jsize n = (*env)->GetArrayLength(env, msg); if (n > (jsize)sizeof s - 1) n = (jsize)sizeof s - 1;
    (*env)->GetByteArrayRegion(env, msg, 0, n, (jbyte *)s); s[n] = 0;
    LOG("front end: %s", s);
}
static void JNICALL pad_removed(JNIEnv *env, jclass c, jint id, jboolean last);   // with the controller's input, below
static void touch_init(ANativeActivity *act) {   // onCreate, the UI thread, once the host is set to start: TouchPad shows no controls while these are not bound
    JNIEnv *env = act->env; jclass c = (*env)->GetObjectClass(env, act->clazz);
    JNINativeMethod m[] = { { "nativeTouchEpoch", "()I", (void *)touch_epoch }, { "nativeTouchSet", "(III[FI)V", (void *)touch_set }, { "nativeTouchLog", "([B)V", (void *)touch_log },
                            { "nativeControllerRemoved", "(IZ)V", (void *)pad_removed } };
    int ok = c && !(*env)->RegisterNatives(env, c, m, (jint)(sizeof m / sizeof *m));
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    if (c) (*env)->DeleteLocalRef(env, c);
    if (!ok) fprintf(stderr, "boot: [RED] HostActivity's touch gamepad natives could not be bound: no touch controls are shown\n");
}
static int32_t axis(float v) { return (int32_t)(fmaxf(-1.f, fminf(1.f, v)) * DM_STICK_MAX); }
// A radial dead zone on each stick, a fraction of full travel (DM_STICK_DEADZONE, 0.15 by default; read once env.txt is in), rescaled so travel past its edge starts from 0 and the rim still
// reaches DM_STICK_MAX: without it a stick at rest that reads a few percent off centre walks the player (0.24 would reach the guest as 7,864)
static float g_dz = 0.15f;
static void stick(float x, float y, int32_t *ox, int32_t *oy) {
    float m = sqrtf(x * x + y * y);
    if (!(m > g_dz)) { *ox = *oy = 0; return; }   // NaN too
    float k = (fminf(m, 1.f) - g_dz) / (1.f - g_dz) / m;
    *ox = axis(x * k); *oy = axis(y * k);
}
static int key_event(const AInputEvent *e) {
    int32_t c = AKeyEvent_getKeyCode(e), act = AKeyEvent_getAction(e);
    if (act != AKEY_EVENT_ACTION_DOWN && act != AKEY_EVENT_ACTION_UP) return 0;
    int down = act == AKEY_EVENT_ACTION_DOWN; uint64_t m;
    if ((m = pad_key_bit(c))) { g_pad_keys = down ? g_pad_keys | m : g_pad_keys & ~m; pad_push(); void dm_hotkeys(uint64_t held); dm_hotkeys(pad_bits()); }   // hotkey.c
    else if ((m = kbd_key_bit(c)) || c == AKEYCODE_W || c == AKEYCODE_A || c == AKEYCODE_S || c == AKEYCODE_D) {
        if (c == AKEYCODE_W) g_w = down; else if (c == AKEYCODE_A) g_a = down; else if (c == AKEYCODE_S) g_s = down; else if (c == AKEYCODE_D) g_d = down;
        g_kbd = down ? g_kbd | m : g_kbd & ~m; kbd_push();
    } else return 0;   // the rest (volume, back, home) is the system's: back finishes the activity, which ends the host (onDestroy)
    if (g_input_log && AKeyEvent_getRepeatCount(e) == 0)
        LOG("front end: key %d %s (source %#x, device %d) → controller %#llx, keyboard and touch %#llx (%.2f s after launch)", c, down ? "down" : "up", AInputEvent_getSource(e),
            AInputEvent_getDeviceId(e), (unsigned long long)pad_bits(), (unsigned long long)kbd_bits(), (now_ns() - g_t0_ns) / 1e9);
    return 1;
}
// The controller's sticks, hat and triggers let go (its keys too when keys is set): no further event would, after a cancel or once the controller is gone. Main thread
static void pad_axes_release(int keys, const char *why) {
    g_lx = g_ly = g_rx = g_ry = 0; g_pad_hat = g_pad_trig = 0;
    if (keys) g_pad_keys = 0;
    pad_push();
    if (g_input_log) LOG("front end: %s: the controller's sticks, hat and triggers%s let go → controller %#llx (%.2f s after launch)", why, keys ? " and keys" : "", (unsigned long long)pad_bits(),
                         (now_ns() - g_t0_ns) / 1e9);
}
// HostActivity.nativeControllerRemoved (TouchPad, InputManager): input device id went. The sticks, hat and triggers let go only when it is the device whose motion set them (g_pad_dev): a
// mouse, a keyboard or an idle second controller unplugged must not drop a stick held steady on the controller in use, which sends no further motion. last (the last controller went, so
// the controls come back): its keys too, as iOS zeroes the controller's state when the last one goes (DmGamepad.m padsChanged); not on any other removal, which would drop a keyboard's
// held arrow (pad_key_bit: DPAD keys go on the controller path). The UI thread
static void JNICALL pad_removed(JNIEnv *env, jclass c, jint id, jboolean last) {
    (void)env; (void)c;
    if (!atomic_load(&g_host_on) || (id != g_pad_dev && !last)) return;
    pad_axes_release(last, last ? "the last controller was removed" : "the controller in use was removed");
    g_pad_dev = -1;
}
// A controller's sticks, triggers and hat: Android's left stick is X / Y and its right stick Z / RZ, y pointing down (Switch sticks point up); analog triggers past half travel press ZL / ZR.
// A cancel (the device reset or gone mid-tilt) lets them go when it is the device whose motion set them (g_pad_dev): another controller's cancel leaves a held tilt alone
static int motion_event(const AInputEvent *e) {
    if (!(AInputEvent_getSource(e) & AINPUT_SOURCE_CLASS_JOYSTICK)) return 0;   // no touches reach the queue: TouchPad's window takes them
    int32_t act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK;
    if (act == AMOTION_EVENT_ACTION_CANCEL) { if (AInputEvent_getDeviceId(e) == g_pad_dev) pad_axes_release(0, "controller motion cancelled"); return 1; }
    if (act != AMOTION_EVENT_ACTION_MOVE) return 0;
    g_pad_dev = AInputEvent_getDeviceId(e);
    uint64_t was = pad_bits(); int tilted = (g_lx || g_ly) | (g_rx || g_ry) << 1;
    stick(AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_X, 0), -AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Y, 0), &g_lx, &g_ly);
    stick(AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Z, 0), -AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RZ, 0), &g_rx, &g_ry);
    float lt = fmaxf(AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_LTRIGGER, 0), AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_BRAKE, 0));
    float rt = fmaxf(AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RTRIGGER, 0), AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_GAS, 0));
    float hx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0), hy = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0);
    g_pad_trig = (lt > 0.5f ? DM_BTN_ZL : 0) | (rt > 0.5f ? DM_BTN_ZR : 0);
    g_pad_hat = (hx < -0.5f ? DM_BTN_LEFT : 0) | (hx > 0.5f ? DM_BTN_RIGHT : 0) | (hy < -0.5f ? DM_BTN_UP : 0) | (hy > 0.5f ? DM_BTN_DOWN : 0);
    pad_push();
    if (g_input_log && (pad_bits() != was || ((g_lx || g_ly) | (g_rx || g_ry) << 1) != tilted))   // a hat or trigger change, a stick leaving or reaching its dead zone (not every move)
        LOG("front end: controller motion (device %d): sticks %d,%d %d,%d → controller %#llx (%.2f s after launch)", AInputEvent_getDeviceId(e), g_lx, g_ly, g_rx, g_ry,
            (unsigned long long)pad_bits(), (now_ns() - g_t0_ns) / 1e9);
    return 1;
}
static AInputQueue *g_queue;
static int on_input(int fd, int events, void *data) {
    (void)fd; (void)events; (void)data;
    AInputEvent *e;
    while (g_queue && AInputQueue_getEvent(g_queue, &e) >= 0) {
        if (AInputQueue_preDispatchEvent(g_queue, e)) continue;   // taken by the IME first
        int handled = 0;
        if (atomic_load(&g_host_on)) handled = AInputEvent_getType(e) == AINPUT_EVENT_TYPE_KEY ? key_event(e) : AInputEvent_getType(e) == AINPUT_EVENT_TYPE_MOTION ? motion_event(e) : 0;
        AInputQueue_finishEvent(g_queue, e, handled);
    }
    return 1;
}
static void on_input_created(ANativeActivity *a, AInputQueue *q) { (void)a; g_queue = q; AInputQueue_attachLooper(q, ALooper_forThread(), 1, on_input, NULL); }
static void on_input_destroyed(ANativeActivity *a, AInputQueue *q) { (void)a; AInputQueue_detachLooper(q); g_queue = NULL; input_release(); }

// ---------- Window: recorded for the GPU backend (front_native_window); the null backend's build paints it black, as nothing else draws to it ----------
// The Vulkan backend (dm/gfx/vkn/vk_present.c) defines both hooks: its swapchain owns the window (a CPU lock would connect the window as a CPU producer, and the swapchain could not connect), and on
// window loss it releases the surface before this callback returns, as Android requires (the window may go once onNativeWindowDestroyed returns). Weak: the null build links neither
__attribute__((weak)) int dm_gfx_owns_window(void) { return 0; }
__attribute__((weak)) void dm_gfx_window_lost(void) {}
static void win_black(ANativeWindow *w) {
    if (dm_gfx_owns_window()) return;
    ANativeWindow_Buffer b;
    if (ANativeWindow_setBuffersGeometry(w, 0, 0, WINDOW_FORMAT_RGBX_8888) || ANativeWindow_lock(w, &b, NULL)) return;
    for (int y = 0; y < b.height; y++) memset((uint8_t *)b.bits + (size_t)y * b.stride * 4, 0, (size_t)b.width * 4);
    ANativeWindow_unlockAndPost(w);
}
static void on_win_created(ANativeActivity *a, ANativeWindow *w) {
    (void)a; atomic_store(&g_win, w);
    LOG("front end: window %dx%d", ANativeWindow_getWidth(w), ANativeWindow_getHeight(w));   // one wording for both backends (parity_expected.txt); the Vulkan backend logs its surface
    win_black(w);
}
static void on_win_redraw(ANativeActivity *a, ANativeWindow *w) { (void)a; win_black(w); }   // the swapchain's next present redraws it (FIFO, every frame) with the Vulkan backend
static void on_win_destroyed(ANativeActivity *a, ANativeWindow *w) { (void)a; (void)w; atomic_store(&g_win, NULL); dm_gfx_window_lost(); }   // NULL first: no new surface is made on it
static void on_focus(ANativeActivity *a, int has) {
    (void)a;
    if (g_input_log && atomic_load(&g_host_on)) LOG("front end: window focus %s (%.2f s after launch)", has ? "gained" : "lost", (now_ns() - g_t0_ns) / 1e9);
    if (!has) input_release();
}

// ---------- Rumble: HapticsRouter (dm/android/app/java/dev/nxnative/haptics, docs/android/rumble.md), made by HostActivity.startHaptics ----------
// Started in onCreate before the host thread, so the game's first nn::hid vibration query already sees the output. HapticsRouter's natives are bound here (dm_haptics_android_register: ART
// does not search a NativeActivity library for exported JNI names), and so is HostActivity.nativeHapticsLog, its lines in log.txt. DM_RUMBLE=0: no router, no output advertised (as before)
static void JNICALL haptics_log(JNIEnv *env, jclass c, jbyteArray msg) {
    (void)c;
    if (!msg) return;
    char s[2048]; jsize n = (*env)->GetArrayLength(env, msg); if (n > (jsize)sizeof s - 1) n = (jsize)sizeof s - 1;
    (*env)->GetByteArrayRegion(env, msg, 0, n, (jbyte *)s); s[n] = 0;
    for (char *l = s, *e; l && *l; l = e) { e = strchr(l, '\n'); if (e) *e++ = 0; fprintf(stderr, "rumble: %s\n", l); }
}
static void haptics_init(ANativeActivity *act) {   // onCreate, the UI thread, before the host thread
    const char *e = getenv("DM_RUMBLE");
    if (e && !atoi(e)) { fprintf(stderr, "boot: rumble off (DM_RUMBLE=%s): no vibration output is advertised to the game\n", e); return; }
    JNIEnv *env = act->env; jclass c = (*env)->GetObjectClass(env, act->clazz);
    jmethodID cls = c ? (*env)->GetStaticMethodID(env, c, "hapticsRouterClass", "()Ljava/lang/Class;") : NULL;
    jmethodID start = c ? (*env)->GetMethodID(env, c, "startHaptics", "()Z") : NULL;
    jclass rc = cls && start && !(*env)->ExceptionCheck(env) ? (jclass)(*env)->CallStaticObjectMethod(env, c, cls) : NULL;
    JNINativeMethod m = { "nativeHapticsLog", "([B)V", (void *)haptics_log };
    int ok = rc && !(*env)->ExceptionCheck(env) && dm_haptics_android_register(env, rc) && !(*env)->RegisterNatives(env, c, &m, 1);
    if ((*env)->ExceptionCheck(env)) { (*env)->ExceptionClear(env); ok = 0; }
    if (ok) { ok = (*env)->CallBooleanMethod(env, act->clazz, start); if ((*env)->ExceptionCheck(env)) { (*env)->ExceptionClear(env); ok = 0; } }
    if (rc) (*env)->DeleteLocalRef(env, rc);
    if (c) (*env)->DeleteLocalRef(env, c);
    if (!ok) fprintf(stderr, "boot: [RED] rumble did not start (HapticsRouter or its natives could not be bound, or it failed; log.txt's rumble: lines say why): the game sees no vibration output\n");
}

// ---------- Lifecycle (iOS front_ios_focus_event): paused = focus lost and audio suspended; stopped = in the background, the guest clock frozen and graphics suspended ----------
// The first onStart / onResume of a launch only begins it: the host starts in focus (hle_oe.c), and pushing 2 then 1 would send the game focus messages while it boots. Main thread only
static int g_paused, g_stopped;
static void on_pause(ANativeActivity *a) { (void)a; if (!atomic_load(&g_host_on) || g_paused) return; g_paused = 1; input_release(); dm_focus_set(2); dm_audio_suspend(1); }
static void on_stop(ANativeActivity *a) { (void)a; if (!atomic_load(&g_host_on) || g_stopped) return; g_stopped = 1; dm_focus_set(3); guest_clock_freeze_for(1, 1); dm_gfx_suspend(1); }
static void on_start(ANativeActivity *a) { (void)a; if (!g_stopped) return; g_stopped = 0; dm_gfx_suspend(0); guest_clock_freeze_for(1, 0); dm_focus_set(2); }
static void on_resume(ANativeActivity *a) { (void)a; if (!g_paused) return; g_paused = 0; dm_audio_suspend(0); dm_focus_set(1); }
// Destroyed (back, finish, the system): the host cannot start again in this process, so it ends it, the way SIGTERM does (hle_lifecycle.c: audio stopped, graphics drained, then _exit)
static void on_destroy(ANativeActivity *a) {
    (void)a;
    if (atomic_load(&g_host_on)) dm_host_shutdown("activity destroyed");
    fprintf(stderr, "boot: activity destroyed\n"); _exit(0);
}
static void *on_save(ANativeActivity *a, size_t *n) { (void)a; *n = 0; return NULL; }
static void on_low_memory(ANativeActivity *a) { (void)a; if (atomic_load(&g_host_on)) LOG("front end: the system reports low memory"); }

void ANativeActivity_onCreate(ANativeActivity *act, void *saved, size_t saved_size) {
    (void)saved; (void)saved_size;
    g_act = act;
    ANativeActivityCallbacks *cb = act->callbacks;
    cb->onStart = on_start; cb->onResume = on_resume; cb->onPause = on_pause; cb->onStop = on_stop; cb->onDestroy = on_destroy; cb->onSaveInstanceState = on_save;
    cb->onWindowFocusChanged = on_focus; cb->onNativeWindowCreated = on_win_created; cb->onNativeWindowRedrawNeeded = on_win_redraw; cb->onNativeWindowDestroyed = on_win_destroyed;
    cb->onInputQueueCreated = on_input_created; cb->onInputQueueDestroyed = on_input_destroyed; cb->onLowMemory = on_low_memory;
    ANativeActivity_setWindowFlags(act, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);   // a player on a controller touches nothing, so nothing else would keep the screen on
    static int once; if (once++) return;   // a second activity in this process (singleTask makes it rare): the host is already running
    snprintf(g_files, sizeof g_files, "%s", act->internalDataPath ? act->internalDataPath : "");
    snprintf(g_ext, sizeof g_ext, "%s", act->externalDataPath ? act->externalDataPath : "");
    if (!*g_files) { __android_log_write(ANDROID_LOG_ERROR, "dmhost", "boot: [RED] no internal data path: the host does not start"); log_nowhere(); return; }
    if (!boot_env()) return;
    text_box_init(act);
    touch_init(act);
    haptics_init(act);
    g_input_log = getenv("DM_INPUT_LOG") && atoi(getenv("DM_INPUT_LOG"));   // log each key press with what reached the controller (the done-test's evidence)
    { const char *e = getenv("DM_STICK_DEADZONE"); char *end; float v = e ? strtof(e, &end) : 0;
      if (e && end != e && v >= 0.f && v <= 0.9f) g_dz = v; else if (e) fprintf(stderr, "boot: DM_STICK_DEADZONE=%s is not a fraction from 0 to 0.9: the dead zone stays %.2f\n", e, g_dz); }
    pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 64u << 20);   // as the iOS host thread: the Switch main thread's stack is set by npdm, the host gives plenty
    pthread_t t;
    if (pthread_create(&t, &at, host_thread, NULL)) { fprintf(stderr, "boot: the host thread cannot start (errno %d)\n", errno); pthread_attr_destroy(&at); return; }
    pthread_detach(t); pthread_attr_destroy(&at);
    atomic_store(&g_host_on, 1);
}
