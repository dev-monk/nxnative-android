// Vulkan backend: the library, the driver, the instance, the device and the validation messenger (vkn.h). Runs once, at the backend's first use (mtl_init).
// DM_VK_DRIVER=system | 842 | turnip (default system) picks the driver. A package driver is the user's own file (842.6 is Qualcomm's and cannot be redistributed), opened through
// libadrenotools as the P0.7 / P0.8 probes do (dm/android/probe/gpu/vk_common.c): from DM_VK_DRIVER_DIR, by default <DM_ROOT>/drivers/<name>/ (the app's files directory, where
// run_app.sh --driver unpacks it); the hook libraries beside this library (the APK's lib directory; DM_VK_HOOK_DIR for the adb route). libadrenotools falls back to the system driver
// without a word, so the driver that answered is checked (P0.7 condition 7: the package's library is mapped and the driverID is the package's) and a fallback is said in red.
// DM_VK_VALIDATION=1: VK_LAYER_KHRONOS_validation (the Mac: VK_LAYER_PATH to Homebrew's; the phone: the layer packaged in the APK's lib directory, which the loader searches for the
// app's own layers) with a VK_EXT_debug_utils messenger that puts every warning and error in the log in red
#include "vkn.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

vkn_fn vk;
vkn_dev_t vd;
_Atomic int vkn_lost;
void vkn_gpu_lost(const char *where, VkResult r) {
    if (!atomic_exchange(&vkn_lost, 1)) VLOG("[RED] the GPU is lost to this run (%s: %d%s): from here submissions complete without it and nothing is shown", where, r, r == VK_ERROR_DEVICE_LOST ? ", VK_ERROR_DEVICE_LOST" : "");
}
static _Atomic uint64_t g_val_n;
uint64_t vkn_validation_count(void) { return atomic_load(&g_val_n); }

static VKAPI_ATTR VkBool32 VKAPI_CALL val_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *d, void *user) {
    (void)type; (void)user;
    if (!(sev & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))) return VK_FALSE;
    uint64_t n = atomic_fetch_add(&g_val_n, 1) + 1;
    if (n > 200) { if (n == 201) VLOG("[RED] validation: more than 200 messages, the rest are only counted"); return VK_FALSE; }   // a message per draw would bury the log
    char m[4096]; snprintf(m, sizeof m, "%s", d && d->pMessage ? d->pMessage : "");
    for (char *q = m; *q; q++) if (*q == '\n' || *q == '\r' || *q == '\t') *q = ' ';   // one line each: parity.py and grep count lines
    fprintf(stderr, "[dm] [RED] [vkn] validation: %s %s: %s\n", sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning", d && d->pMessageIdName ? d->pMessageIdName : "-", m);
    return VK_FALSE;
}

// ---------- The library: the system loader, or libadrenotools in front of a package driver ----------
static const char *g_req = "system";     // DM_VK_DRIVER as understood
static const char *g_pkg_lib;            // the package's library name, NULL for the system driver
static char g_how[1400];                 // how the library was opened (the identity line says it)
static char g_pkg_tail[400];             // "<driver directory's name>/<library>": the end of the package library's mapped path (lib_mapped)
#ifdef __ANDROID__
static void *open_package(const char *name, const char *lib) {
    char hook[600], ddir[700], path[800];
    const char *h = getenv("DM_VK_HOOK_DIR");
    if (h && *h) snprintf(hook, sizeof hook, "%s", h);
    else { Dl_info di; if (!dladdr((void *)&open_package, &di) || !di.dli_fname) { VLOG("[RED] driver %s: this library's directory is unknown (dladdr), the hook libraries cannot be found: the system driver is used", name); return NULL; }
           snprintf(hook, sizeof hook, "%s", di.dli_fname); char *s = strrchr(hook, '/'); if (s) *s = 0; }
    const char *dd = getenv("DM_VK_DRIVER_DIR"), *root = getenv("DM_ROOT");
    if (dd && *dd) snprintf(ddir, sizeof ddir, "%s", dd); else snprintf(ddir, sizeof ddir, "%s/drivers/%s", root && *root ? root : ".", name);
    size_t dl = strlen(ddir); if (dl && ddir[dl - 1] != '/' && dl + 1 < sizeof ddir) { ddir[dl] = '/'; ddir[dl + 1] = 0; }   // libadrenotools joins the directory and the name as they are
    snprintf(path, sizeof path, "%s%s", ddir, lib);
    if (access(path, R_OK)) { VLOG("[RED] driver %s: no %s (the package unpacked there: run_app.sh --driver %s, or DM_VK_DRIVER_DIR): the system driver is used", name, path, name); return NULL; }
    snprintf(path, sizeof path, "%s/libadrenotools.so", hook);
    void *at = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!at) { VLOG("[RED] driver %s: dlopen %s: %s: the system driver is used", name, path, dlerror()); return NULL; }
    typedef void *(*open_fn)(int, int, const char *, const char *, const char *, const char *, const char *, void **);
    open_fn fn = (open_fn)dlsym(at, "adrenotools_open_libvulkan");
    if (!fn) { VLOG("[RED] driver %s: no adrenotools_open_libvulkan in %s: the system driver is used", name, path); return NULL; }
    snprintf(g_how, sizeof g_how, "adrenotools_open_libvulkan(RTLD_NOW, ADRENOTOOLS_DRIVER_CUSTOM, NULL, \"%s\", \"%s\", \"%s\", NULL, NULL)", hook, ddir, lib);
    void *v = fn(RTLD_NOW, 1 /* ADRENOTOOLS_DRIVER_CUSTOM */, NULL, hook, ddir, lib, NULL, NULL);
    const char *de = v ? NULL : dlerror();   // once: a second call returns NULL (the error is cleared)
    if (!v) VLOG("[RED] driver %s: %s returned NULL (dlerror: %s): the system driver is used", name, g_how, de ? de : "-");
    else { size_t dn = strlen(ddir); if (dn && ddir[dn - 1] == '/') ddir[--dn] = 0; const char *b = strrchr(ddir, '/'); snprintf(g_pkg_tail, sizeof g_pkg_tail, "%s/%s", b ? b + 1 : ddir, lib); }
    return v;
}
#endif
static int g_pkg_open;   // libadrenotools returned a library for the package (it may still be the system driver's: the identity check)
static int g_lib_there;  // a loader library that failed to open was there (a file, or DM_VK_LOADER named one): a broken Vulkan, not a missing one (vd.absent)
#ifdef __APPLE__
// dyld's error for a bare name lists each path it tried as "'<path>' (<why>)": 1 when there is such a list and every entry is "no such file". A library there but broken (another
// architecture) has another reason, and one whose dependency is missing says "Library not loaded" (its own path was found)
static int dyld_none_there(const char *de) {
    if (!de || strstr(de, "Library not loaded")) return 0;
    int n = 0;
    for (const char *p = de; (p = strstr(p, "' (")) != NULL; p += 3, n++) if (strncmp(p + 3, "no such file", 12)) return 0;
    return n > 0;
}
#endif
static void *open_lib(void) {
    const char *e = getenv("DM_VK_DRIVER");
    if (e && *e && strcmp(e, "system")) {
        if (!strcmp(e, "842")) { g_req = "842"; g_pkg_lib = getenv("DM_VK_DRIVER_LIB") && *getenv("DM_VK_DRIVER_LIB") ? getenv("DM_VK_DRIVER_LIB") : "vulkan.ad07xx.so"; }
        else if (!strcmp(e, "turnip")) { g_req = "turnip"; g_pkg_lib = getenv("DM_VK_DRIVER_LIB") && *getenv("DM_VK_DRIVER_LIB") ? getenv("DM_VK_DRIVER_LIB") : "vulkan.ad08xx.so"; }
        else VLOG("[RED] DM_VK_DRIVER=%s: not system, 842 or turnip: the system driver is used", e);
    }
    void *l = NULL;
#ifdef __ANDROID__
    if (g_pkg_lib) g_pkg_open = (l = open_package(g_req, g_pkg_lib)) != NULL;   // not opened: said in red there; the identity line below still shows what answered
    if (!l) { l = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL); snprintf(g_how, sizeof g_how, "dlopen(\"libvulkan.so\")"); }
    if (!l) g_lib_there = !access("/system/lib64/libvulkan.so", F_OK);
#else
    if (g_pkg_lib) { VLOG("[RED] DM_VK_DRIVER=%s: driver packages are for the phone; the system loader is used", g_req); g_pkg_lib = NULL; }
    const char *cand[3] = {getenv("DM_VK_LOADER"), "/opt/homebrew/lib/libvulkan.1.dylib", "libvulkan.1.dylib"};   // Homebrew's loader (VK_DRIVER_FILES picks KosmicKrisp)
    for (int i = 0; i < 3 && !l; i++) if (cand[i] && *cand[i] && (l = dlopen(cand[i], RTLD_NOW | RTLD_LOCAL))) snprintf(g_how, sizeof g_how, "dlopen(\"%s\")", cand[i]);
    if (!l) g_lib_there = (cand[0] && *cand[0]) || !access(cand[1], F_OK);   // the bare name: dyld's search, told apart below by its error
#endif
    const char *de = l ? NULL : dlerror();
#ifdef __APPLE__
    if (!l && !g_lib_there) g_lib_there = !dyld_none_there(de);
#endif
    if (!l) VLOG("[RED] no Vulkan library could be opened (%s): nothing is drawn", de ? de : "-");
    return l;
}

// Whether a shared object whose path ends in /s is mapped (Linux: /proc/self/maps; s = "<driver directory's name>/<library>": the package library keeps its name and directory, and the
// system driver's vulkan.adreno.so in /vendor cannot match it)
// near: when none does, the first mapped path that holds the bare name (lib), or "none": the fallback line says it (a copy, a memfd or a "(deleted)" mapping would show there)
static int lib_mapped(const char *s, const char *lib, char *near, size_t nn) {
    snprintf(near, nn, "none");
#ifdef __linux__
    FILE *f = fopen("/proc/self/maps", "r"); if (!f || !*s) { if (f) fclose(f); return 0; }
    char line[1024]; int hit = 0, said = 0; size_t n = strlen(s);
    while (!hit && fgets(line, sizeof line, f)) {
        size_t l = strcspn(line, "\n"); line[l] = 0; hit = l > n && line[l - n - 1] == '/' && !strcmp(line + l - n, s);
        if (!hit && !said && strstr(line, lib)) { const char *q = strchr(line, '/'); snprintf(near, nn, "%s", q ? q : line); said = 1; }
    }
    fclose(f); return hit;
#else
    (void)s; (void)lib; return 0;
#endif
}
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }

int vkn_mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < vd.mem.memoryTypeCount; i++) if ((bits >> i & 1) && (vd.mem.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}

int vkn_dev_init(void) {
    void *lib = open_lib(); if (!lib) { vd.absent = !g_lib_there; return 0; }
    vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    if (!vk.vkGetInstanceProcAddr) { VLOG("[RED] %s has no vkGetInstanceProcAddr: nothing is drawn", g_how); return 0; }
#define VKN_G(n) vk.n = (PFN_##n)vk.vkGetInstanceProcAddr(NULL, #n);
    VKN_GLOBAL(VKN_G)
#undef VKN_G
    if (!vk.vkCreateInstance || !vk.vkEnumerateInstanceVersion) { VLOG("[RED] %s: no Vulkan 1.1 loader entry points: nothing is drawn", g_how); return 0; }
    uint32_t iv = 0; vk.vkEnumerateInstanceVersion(&iv);
    if (iv < VK_API_VERSION_1_3) { VLOG("[RED] the Vulkan loader is %u.%u, the backend needs 1.3: nothing is drawn", VK_API_VERSION_MAJOR(iv), VK_API_VERSION_MINOR(iv)); return 0; }

    // Instance: the surface extensions this platform has, debug utilities and the layer for validation, portability enumeration where the loader offers it (a portability driver is listed only then)
    const char *e = getenv("DM_VK_VALIDATION"); vd.validation = e && atoi(e) > 0;
    const char *lay[1]; uint32_t nl = 0;
    if (vd.validation) {
        uint32_t n = 0; vk.vkEnumerateInstanceLayerProperties(&n, NULL);
        VkLayerProperties *lp = calloc(n ? n : 1, sizeof *lp); vk.vkEnumerateInstanceLayerProperties(&n, lp);
        for (uint32_t i = 0; i < n; i++) if (!strcmp(lp[i].layerName, "VK_LAYER_KHRONOS_validation")) { lay[nl++] = "VK_LAYER_KHRONOS_validation"; break; }
        free(lp);
        if (!nl) { VLOG("[RED] DM_VK_VALIDATION=1, but VK_LAYER_KHRONOS_validation is not among the %u instance layers (the Mac: VK_LAYER_PATH; the phone: the layer in the APK, build_app.sh VALIDATION_LAYER): validation is off", n); vd.validation = 0; }
    }
    uint32_t nie = 0; vk.vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
    VkExtensionProperties *ie = calloc(nie + 64, sizeof *ie); vk.vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
    if (nl) { uint32_t m = 64; vk.vkEnumerateInstanceExtensionProperties(lay[0], &m, ie + nie); nie += m; }   // VK_EXT_debug_utils comes from the layer on Android
    const char *ext[8]; uint32_t ne = 0; VkInstanceCreateFlags flags = 0;
#ifdef __ANDROID__
    const char *sx = "VK_KHR_android_surface";
#else
    const char *sx = "VK_EXT_metal_surface";
#endif
    if (has_ext(ie, nie, "VK_KHR_surface") && has_ext(ie, nie, sx)) { ext[ne++] = "VK_KHR_surface"; ext[ne++] = sx; vd.surface_ext = sx; }
    else VLOG("[RED] the instance has no %s: nothing is shown", sx);
    if (vd.validation) { if (has_ext(ie, nie, "VK_EXT_debug_utils")) ext[ne++] = "VK_EXT_debug_utils"; else { VLOG("[RED] DM_VK_VALIDATION=1, but no VK_EXT_debug_utils: validation messages are not seen"); vd.validation = 0; nl = 0; } }
    if (has_ext(ie, nie, "VK_KHR_portability_enumeration")) { ext[ne++] = "VK_KHR_portability_enumeration"; flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR; }
    free(ie);
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nxnative", 1, "nxnative-vkn", 1, VK_API_VERSION_1_3};
    VkDebugUtilsMessengerCreateInfoEXT mci = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0,
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, val_cb, NULL};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, vd.validation ? &mci : NULL, flags, &ai, nl, lay, ne, ext};   // chained: instance creation's own messages too
#ifdef VK_EXT_layer_settings
    // DM_VK_VALIDATION=2: synchronization validation too (the layer's validate_sync setting, given through VK_EXT_layer_settings: on Android no environment reaches the layer)
    static const VkBool32 on32 = VK_TRUE;
    static const VkLayerSettingEXT ls = {"VK_LAYER_KHRONOS_validation", "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on32};
    VkLayerSettingsCreateInfoEXT lsi = {VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT, NULL, 1, &ls};
    if (vd.validation && atoi(e) >= 2) { mci.pNext = &lsi; VLOG("DM_VK_VALIDATION=2: synchronization validation on"); }
#endif
    VkResult r = vk.vkCreateInstance(&ci, NULL, &vd.inst);
    if (r) { VLOG("[RED] vkCreateInstance: %d%s: nothing is drawn", r, r == VK_ERROR_INCOMPATIBLE_DRIVER ? " (VK_ERROR_INCOMPATIBLE_DRIVER: the loader found no driver)" : ""); vd.inst = NULL; vd.absent = r == VK_ERROR_INCOMPATIBLE_DRIVER; return 0; }
#define VKN_I(n) vk.n = (PFN_##n)vk.vkGetInstanceProcAddr(vd.inst, #n);
    VKN_INSTANCE(VKN_I)
#undef VKN_I
    if (vd.surface_ext) vk.create_surface = (vkn_create_surface_fn)(void *)vk.vkGetInstanceProcAddr(vd.inst,
#ifdef __ANDROID__
        "vkCreateAndroidSurfaceKHR");
#else
        "vkCreateMetalSurfaceEXT");
#endif
    if (vd.validation) { VkDebugUtilsMessengerEXT m; if (!vk.vkCreateDebugUtilsMessengerEXT || vk.vkCreateDebugUtilsMessengerEXT(vd.inst, &mci, NULL, &m)) { VLOG("[RED] vkCreateDebugUtilsMessengerEXT failed: validation messages are not seen"); vd.validation = 0; } }

    // Physical device: the first Vulkan 1.3 one with a graphics queue, a GPU before a CPU one (lavapipe beside KosmicKrisp when VK_DRIVER_FILES lists both)
    uint32_t np = 0; VkPhysicalDevice pds[8];
    if ((r = vk.vkEnumeratePhysicalDevices(vd.inst, &np, NULL)) < 0) { VLOG("[RED] vkEnumeratePhysicalDevices: %d: nothing is drawn", r); return 0; }   // an error is not an empty list (vd.absent stays 0)
    if (np > 8) np = 8;
    if ((r = vk.vkEnumeratePhysicalDevices(vd.inst, &np, pds)) < 0) { VLOG("[RED] vkEnumeratePhysicalDevices: %d: nothing is drawn", r); return 0; }   // VK_INCOMPLETE (more than 8): the first 8
    int best = -1, best_cpu = 1; uint32_t best_q = 0;
    for (uint32_t i = 0; i < np; i++) {
        VkPhysicalDeviceProperties p; vk.vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        uint32_t nq = 0; vk.vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &nq, NULL);
        VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16; vk.vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &nq, qf);
        int q = -1; for (uint32_t k = 0; k < nq && q < 0; k++) if (qf[k].queueFlags & VK_QUEUE_GRAPHICS_BIT) q = (int)k;
        int cpu = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        if (q >= 0 && (best < 0 || (best_cpu && !cpu))) { best = (int)i; best_cpu = cpu; best_q = (uint32_t)q; }
    }
    if (best < 0) { VLOG("[RED] none of the %u physical devices has Vulkan 1.3 and a graphics queue: nothing is drawn", np); vd.absent = np == 0; return 0; }   // none at all, the enumeration having succeeded: no driver
    vd.pd = pds[best]; vd.qfam = best_q;
    vk.vkGetPhysicalDeviceProperties(vd.pd, &vd.props);
    vd.drv = (VkPhysicalDeviceDriverProperties){VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &vd.drv};
    vk.vkGetPhysicalDeviceProperties2(vd.pd, &p2);
    vk.vkGetPhysicalDeviceMemoryProperties(vd.pd, &vd.mem);
    for (char *q = vd.drv.driverInfo; *q; q++) if (*q == '\n' || *q == '\r') *q = ';';

    // Identity (P0.7 condition 7): a package counts as loaded only when its library is mapped and the driver says it is that driver; anything else is the silent fallback
    const char *act = "system"; char why[600] = "", near[256];
    if (g_pkg_lib) {
        int mapped = lib_mapped(g_pkg_tail, g_pkg_lib, near, sizeof near), want_id = !strcmp(g_req, "turnip") ? VK_DRIVER_ID_MESA_TURNIP : VK_DRIVER_ID_QUALCOMM_PROPRIETARY;
        const char *xi = getenv("DM_VK_EXPECT_INFO");   // e.g. E031.50 for 842.6 (its driverID is the stock driver's, 8)
        int info_ok = !xi || !*xi || strstr(vd.drv.driverInfo, xi) || strstr(vd.drv.driverName, xi);
        if (mapped && (int)vd.drv.driverID == want_id && info_ok) act = g_req;
        else { snprintf(why, sizeof why, "%s%s%s%s%s%s%s%s%s", mapped ? "" : g_pkg_tail, mapped ? "" : " not mapped (a path holding ", mapped ? "" : g_pkg_lib, mapped ? "" : ": ", mapped ? "" : near, mapped ? "" : "); ", (int)vd.drv.driverID == want_id ? "" : "driverID not the package's; ",
                        info_ok ? "" : "driverInfo lacks ", info_ok ? "" : xi);
               size_t n = strlen(why); if (n >= 2 && why[n - 2] == ';') why[n - 2] = 0; }
    }
    VLOG("driver: requested %s, actual %s: %s, driverID %d \"%s\" / \"%s\", api %u.%u.%u, driverVersion %#x; %s; validation %s", g_req, act, vd.props.deviceName, (int)vd.drv.driverID,
         vd.drv.driverName, vd.drv.driverInfo, VK_API_VERSION_MAJOR(vd.props.apiVersion), VK_API_VERSION_MINOR(vd.props.apiVersion), VK_API_VERSION_PATCH(vd.props.apiVersion),
         vd.props.driverVersion, g_how, vd.validation ? "on" : "off");
    if (g_pkg_open && strcmp(act, g_req)) VLOG("[RED] driver: requested %s, but the system driver answered (%s): libadrenotools fell back to it silently", g_req, why);

    // Device: Vulkan 1.3's dynamic rendering and synchronization2, 1.2's timeline semaphores; the swapchain when there is a surface; portability subset where the driver has it (required then)
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    vk.vkGetPhysicalDeviceFeatures2(vd.pd, &f2);
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore) { VLOG("[RED] the device lacks dynamic rendering, synchronization2 or timeline semaphores: nothing is drawn"); return 0; }
    VkPhysicalDeviceVulkan11Features w11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, NULL};
    VkPhysicalDeviceVulkan12Features w12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &w11}; w12.timelineSemaphore = VK_TRUE;
    // Buffer device addresses for the shadow buffers (PLAN.md 2.2 B: the address table gives one per pool); required by 1.3, so a driver without it is said and the shadows have none
    if (f12.bufferDeviceAddress) { w12.bufferDeviceAddress = VK_TRUE; vd.bda = 1; } else VLOG("[RED] the device has no bufferDeviceAddress (Vulkan 1.3 requires it): shadow buffers have no device address");
    VkPhysicalDeviceVulkan13Features w13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &w12}; w13.dynamicRendering = VK_TRUE; w13.synchronization2 = VK_TRUE;
    // Samplers (nvn_vk.c samp_make, PLAN.md 2.3): anisotropic filtering, the mirror-clamp-to-edge wrap and min / max reduction where the device has them; a sampler that needs one it lacks is said there
    VkPhysicalDeviceFeatures2 w2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &w13};
    if (f2.features.samplerAnisotropy) { w2.features.samplerAnisotropy = VK_TRUE; vd.aniso = 1; }
    if (f12.samplerMirrorClampToEdge) { w12.samplerMirrorClampToEdge = VK_TRUE; vd.mirror = 1; }
    if (f12.samplerFilterMinmax) { w12.samplerFilterMinmax = VK_TRUE; vd.minmax = 1; }
    // Draws (nvn_vk.c, PLAN.md 2.5 "As built, slice 2c"): depth clamp, independent blend, dual-source blend, depth-bias clamp and non-solid fill where the device has them (a draw that needs a missing one is said there)
    if (f2.features.depthClamp) { w2.features.depthClamp = VK_TRUE; vd.depth_clamp = 1; }
    if (f2.features.independentBlend) { w2.features.independentBlend = VK_TRUE; vd.indep_blend = 1; }
    if (f2.features.dualSrcBlend) { w2.features.dualSrcBlend = VK_TRUE; vd.dual_src = 1; }
    if (f2.features.depthBiasClamp) { w2.features.depthBiasClamp = VK_TRUE; vd.bias_clamp = 1; }
    if (f2.features.fillModeNonSolid) { w2.features.fillModeNonSolid = VK_TRUE; vd.fill_ns = 1; }
    // The shader capabilities slice 0's modules declare beyond core (ImageGatherExtended, DrawParameters), and InterpolationFunction, where the device has them: vk_shader.c refuses a module whose
    // feature is off
    if (f2.features.shaderImageGatherExtended) { w2.features.shaderImageGatherExtended = VK_TRUE; vd.gather_ext = 1; }
    if (f2.features.sampleRateShading) { w2.features.sampleRateShading = VK_TRUE; vd.sample_rate = 1; }
    if (f11.shaderDrawParameters) { w11.shaderDrawParameters = VK_TRUE; vd.draw_params = 1; }
    // Out-of-bounds buffer accesses return zero instead of faulting the GPU (Mali loses the device on them; the translated shaders compute their own offsets into storage buffers):
    // robustBufferAccess where the device has it, DM_VK_ROBUST=0 leaves it off
    { const char *e = getenv("DM_VK_ROBUST"); if (f2.features.robustBufferAccess && !(e && *e == '0')) w2.features.robustBufferAccess = VK_TRUE;
      if (f13.robustImageAccess && !(e && *e == '0')) w13.robustImageAccess = VK_TRUE;   // image accesses outside the image too (compute stores)
      VLOG("robust buffer access: %s, image access %s", w2.features.robustBufferAccess ? "on" : f2.features.robustBufferAccess ? "off (DM_VK_ROBUST=0)" : "off (the device has none)", w13.robustImageAccess ? "on" : "off"); }
    // Subgroup operations (core 1.1, no feature to enable): what the device runs where. Paper Mario: The Thousand-Year Door's in-game lighting compute shaders elect a leader
    // among the active lanes with subgroupBallot and broadcast with subgroupShuffle, which works at any subgroup size (Mali's 16 as Maxwell's 32); refused, the scenes stayed dark
    { VkPhysicalDeviceSubgroupProperties sp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES}; VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sp};
      vk.vkGetPhysicalDeviceProperties2(vd.pd, &p2); const char *e = getenv("DM_VK_SUBGROUP");
      if (!(e && *e == '0')) { vd.sg_ops = sp.supportedOperations; vd.sg_stages = sp.supportedStages; vd.sg_size = sp.subgroupSize; }
      VLOG("subgroups: size %u, operations %#x in stages %#x%s", sp.subgroupSize, sp.supportedOperations, sp.supportedStages, e && *e == '0' ? " (DM_VK_SUBGROUP=0: not used)" : ""); }
    { const char *e = getenv("DM_VK_ASTC"); if (f2.features.textureCompressionASTC_LDR && !(e && *e == '0')) w2.features.textureCompressionASTC_LDR = VK_TRUE;   // ASTC textures (nvn_fmt_vk.c); DM_VK_ASTC=0: none
      VLOG("ASTC textures: %s", w2.features.textureCompressionASTC_LDR ? "sampled natively" : "not available (no host images)"); }
    // BC textures sampled natively where the device has them (vk_bc.c decodes them on the CPU otherwise); DM_VK_BC_NATIVE=0 decodes them anyway
    { const char *e = getenv("DM_VK_BC_NATIVE"); if (f2.features.textureCompressionBC && !(e && *e == '0')) { w2.features.textureCompressionBC = VK_TRUE; vd.bc_native = 1; } }
    // Storage buffers the vertex and pixel shaders write (nvn_vk.c draw_sbuf), where the device has the features
    if (f2.features.fragmentStoresAndAtomics) { w2.features.fragmentStoresAndAtomics = VK_TRUE; vd.frag_stores = 1; }
    if (f2.features.vertexPipelineStoresAndAtomics) w2.features.vertexPipelineStoresAndAtomics = VK_TRUE;
    // Storage images of Unknown format (dm/android/tools/spv_add_caps.py declares StorageImage{Read,Write}WithoutFormat for them: Paper Mario: The Thousand-Year Door's compute shaders);
    // DM_VK_NOFMT=0 leaves the features off, and vk_shader.c then refuses those modules
    { const char *e = getenv("DM_VK_NOFMT"); int on = !(e && *e == '0');
      if (on && f2.features.shaderStorageImageWriteWithoutFormat) { w2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE; vd.nofmt_w = 1; }
      if (on && f2.features.shaderStorageImageReadWithoutFormat) { w2.features.shaderStorageImageReadWithoutFormat = VK_TRUE; vd.nofmt_r = 1; }
      VLOG("storage images without format: write %s, read %s%s", vd.nofmt_w ? "on" : "off", vd.nofmt_r ? "on" : "off", on ? "" : " (DM_VK_NOFMT=0)"); }
    uint32_t nde = 0; vk.vkEnumerateDeviceExtensionProperties(vd.pd, NULL, &nde, NULL);
    VkExtensionProperties *de = calloc(nde ? nde : 1, sizeof *de); vk.vkEnumerateDeviceExtensionProperties(vd.pd, NULL, &nde, de);
    const char *dx[8]; uint32_t ndx = 0;
    // The draw's resources go in one push-descriptor set (nvn_vk.c draw_desc); near = -w clips as NVN's default does with depth clip control's negativeOneToOne (mtl_set_depth_zero). Without
    // either nothing is drawn (said once at the first draw)
    if (has_ext(de, nde, "VK_KHR_push_descriptor")) {
        dx[ndx++] = "VK_KHR_push_descriptor"; vd.push = 1;
        VkPhysicalDevicePushDescriptorPropertiesKHR pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pp}; vk.vkGetPhysicalDeviceProperties2(vd.pd, &q2); vd.max_push = pp.maxPushDescriptors;
    }
    VkPhysicalDeviceDepthClipControlFeaturesEXT dcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    if (has_ext(de, nde, "VK_EXT_depth_clip_control")) {
        VkPhysicalDeviceFeatures2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &dcc}; vk.vkGetPhysicalDeviceFeatures2(vd.pd, &q2);
        if (dcc.depthClipControl) { dx[ndx++] = "VK_EXT_depth_clip_control"; dcc.pNext = w12.pNext; w12.pNext = &dcc; vd.clip_ctl = 1; }
    }
    // DM_VK_DIAG=afbc (diagnostics): VK_EXT_image_compression_control, to ask the driver whether each kind of host image is compressed (AFBC on Mali; nvn_vk.c img_comp)
    VkPhysicalDeviceImageCompressionControlFeaturesEXT icc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT};
    { const char *dg = getenv("DM_VK_DIAG");
      if (dg && strstr(dg, "afbc") && has_ext(de, nde, "VK_EXT_image_compression_control")) {
        VkPhysicalDeviceFeatures2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &icc}; vk.vkGetPhysicalDeviceFeatures2(vd.pd, &q2);
        if (icc.imageCompressionControl) { dx[ndx++] = "VK_EXT_image_compression_control"; icc.pNext = w12.pNext; w12.pNext = &icc; vd.layout2 = (PFN_vkGetImageSubresourceLayout2EXT)1; }
      } else if (dg && strstr(dg, "afbc")) VLOG("[RED] DM_VK_DIAG=afbc: the device has no VK_EXT_image_compression_control"); }
    if (vd.surface_ext) { if (has_ext(de, nde, "VK_KHR_swapchain")) dx[ndx++] = "VK_KHR_swapchain"; else { VLOG("[RED] the device has no VK_KHR_swapchain: nothing is shown"); vd.surface_ext = NULL; } }
    // Portability subset (MoltenVK): what it lacks is invalid to use, so every feature it offers is enabled; nvn_vk.c avoids the uses whose feature is absent (samp_make, img_sview)
    vkn_portability_features pf = {VKN_STYPE_PORTABILITY_SUBSET_FEATURES, NULL};
    if (has_ext(de, nde, "VK_KHR_portability_subset")) {
        dx[ndx++] = "VK_KHR_portability_subset";
        VkPhysicalDeviceFeatures2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &pf}; vk.vkGetPhysicalDeviceFeatures2(vd.pd, &q2);
        pf.pNext = w12.pNext; w12.pNext = &pf;
        vd.port = 1; vd.port_bias = pf.samplerMipLodBias != 0; vd.port_swz = pf.imageViewFormatSwizzle != 0;
        VLOG("portability subset: on; samplerMipLodBias %d, imageViewFormatSwizzle %d, imageViewFormatReinterpretation %d, mutableComparisonSamplers %d, triangleFans %d, pointPolygons %d",
             vd.port_bias, vd.port_swz, (int)pf.imageViewFormatReinterpretation, (int)pf.mutableComparisonSamplers, (int)pf.triangleFans, (int)pf.pointPolygons);
    }
    free(de);
    float prio = 1.f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, vd.qfam, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &w2, 0, 1, &qi, 0, NULL, ndx, dx, NULL};
    if ((r = vk.vkCreateDevice(vd.pd, &dci, NULL, &vd.dev))) { VLOG("[RED] vkCreateDevice: %d: nothing is drawn", r); vd.dev = NULL; return 0; }
#define VKN_D(n) vk.n = (PFN_##n)vk.vkGetDeviceProcAddr(vd.dev, #n);
    VKN_DEVICE(VKN_D)
#undef VKN_D
    if (vd.layout2) vd.layout2 = (PFN_vkGetImageSubresourceLayout2EXT)vk.vkGetDeviceProcAddr(vd.dev, "vkGetImageSubresourceLayout2EXT");
    vk.vkGetDeviceQueue(vd.dev, vd.qfam, 0, &vd.q);
    VLOG("draw features: push descriptors %d (max %u), depth clip control %d, depthClamp %d, independentBlend %d, dualSrcBlend %d, depthBiasClamp %d, fillModeNonSolid %d, shaderImageGatherExtended %d, "
         "shaderDrawParameters %d, sampleRateShading %d; minUniformBufferOffsetAlignment %llu, maxUniformBufferRange %u", vd.push, vd.max_push, vd.clip_ctl, vd.depth_clamp, vd.indep_blend, vd.dual_src, vd.bias_clamp,
         vd.fill_ns, vd.gather_ext, vd.draw_params, vd.sample_rate, (unsigned long long)vd.props.limits.minUniformBufferOffsetAlignment, vd.props.limits.maxUniformBufferRange);
    return 1;
}
