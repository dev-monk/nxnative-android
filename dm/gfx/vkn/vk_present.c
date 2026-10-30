// Vulkan backend: the window surface, the swapchain and the present thread (vkn.h). hle_nvn.c's present (mtl_present) only enqueues a ticket, as nvn_mtl.m's present_enqueue does; this thread
// takes the tickets in order, draws the window texture onto a swapchain image and presents it, then marks the ticket done (mtl_present_wait: the image may be acquired again; its later rendering
// is queued after this read). The surface is made on the front end's window: Android front_native_window() (VK_KHR_android_surface), the Mac front_metal_layer() (VK_EXT_metal_surface, KosmicKrisp).
// The picture: aspect kept with black bars (filled when the aspects differ by under 1%, as nvn_mtl.m present_encode), the crop, and for NVN's lower-left origin a vertical flip: PLAN.md 2.4's
// Y paragraph keeps positive viewport heights, so the window image is stored bottom-up. On Android the swapchain takes the surface's current transform as its preTransform and the picture is
// drawn rotated (no compositor rotation, no SUBOPTIMAL). Present work signals its own timeline semaphore; NVN fence values stay the NVN timeline's. DM_NVN_SHOT's screenshots are taken on
// this thread too, after their present and before its ticket is done (shot_one), with or without a window
#include "vkn.h"
#include "present_spv.h"
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef __ANDROID__
#include <android/native_window.h>
#endif
void *front_metal_layer(void);
void *front_native_window(void);

// ---------- Tickets (any thread → the present thread) ----------
#define PQ 8
typedef struct { uint64_t ticket, serial; uint32_t w, h; int crop[4]; char shot[1024]; } pq_ent;   // shot: DM_NVN_SHOT's path for this present, "" = none (shot_one)
static pq_ent g_pq[PQ];
static uint64_t g_pq_in, g_pq_read, g_pq_done;   // under g_pmu
static int g_lost_req, g_thread_on, g_dead;       // g_dead: the present objects could not be made, or the queue could not be idled to let the swapchain go (wedge): tickets are retired at once
static pthread_mutex_t g_pmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pcv = PTHREAD_COND_INITIALIZER;
// The window going: dm_gfx_window_lost has set g_lost_req (present_one asks before each step that uses the window). Taken under vkn_qmu too: nothing takes vkn_qmu under g_pmu
static int win_going(void) { pthread_mutex_lock(&g_pmu); int r = g_lost_req; pthread_mutex_unlock(&g_pmu); return r; }
static _Atomic int g_plog;   // the next present logs what it shows (presents 1, 60, 600, then every 1800: mtl_present)

// ---------- Present objects (made once on the present thread) ----------
#define NS 3   // present slots in flight: command buffer, acquire semaphore, descriptor set
typedef struct { VkCommandPool pool; VkCommandBuffer cb; VkSemaphore acq; VkDescriptorSet ds; uint64_t value; } pslot;
static pslot g_ps[NS]; static uint64_t g_pseq;
static _Atomic uint64_t g_pval;   // the present timeline's last signalled value (present thread)
// The present timeline: made on the present thread (objs_make) and read on others too (vkn_pres_done in the sweep, the flush and the drain), so published with a release store and read with ptl()
static _Atomic(VkSemaphore) g_ptl;
static VkSemaphore ptl(void) { return atomic_load_explicit(&g_ptl, memory_order_acquire); }
static VkSampler g_samp; static VkDescriptorSetLayout g_dsl; static VkPipelineLayout g_pl; static VkDescriptorPool g_dp; static VkShaderModule g_vs, g_fs;
// The present's vertex shader on its own, for nvn_vk.c's passes that cover their targets with one triangle (csf_run: a dispatch as a fragment pass; uv unused). Submit thread
VkShaderModule vkn_tri_vs(void) {
    static VkShaderModule m;
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof present_vert_spv, present_vert_spv};
    if (!m && vk.vkCreateShaderModule(vd.dev, &ci, NULL, &m)) m = VK_NULL_HANDLE;
    return m;
}
// The present pipelines by their parameters (specialization constants): a few kept, so a crop that alternates (or a format change and back) does not recompile; the least recently used goes
#define NPC 4
typedef struct { VkFormat fmt; int32_t rot; float p[8]; } pkey;   // p: U0 V0 US VS (the vertex stage), then the clamp X0 Y0 X1 Y1 (the fragment stage); no padding, compared with memcmp
static struct { pkey k; VkPipeline p; uint64_t used; } g_pc[NPC];   // used: the present timeline value of its last present
static vkn_img g_test; static int g_test_on;   // DM_VK_PRESENT_TEST=1: a quadrant pattern in place of the window texture (orientation and flip check)

// ---------- Surface and swapchain (present thread; torn down on window loss) ----------
static void *g_win;   // the window / layer the surface is on (Android: holds a reference)
#define SCN 16   // swapchain images at most (minImageCount + 1 asked; a surface that gives more is refused in red, so an acquired index always fits the arrays)
static VkSurfaceKHR g_surf; static VkSwapchainKHR g_sc; static uint32_t g_sc_n; static VkImage g_sc_img[SCN]; static VkImageView g_sc_view[SCN]; static VkSemaphore g_sc_done[SCN];
static VkExtent2D g_sc_ext, g_sc_req; static VkFormat g_sc_fmt; static VkSurfaceTransformFlagBitsKHR g_sc_tf; static int g_sc_stale, g_sc_forced, g_sw_said; static uint64_t g_sc_made;
// g_sc_forced: the preTransform is not the surface's (it reported no rotation), so SUBOPTIMAL is expected every present and does not remake the swapchain; g_sw_said: a failure to make one was
// logged since the last one was made (said once per spell)
// Presents in a row that ended in a remade swapchain or surface (OUT_OF_DATE, SURFACE_LOST, SUBOPTIMAL not expected), or in a swapchain that could not be made for one of the expected reasons
// (swap_fail's plain ones: OUT_OF_DATE, SURFACE_LOST, a window with no size): normal once around a window change, not every frame. The first few are listed; after that the lines they bring
// (dropped, swapchain recreated, surface made / released) are counted only, and BAD_RED in a row is said in red, once. A present that asks for no new swapchain ends the run (bad_end)
#define BAD_SAY 3
#define BAD_RED 120
static uint64_t g_bad_run;
static int bad_quiet(void) { return g_bad_run >= BAD_SAY; }
static void bad_one(uint64_t ticket, const char *what, VkResult r) {   // a present that left the swapchain or surface to be remade (once a present: present_one)
    char rs[24] = ""; if (r) snprintf(rs, sizeof rs, " %d", r);
    g_bad_run++;
    if (g_bad_run <= BAD_SAY) VLOG("present %llu: %s%s: the swapchain is remade before the next%s", (unsigned long long)ticket, what, rs, g_bad_run == BAD_SAY ? " (more of these in a row are counted, not listed)" : "");
    if (g_bad_run == BAD_RED) {
        if (r == VK_SUBOPTIMAL_KHR) VRED_ONCE("present: %d presents in a row ended in %s%s (SUBOPTIMAL): frames are shown, but the swapchain is remade every frame", BAD_RED, what, rs);
        else VRED_ONCE("present: %d presents in a row ended in %s%s, the swapchain or surface remade (or tried) before each: nothing is shown", BAD_RED, what, rs);
    }
}
static const char *g_sf_what; static VkResult g_sf_r; static int g_sf;   // this present's swapchain could not be made for an expected reason (swap_fail): present_one counts it
static void bad_end(void) { if (g_bad_run > BAD_SAY) VLOG("presents go through again after %llu in a row that ended in a remade swapchain", (unsigned long long)g_bad_run); g_bad_run = 0; }

// The present timeline's completed value (vkn_sweep): a value that cannot be read (an error leaves it undefined) counts as nothing done, so nothing a present used is freed on it
uint64_t vkn_pres_done(void) { VkSemaphore t = ptl(); uint64_t v = 0; if (!t || vk.vkGetSemaphoreCounterValue(vd.dev, t, &v) != VK_SUCCESS) return 0; return v; }
// 1 = the present timeline reached v. 0 = not in 10 s, or an error: the caller leaves what that present used alone (a slow GPU is not taken for a lost one: only an error says that)
static int pres_wait(uint64_t v) {
    VkSemaphore t = ptl();
    if (!t || !v) return 1;
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &t, &v};
    for (int i = 0; i < 10; i++) { VkResult r = vk.vkWaitSemaphores(vd.dev, &wi, 1000000000ull); if (r == VK_SUCCESS) return 1; if (r != VK_TIMEOUT) { vkn_gpu_lost("the present's vkWaitSemaphores", r); return 0; } }
    VRED_ONCE("a present has not finished on the GPU in 10 s: that frame is not shown (and the next waits again)"); return 0;
}
static int tf_rot(VkSurfaceTransformFlagBitsKHR t) { return t == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR || t == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR || t == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR || t == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR; }
// Android reports the extent as the window is now (landscape here), the transform from the display's natural orientation (portrait): with the transform taken as the preTransform, the
// swapchain is made in the natural orientation (sides swapped for 90 / 270) and the picture is drawn rotated into it, as Google's pre-rotation guide does (present_one's lw / lh assume this)
static VkExtent2D nat_ext(VkExtent2D e, VkSurfaceTransformFlagBitsKHR t) { if (t == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR || t == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) { uint32_t w = e.width; e.width = e.height; e.height = w; } return e; }
static const char *tf_name(VkSurfaceTransformFlagBitsKHR t) {
    return t == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR ? "identity" : t == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ? "rotate-90" : t == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ? "rotate-180" :
           t == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR ? "rotate-270" : "other";
}

// The queue idled before the swapchain's objects go (what). 1 = idle, or the device lost (it runs nothing more): they may go. Any other error (out of host / device memory) leaves the queue's
// state unknown: the wait is tried again for up to QIDLE_NS (inside the window loss's 4 s, dm_gfx_window_lost), then 0: the caller keeps them (wedge)
#define QIDLE_NS 3000000000ull
static int g_wedged;   // present thread: the queue could not be idled, the surface, swapchain, views and semaphores are kept (leaked) and nothing is presented again (present_main: g_dead)
static int queue_idle(const char *what) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        pthread_mutex_lock(&vkn_qmu); VkResult r = vk.vkQueueWaitIdle(vd.q); pthread_mutex_unlock(&vkn_qmu);
        if (r == VK_SUCCESS) return 1;
        if (r == VK_ERROR_DEVICE_LOST) { vkn_gpu_lost("vkQueueWaitIdle", r); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if ((uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)t1.tv_nsec - (uint64_t)t0.tv_nsec >= QIDLE_NS) {
            VRED_ONCE("vkQueueWaitIdle %d before %s, for %llu s: the queue may still use the swapchain, so it, its views and semaphores and the surface are kept (leaked) and nothing is presented for the rest of the run",
                      r, what, (unsigned long long)(QIDLE_NS / 1000000000ull));
            return 0;
        }
        struct timespec d = {0, 10000000}; nanosleep(&d, NULL);
    }
}
// The queue could not be idled: what was made on the window stays made (destroying it could pull objects from under a pending present), only this thread's own reference to the window goes,
// so the front end can let it go; nothing draws into it again (present_main ends the present thread with g_dead set)
static void wedge(void) {
    g_wedged = 1;
#ifdef __ANDROID__
    if (g_win) ANativeWindow_release((ANativeWindow *)g_win);
#endif
    g_win = NULL;
}
static void swap_free(void) {   // the swapchain's images' views and semaphores, then the swapchain; the queue is idle (callers)
    for (uint32_t i = 0; i < g_sc_n; i++) { vk.vkDestroyImageView(vd.dev, g_sc_view[i], NULL); vk.vkDestroySemaphore(vd.dev, g_sc_done[i], NULL); g_sc_view[i] = VK_NULL_HANDLE; g_sc_done[i] = VK_NULL_HANDLE; }
    g_sc_n = 0;
    if (g_sc) vk.vkDestroySwapchainKHR(vd.dev, g_sc, NULL);
    g_sc = VK_NULL_HANDLE;
}
// Window loss (background, a new window): everything made on the window goes before the front end lets it go. The queue is idled first: a present's wait on its semaphore has no fence to
// wait for (no VK_EXT_swapchain_maintenance1 assumed), and an image still on screen belongs to the swapchain
static void surface_free(const char *why) {
    if (g_wedged || (!g_surf && !g_win)) return;
    if (!queue_idle("the surface and swapchain are released")) { wedge(); return; }
    swap_free();
    if (g_surf) vk.vkDestroySurfaceKHR(vd.inst, g_surf, NULL);
    g_surf = VK_NULL_HANDLE;
#ifdef __ANDROID__
    if (g_win) ANativeWindow_release((ANativeWindow *)g_win);
#endif
    if (!bad_quiet() || win_going()) VLOG("surface and swapchain released (%s)", why);   // the window loss's line always
    g_win = NULL;
}
static int surface_make(void *w) {
#ifdef __ANDROID__
    vkn_android_surface_ci ci = {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, NULL, 0, w};
#else
    vkn_metal_surface_ci ci = {VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT, NULL, 0, w};
#endif
    VkResult r = vk.create_surface(vd.inst, &ci, NULL, &g_surf);
    if (r) { VRED_ONCE("%s: surface creation failed (%d): nothing is shown", vd.surface_ext, r); g_surf = VK_NULL_HANDLE; return 0; }
    VkBool32 ok = VK_FALSE; vk.vkGetPhysicalDeviceSurfaceSupportKHR(vd.pd, vd.qfam, g_surf, &ok);
    if (!ok) { VRED_ONCE("queue family %u cannot present to the surface: nothing is shown", vd.qfam); vk.vkDestroySurfaceKHR(vd.inst, g_surf, NULL); g_surf = VK_NULL_HANDLE; return 0; }
#ifdef __ANDROID__
    ANativeWindow_acquire((ANativeWindow *)w);   // kept until surface_free: the front end waits for it in dm_gfx_window_lost before the window goes
#endif
    g_win = w;
    if (!bad_quiet()) VLOG("surface made on the window %p (%s)", w, vd.surface_ext);
    return 1;
}
// A swapchain that cannot be made: expected while a window comes or goes (OUT_OF_DATE, SURFACE_LOST, a window with no size yet: plain, once per spell), red otherwise; a lost GPU says so
static int swap_fail(const char *what, VkResult r, const char *why) {
    if (r == VK_ERROR_DEVICE_LOST) { vkn_gpu_lost(what, r); return 0; }
    if (r == VK_ERROR_SURFACE_LOST_KHR) surface_free("the surface was lost");   // made again on the front end's window at the next present
    if (r && r != VK_ERROR_OUT_OF_DATE_KHR && r != VK_ERROR_SURFACE_LOST_KHR) VRED_ONCE("swapchain: %s: %d (%s): nothing is shown until one can be made", what, r, why);
    else {   // expected around a window change; said plainly once per spell, and counted in the run of bad presents (bad_one, via present_one) so a spell that never ends turns red
        if (!g_sw_said) { if (r) VLOG("swapchain: %s: %d (%s): presents dropped until one can be made", what, r, why); else VLOG("swapchain: %s (%s): presents dropped until one can be made", what, why); }
        g_sf = 1; g_sf_what = what; g_sf_r = r;
    }
    g_sw_said = 1; return 0;
}
// The swapchain for the surface as it is now: the current extent and transform, a UNORM format (the window texture's bytes pass through), FIFO. 1 = made, 0 = none, -1 = the window is going
// (asked before the window is used and again after the queue's wait, just before vkCreateSwapchainKHR: the old swapchain, if any, is left for surface_free, next on this thread)
static int swap_make(const char *why) {
    if (win_going()) return -1;
    VkSurfaceCapabilitiesKHR c; VkResult r = vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vd.pd, g_surf, &c);
    if (r) return swap_fail("surface capabilities", r, why);
    VkSurfaceTransformFlagBitsKHR tf = c.currentTransform;
    if (!tf_rot(tf) || !(c.supportedTransforms & tf)) {
        VRED_ONCE("swapchain: surface transform %#x is not a rotation: shown unrotated by the compositor", (unsigned)tf);
        tf = c.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : c.currentTransform;
    }
    VkExtent2D ext = c.currentExtent;
    if (ext.width == 0xFFFFFFFFu) {   // the surface takes the swapchain's size: the window's
#ifdef __ANDROID__
        ext.width = (uint32_t)ANativeWindow_getWidth((ANativeWindow *)g_win); ext.height = (uint32_t)ANativeWindow_getHeight((ANativeWindow *)g_win);
#else
        ext.width = 1920; ext.height = 1080;   // Apple surfaces report the layer's size (KosmicKrisp, MoltenVK): only a driver that does not lands here
#endif
    }
    if (!ext.width || !ext.height) return swap_fail("the window has no size yet", VK_SUCCESS, why);
    ext = nat_ext(ext, tf); VkExtent2D req = ext;
    if (ext.width < c.minImageExtent.width) ext.width = c.minImageExtent.width; if (ext.width > c.maxImageExtent.width) ext.width = c.maxImageExtent.width;
    if (ext.height < c.minImageExtent.height) ext.height = c.minImageExtent.height; if (ext.height > c.maxImageExtent.height) ext.height = c.maxImageExtent.height;
    uint32_t nf = 0; vk.vkGetPhysicalDeviceSurfaceFormatsKHR(vd.pd, g_surf, &nf, NULL);
    VkSurfaceFormatKHR fm[64]; if (nf > 64) nf = 64; vk.vkGetPhysicalDeviceSurfaceFormatsKHR(vd.pd, g_surf, &nf, fm);
    if (!nf) return swap_fail("the surface lists no formats", VK_ERROR_FORMAT_NOT_SUPPORTED, why);
    VkSurfaceFormatKHR f = fm[0]; int unorm = 0;
    for (uint32_t i = 0; i < nf; i++) if ((fm[i].format == VK_FORMAT_R8G8B8A8_UNORM || fm[i].format == VK_FORMAT_B8G8R8A8_UNORM) && fm[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { f = fm[i]; unorm = 1; break; }
    if (!unorm) VRED_ONCE("swapchain: no R8G8B8A8 / B8G8R8A8 UNORM format among the surface's %u: format %d, colour space %d taken (the colours may be off)", nf, f.format, f.colorSpace);
    VkCompositeAlphaFlagBitsKHR ca = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;   // Android usually offers only INHERIT: the window is opaque, and the shader writes alpha 1 either way
    const VkCompositeAlphaFlagBitsKHR cas[4] = {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR};
    for (int i = 0; i < 4; i++) if (c.supportedCompositeAlpha & cas[i]) { ca = cas[i]; break; }
    uint32_t n = c.minImageCount + 1; if (c.maxImageCount && n > c.maxImageCount) n = c.maxImageCount;
    VkSwapchainKHR old = g_sc;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, NULL, 0, g_surf, n, f.format, f.colorSpace, ext, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL, tf, ca, VK_PRESENT_MODE_FIFO_KHR, VK_TRUE, old};
    if (old && !queue_idle("the old swapchain is retired")) { wedge(); return 0; }   // the old one's semaphores and views go below (or, the queue not idle, never: wedge)
    if (win_going()) return -1;   // the wait may have been long: nothing new on a window that is going
    VkSwapchainKHR sc; r = vk.vkCreateSwapchainKHR(vd.dev, &ci, NULL, &sc);
    swap_free();   // the old one is retired either way
    if (r) return swap_fail("vkCreateSwapchainKHR", r, why);
    g_sc = sc; g_sc_ext = ext; g_sc_req = req; g_sc_fmt = f.format; g_sc_tf = tf; g_sc_forced = tf != c.currentTransform; g_sc_stale = 0;
    uint32_t ni = 0; vk.vkGetSwapchainImagesKHR(vd.dev, sc, &ni, NULL);
    if (ni > SCN) { swap_free(); return swap_fail("the swapchain has more images than the present keeps", VK_ERROR_TOO_MANY_OBJECTS, why); }
    if ((r = vk.vkGetSwapchainImagesKHR(vd.dev, sc, &ni, g_sc_img))) { swap_free(); return swap_fail("vkGetSwapchainImagesKHR", r, why); }
    g_sc_n = ni;   // swap_free destroys what was made below, should one fail
    for (uint32_t i = 0; i < ni; i++) {
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_sc_img[i], VK_IMAGE_VIEW_TYPE_2D, f.format, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        // per image: a present's wait semaphore is free again once its image is re-acquired
        if ((r = vk.vkCreateImageView(vd.dev, &vi, NULL, &g_sc_view[i])) || (r = vk.vkCreateSemaphore(vd.dev, &si, NULL, &g_sc_done[i]))) { swap_free(); return swap_fail("the swapchain images' views / semaphores", r, why); }
    }
    g_sc_made++; g_sw_said = 0;
    if (!bad_quiet()) VLOG("swapchain %s (%s): %ux%u (the display's natural orientation) format %d, %u images, FIFO, transform %s%s, composite alpha %#x", g_sc_made == 1 ? "made" : "recreated", why, ext.width, ext.height, f.format, ni, tf_name(tf),
         tf == c.currentTransform ? "" : " (not the surface's)", (unsigned)ca);
    return 1;
}
// Before each present: the window the front end has now, and a swapchain that fits the surface as it is (rotation, size). 0 = nothing to present to, -1 = the window is going
static int swap_ensure(void) {
#ifdef __ANDROID__
    void *w = front_native_window();
#else
    void *w = front_metal_layer();
#endif
    if (w != g_win && g_win) { surface_free("the front end's window changed"); if (g_wedged) return 0; }   // wedged: said in red (queue_idle), present_one stops
    if (!w || !vd.surface_ext) return 0;
    if (win_going()) return -1;   // surface_free idles the queue: asked again before a surface is made
    if (!g_surf && !surface_make(w)) return 0;
    if (g_sc && !g_sc_stale) {   // the surface's extent or transform moved (rotation, resize): a new swapchain before this present
        VkSurfaceCapabilitiesKHR c;
        if (!vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vd.pd, g_surf, &c)) {
            VkExtent2D ne = nat_ext(c.currentExtent, g_sc_tf);   // against what swap_make asked for (before its clamp), so a clamped extent does not remake it every frame
            if ((c.currentExtent.width != 0xFFFFFFFFu && (ne.width != g_sc_req.width || ne.height != g_sc_req.height))
                || (c.currentTransform != g_sc_tf && tf_rot(c.currentTransform) && (c.supportedTransforms & c.currentTransform))) g_sc_stale = 2;
        }
    }
    if (!g_sc || g_sc_stale) return swap_make(!g_sc ? "a new surface" : g_sc_stale == 2 ? "the surface's size or rotation changed" : "the swapchain was out of date");
    return 1;
}

// ---------- The present pipeline: rebuilt when the swapchain format, the crop / flip or the rotation changes (all rare) ----------
static VkPipeline pipe_get(const pkey *k, uint64_t use) {   // use: the present timeline value of the present that binds it
    int v = -1;
    for (int i = 0; i < NPC; i++) if (g_pc[i].p && !memcmp(&g_pc[i].k, k, sizeof *k)) { g_pc[i].used = use; return g_pc[i].p; }
    for (int i = 0; i < NPC && v < 0; i++) if (!g_pc[i].p) v = i;
    if (v < 0) {   // the least recently used goes, once its last present has completed (long since, as a rule)
        v = 0; for (int i = 1; i < NPC; i++) if (g_pc[i].used < g_pc[v].used) v = i;
        if (!pres_wait(g_pc[v].used)) return VK_NULL_HANDLE;
        vk.vkDestroyPipeline(vd.dev, g_pc[v].p, NULL); g_pc[v].p = VK_NULL_HANDLE;
    }
    struct { float u0, v0, us, vs; int32_t rot; float x0, y0, x1, y1; } sd = {k->p[0], k->p[1], k->p[2], k->p[3], k->rot, k->p[4], k->p[5], k->p[6], k->p[7]};
    VkSpecializationMapEntry me[9] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}, {5, 20, 4}, {6, 24, 4}, {7, 28, 4}, {8, 32, 4}};
    VkSpecializationInfo si = {9, me, sizeof sd, &sd};   // one map for both stages: an ID a stage does not declare is ignored there
    VkFormat fmt = k->fmt;
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_vs, "main", &si},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_fs, "main", &si}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba = {0}; ba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount = 1; cb.pAttachments = &ba;
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &fmt, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo pi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, 2, st, &vi, &ia, NULL, &vp, &rs, &ms, NULL, &cb, &ds, g_pl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VkPipeline pl; VkResult r = vk.vkCreateGraphicsPipelines(vd.dev, VK_NULL_HANDLE, 1, &pi, NULL, &pl);
    if (r) { VRED_ONCE("present: pipeline creation failed (%d): frames show black", r); return VK_NULL_HANDLE; }
    g_pc[v].k = *k; g_pc[v].p = pl; g_pc[v].used = use;
    return pl;
}

static int objs_make(void) {
    VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo tsi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti}, bsi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore tl; if (vk.vkCreateSemaphore(vd.dev, &tsi, NULL, &tl)) return 0;
    atomic_store_explicit(&g_ptl, tl, memory_order_release);
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; sci.maxLod = 0.f;
    if (vk.vkCreateSampler(vd.dev, &sci, NULL, &g_samp)) return 0;
    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &g_samp};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b};
    if (vk.vkCreateDescriptorSetLayout(vd.dev, &dli, NULL, &g_dsl)) return 0;
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_dsl, 0, NULL};
    if (vk.vkCreatePipelineLayout(vd.dev, &pli, NULL, &g_pl)) return 0;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NS};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, NS, 1, &ps};
    if (vk.vkCreateDescriptorPool(vd.dev, &dpi, NULL, &g_dp)) return 0;
    VkShaderModuleCreateInfo vsi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof present_vert_spv, present_vert_spv};
    VkShaderModuleCreateInfo fsi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof present_frag_spv, present_frag_spv};
    if (vk.vkCreateShaderModule(vd.dev, &vsi, NULL, &g_vs) || vk.vkCreateShaderModule(vd.dev, &fsi, NULL, &g_fs)) return 0;
    for (int i = 0; i < NS; i++) {
        VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, vd.qfam};
        if (vk.vkCreateCommandPool(vd.dev, &cpi, NULL, &g_ps[i].pool)) return 0;
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_ps[i].pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vk.vkAllocateCommandBuffers(vd.dev, &ai, &g_ps[i].cb) || vk.vkCreateSemaphore(vd.dev, &bsi, NULL, &g_ps[i].acq)) return 0;
        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_dp, 1, &g_dsl};
        if (vk.vkAllocateDescriptorSets(vd.dev, &dai, &g_ps[i].ds)) return 0;
    }
    return 1;
}

static void img_barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags asp, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, ss, sa, ds, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {asp, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &b};
    vk.vkCmdPipelineBarrier2(cb, &di);
}
static VkResult submit_one(VkCommandBuffer cb, VkSemaphore wait, VkSemaphore sig, uint64_t tl) {   // under vkn_qmu; failed = the GPU lost (said there): nothing of it is signalled
    VkSemaphoreSubmitInfo w = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, wait, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0};
    VkSemaphoreSubmitInfo s[2] = {{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, ptl(), tl, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0},
                                  {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, sig, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0}};
    VkCommandBufferSubmitInfo c = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, cb, 0};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, wait ? 1 : 0, &w, 1, &c, sig ? 2 : 1, s};
    VkResult r = vk.vkQueueSubmit2(vd.q, 1, &si, VK_NULL_HANDLE);
    if (r) vkn_gpu_lost("the present's vkQueueSubmit2", r);
    return r;
}
static pslot *slot_next(void) {
    pslot *s = &g_ps[g_pseq++ % NS];
    if (!pres_wait(s->value)) return NULL;   // its last present never finished (the GPU lost): its command buffer and semaphore cannot be reused
    vk.vkResetCommandPool(vd.dev, s->pool, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    vk.vkBeginCommandBuffer(s->cb, &bi);
    return s;
}
// The quadrant pattern (DM_VK_PRESENT_TEST=1), stored as NVN's lower-left origin stores a picture: memory rows 0-17, the picture's bottom half, red | green; rows 18-35, its top, blue | white.
// The screen must show blue top left, white top right, red bottom left, green bottom right, whatever the phone's rotation
static int test_make(void) {   // 1 = the pattern is shown
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {64, 36, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkMemoryRequirements mr; int mt;
    if (vk.vkCreateImage(vd.dev, &ii, NULL, &g_test.img)) return 0;
    vk.vkGetImageMemoryRequirements(vd.dev, g_test.img, &mr);
    if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) return 0;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (vk.vkAllocateMemory(vd.dev, &ai, NULL, &g_test.mem) || vk.vkBindImageMemory(vd.dev, g_test.img, g_test.mem, 0)) return 0;
    g_test.fmt = g_test.sfmt = VK_FORMAT_R8G8B8A8_UNORM; g_test.aspect = VK_IMAGE_ASPECT_COLOR_BIT; g_test.w = 64; g_test.h = 36; g_test.layers = g_test.levels = 1; g_test.nvn_fmt = 0x25;
    VkImageView v = vkn_img_view(&g_test, 0, 0, 0); if (!v) return 0;
    pslot *s = slot_next(); if (!s) return 0;
    img_barrier(s->cb, g_test.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {64, 36}}, 1, 0, 1, &ca, NULL, NULL};
    vk.vkCmdBeginRendering(s->cb, &ri);
    const float col[4][4] = {{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 1}};
    for (int q = 0; q < 4; q++) {
        VkClearAttachment a = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {{{col[q][0], col[q][1], col[q][2], col[q][3]}}}};
        VkClearRect r = {{{(q & 1) * 32, (q >> 1) * 18}, {32, 18}}, 0, 1};
        vk.vkCmdClearAttachments(s->cb, 1, &a, 1, &r);
    }
    vk.vkCmdEndRendering(s->cb);
    vk.vkEndCommandBuffer(s->cb);
    pthread_mutex_lock(&vkn_qmu); uint64_t tv = atomic_load(&g_pval) + 1; VkResult r = submit_one(s->cb, VK_NULL_HANDLE, VK_NULL_HANDLE, tv);
    if (!r) { s->value = tv; atomic_store(&g_pval, tv); }
    pthread_mutex_unlock(&vkn_qmu);
    if (r) return 0;
    g_test.layout = g_test.rec = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; g_test_on = 1;
    VLOG("DM_VK_PRESENT_TEST=1: a quadrant pattern is shown in place of the game's frames (blue top left, white top right, red bottom left, green bottom right when the orientation and the flip are right)");
    return 1;
}

// The window going (dm_gfx_window_lost has set g_lost_req: the front end's onNativeWindowDestroyed is waiting, or has given up waiting after 4 s). Android wants drawing into the window to
// stop before that callback returns, so present_one asks before each step that uses the window (the swapchain, the acquire, the present) and skips the rest: the present thread lets the surface
// go next (present_main). A call already inside the driver when the callback gives up (an acquire, a vkQueuePresentKHR, a swapchain being made) cannot be called back: it ends on the window after the callback has returned
static uint64_t g_shown, g_nowin, g_acq_to, g_going;
static void present_one(const pq_ent *q) {
    if (atomic_load(&vkn_lost)) return;   // said in red once (vkn_gpu_lost): the ticket is retired unshown
    if (win_going()) { if (!g_going++) VLOG("present %llu dropped: the window is going", (unsigned long long)q->ticket); return; }
    g_sf = 0; int se = swap_ensure();
    if (g_wedged) return;   // the queue could not be idled (said in red): nothing more is presented (present_main)
    if (se < 0) { if (!g_going++) VLOG("present %llu dropped: the window is going", (unsigned long long)q->ticket); return; }   // not g_nowin's line: that one is said once a run, for no window
    if (!se && g_sf) { bad_one(q->ticket, g_sf_what, g_sf_r); return; }   // no swapchain for an expected reason (swap_fail): one bad present in the run
    if (!se) { if (!g_nowin++) VLOG("presents are dropped while there is no window to show them on"); return; }
    pslot *s = slot_next(); if (!s) return;
    if (win_going()) { vk.vkEndCommandBuffer(s->cb); if (!g_going++) VLOG("present %llu dropped: the window is going", (unsigned long long)q->ticket); return; }   // slot_next may have waited
    uint32_t idx; VkResult r = vk.vkAcquireNextImageKHR(vd.dev, g_sc, 250000000ull, s->acq, VK_NULL_HANDLE, &idx);   // finite: window loss must not wait behind it (dm_gfx_window_lost)
    int bad = 0;   // this present has asked for a remade swapchain already (bad_one: once a present)
    if (r == VK_SUBOPTIMAL_KHR) { if (!g_sc_forced) { g_sc_stale = 1; bad_one(q->ticket, "vkAcquireNextImageKHR", r); bad = 1; } }   // still presentable: this frame goes, the swapchain is remade before the next
    else if (r) {   // nothing acquired: the semaphore is not signalled, the slot's buffer was never submitted
        vk.vkEndCommandBuffer(s->cb);
        if (r == VK_ERROR_DEVICE_LOST) { vkn_gpu_lost("vkAcquireNextImageKHR", r); return; }
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) {   // expected around window changes: not red (unless it never ends: bad_one)
            g_sc_stale = 1; bad_one(q->ticket, "vkAcquireNextImageKHR", r);
            if (r == VK_ERROR_SURFACE_LOST_KHR) surface_free("the surface was lost");
        } else if (r == VK_TIMEOUT || r == VK_NOT_READY) { if (!g_acq_to++) VLOG("present %llu dropped: no swapchain image within 250 ms (the display held them)", (unsigned long long)q->ticket); }
        else VRED_ONCE("present %llu dropped: vkAcquireNextImageKHR %d", (unsigned long long)q->ticket, r);
        return;
    }
    if (idx >= g_sc_n) {   // cannot happen (swap_make keeps every image); the image stays acquired, so this swapchain is done with
        vk.vkEndCommandBuffer(s->cb); VRED_ONCE("present: acquired image %u of %u", idx, g_sc_n); g_sc_stale = 1; return;
    }
    // The picture's rectangle: aspect kept in the frame as the player sees it (rotated back), then placed on the image (the rectangle is centred, so a rotation only swaps its sides)
    int rot = g_sc_tf == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ? 1 : g_sc_tf == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ? 2 : g_sc_tf == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR ? 3 : 0;
    float lw = (float)(rot & 1 ? g_sc_ext.height : g_sc_ext.width), lh = (float)(rot & 1 ? g_sc_ext.width : g_sc_ext.height);
    uint32_t tw = g_test_on ? g_test.w : q->w, th = g_test_on ? g_test.h : q->h;
    int cx = 0, cy = 0, cw = (int)tw, ch = (int)th;
    if (!g_test_on && (q->crop[2] || q->crop[3])) {   // as nvn_mtl.m present_encode: a crop is set when either size is, and must lie in the texture
        const int *c = q->crop;
        if (c[0] < 0 || c[1] < 0 || c[2] <= 0 || c[3] <= 0 || (uint64_t)c[0] + (uint64_t)c[2] > tw || (uint64_t)c[1] + (uint64_t)c[3] > th)
            VRED_ONCE("present crop out of bounds: %d,%d %dx%d / window texture %ux%u: the whole texture is shown", c[0], c[1], c[2], c[3], tw, th);
        else { cx = c[0]; cy = c[1]; cw = c[2]; ch = c[3]; }
    }
    float src = (float)cw / (float)(ch ? ch : 1), dst = lw / lh, pw = lw, ph = lh;
    if (src > 0 && (dst - src) / src > 0.01f) pw = lh * src; else if (src > 0 && (src - dst) / src > 0.01f) ph = lw / src;
    if (rot & 1) { float t = pw; pw = ph; ph = t; }
    VkViewport vp = {((float)g_sc_ext.width - pw) * 0.5f, ((float)g_sc_ext.height - ph) * 0.5f, pw, ph, 0.f, 1.f};
    float fw = (float)(tw ? tw : 1), fh = (float)(th ? th : 1);
    pkey k = {g_sc_fmt, rot, {cx / fw, cy / fh, cw / fw, ch / fh, (cx + 0.5f) / fw, (cy + 0.5f) / fh, (cx + cw - 0.5f) / fw, (cy + ch - 0.5f) / fh}};   // clamp: the crop's texel centres
    if (vkn_origin_ll() || g_test_on) { k.p[1] += k.p[3]; k.p[3] = -k.p[3]; }   // lower-left origin: the window image is stored bottom-up (the pattern always is); the clamp is unflipped
    VkPipeline pipe = pipe_get(&k, atomic_load(&g_pval) + 1);   // the value this present signals (only this thread advances g_pval)

    // Under the queue lock, so the layout read here is the one the queue will have when this runs (the submit thread publishes layouts and submits under it), and is restored after
    pthread_mutex_lock(&vkn_qmu);
    pthread_mutex_lock(&vkn_tmu);
    vkn_img *e = g_test_on ? &g_test : vkn_img_get(q->serial);
    VkImageLayout L = e ? e->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageView view = e && L != VK_IMAGE_LAYOUT_UNDEFINED && e->aspect == VK_IMAGE_ASPECT_COLOR_BIT ? vkn_img_view(e, 0, 0, 1) : VK_NULL_HANDLE;
    uint64_t val = atomic_load(&g_pval) + 1;
    if (view) e->use_pres = val;
    if (atomic_exchange(&g_plog, 0)) VLOG("present %llu: window texture ser=%llu %ux%u format %#x %s; its last colour clear %g,%g,%g,%g; swapchain %ux%u %s, picture %.0fx%.0f", (unsigned long long)q->ticket, (unsigned long long)q->serial,
                e ? e->w : 0, e ? e->h : 0, e ? e->nvn_fmt : 0, !e ? "has no host image (shown black)" : view ? "shown" : "never written (shown black)", e && e->cleared ? e->clear[0] : 0., e && e->cleared ? e->clear[1] : 0.,
                e && e->cleared ? e->clear[2] : 0., e && e->cleared ? e->clear[3] : 0., g_sc_ext.width, g_sc_ext.height, tf_name(g_sc_tf), pw, ph);
    VkImage simg = view ? e->img : VK_NULL_HANDLE;
    pthread_mutex_unlock(&vkn_tmu);
    if (view) {
        VkDescriptorImageInfo dii = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s->ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii, NULL, NULL};
        vk.vkUpdateDescriptorSets(vd.dev, 1, &w, 0, NULL);
        img_barrier(s->cb, simg, VK_IMAGE_ASPECT_COLOR_BIT, L, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    img_barrier(s->cb, g_sc_img[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);   // after the acquire semaphore's wait, which is at this stage
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, g_sc_view[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0.f, 0.f, 0.f, 1.f}}}};   // the bars
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, g_sc_ext}, 1, 0, 1, &ca, NULL, NULL};
    vk.vkCmdBeginRendering(s->cb, &ri);
    if (view && pipe) {
        VkRect2D sc = {{0, 0}, g_sc_ext};
        vk.vkCmdSetViewport(s->cb, 0, 1, &vp); vk.vkCmdSetScissor(s->cb, 0, 1, &sc);
        vk.vkCmdBindPipeline(s->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vk.vkCmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl, 0, 1, &s->ds, 0, NULL);
        vk.vkCmdDraw(s->cb, 3, 1, 0, 0);
    }
    vk.vkCmdEndRendering(s->cb);
    img_barrier(s->cb, g_sc_img[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
    if (view) img_barrier(s->cb, simg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, L, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);   // back as the next NVN submission expects it
    vk.vkEndCommandBuffer(s->cb);
    if (submit_one(s->cb, s->acq, g_sc_done[idx], val)) { pthread_mutex_unlock(&vkn_qmu); return; }   // the GPU lost (said in red): no present waits on a semaphore nothing signals
    atomic_store(&g_pval, val); s->value = val;
    // Submitted (it waits on the acquire and puts the window texture's layout back): only the present is left out when the window is going. The image stays acquired and its semaphore signalled,
    // both gone with the swapchain when the surface goes (surface_free, next on this thread: it idles the queue first)
    if (win_going()) { pthread_mutex_unlock(&vkn_qmu); if (!g_going++) VLOG("present %llu not shown: the window is going", (unsigned long long)q->ticket); return; }
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &g_sc_done[idx], 1, &g_sc, &idx, NULL};
    r = vk.vkQueuePresentKHR(vd.q, &pi);
    pthread_mutex_unlock(&vkn_qmu);
    if ((r == VK_SUBOPTIMAL_KHR && !g_sc_forced) || r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { g_sc_stale = 1; if (!bad) bad_one(q->ticket, "vkQueuePresentKHR", r); }
    else if ((!r || r == VK_SUBOPTIMAL_KHR) && !g_sc_stale) bad_end();   // the run ends only with a present that asked for no remade swapchain (an acquire's SUBOPTIMAL counts too)
    if (r == VK_ERROR_SURFACE_LOST_KHR) surface_free("the surface was lost at present");
    else if (r == VK_ERROR_DEVICE_LOST) vkn_gpu_lost("vkQueuePresentKHR", r);
    else if (r && r != VK_SUBOPTIMAL_KHR && r != VK_ERROR_OUT_OF_DATE_KHR) VRED_ONCE("present: vkQueuePresentKHR %d", r);
    if ((!r || r == VK_SUBOPTIMAL_KHR) && !g_shown++) VLOG("presenting on the swapchain: frames are shown");   // not worded "first present" (lognorm.py --until-present cuts there)
}

// ---------- Screenshots (DM_NVN_SHOT: nvn_vk.c picks the presents and the path): the window texture as nvn_mtl.m's shot() writes it, so the two backends' frames diff directly ----------
// Level 0 of the whole texture at its own size (no crop, no scaling: DM_NVN_SHOT_OUT is not ported), rows flipped for NVN's lower-left origin (the image is stored bottom-up, the file top-down),
// written as an 8-bit RGB PNG with an sRGB chunk; the bytes as stored (an sRGB target's are not converted, nor are they by Metal's blit). Taken on this thread after the present's own work: the copy
// is queued behind every NVN submission made before this present, and the ticket is done only after it, so the game has not drawn its next frame into the texture yet (it gets the texture back only
// then: mtl_present_wait). Independent of the window and the swapchain. Only a shot present waits for the GPU (pres_wait); the others never come here
static VkCommandPool g_shot_pool; static VkCommandBuffer g_shot_cb; static VkBuffer g_shot_buf; static VkDeviceMemory g_shot_mem; static void *g_shot_map; static VkDeviceSize g_shot_cap;
static int g_shot_coh, g_shot_dead;   // g_shot_dead: a shot never finished on the GPU, so its buffer may still be written: no more shots (said in red)
static void shot_buf_free(void) {
    if (g_shot_map) vk.vkUnmapMemory(vd.dev, g_shot_mem);
    if (g_shot_buf) vk.vkDestroyBuffer(vd.dev, g_shot_buf, NULL);
    if (g_shot_mem) vk.vkFreeMemory(vd.dev, g_shot_mem, NULL);
    g_shot_map = NULL; g_shot_buf = VK_NULL_HANDLE; g_shot_mem = VK_NULL_HANDLE; g_shot_cap = 0;
}
static int shot_buf(VkDeviceSize n) {   // a host-visible readback buffer of n bytes at least (the last shot has completed: shot_one waits for each)
    if (g_shot_cap >= n) return 1;
    shot_buf_free();
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vk.vkCreateBuffer(vd.dev, &bi, NULL, &g_shot_buf)) { g_shot_buf = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements mr; vk.vkGetBufferMemoryRequirements(vd.dev, g_shot_buf, &mr);
    int mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);   // read by the CPU: cached where there is such a type
    if (mt < 0) mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &g_shot_mem)) { g_shot_mem = VK_NULL_HANDLE; shot_buf_free(); return 0; }
    if (vk.vkBindBufferMemory(vd.dev, g_shot_buf, g_shot_mem, 0) || vk.vkMapMemory(vd.dev, g_shot_mem, 0, VK_WHOLE_SIZE, 0, &g_shot_map)) { g_shot_map = NULL; shot_buf_free(); return 0; }
    g_shot_coh = (vd.mem.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0; g_shot_cap = n;
    return 1;
}
// PNG without zlib (the link lines stay as they are): stored deflate blocks, CRC-32 for the chunks, Adler-32 for the stream. 1 = written
static uint32_t g_crc_t[256];
static uint32_t crc32_of(uint32_t c, const uint8_t *p, size_t n) {
    if (!g_crc_t[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t v = i; for (int k = 0; k < 8; k++) v = v & 1 ? 0xedb88320u ^ (v >> 1) : v >> 1; g_crc_t[i] = v; }
    c = ~c; while (n--) c = g_crc_t[(c ^ *p++) & 0xff] ^ (c >> 8);
    return ~c;
}
static void be32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static int png_chunk(FILE *f, const char *type, const uint8_t *d, uint32_t n) {
    uint8_t h[8]; be32(h, n); memcpy(h + 4, type, 4);
    uint32_t c = crc32_of(crc32_of(0, h + 4, 4), d, n); uint8_t t[4]; be32(t, c);
    return fwrite(h, 1, 8, f) == 8 && (!n || fwrite(d, 1, n, f) == n) && fwrite(t, 1, 4, f) == 4;
}
static int png_write(const char *path, uint32_t w, uint32_t h, const uint8_t *raw) {   // raw: h scanlines of a filter byte (0) and 3 * w bytes of RGB
    size_t rn = (size_t)h * (1 + 3 * (size_t)w), nb = rn / 65535 + 1, zn = 2 + nb * 5 + rn + 4;
    if (zn > 0x7fffffffu) return 0;   // one IDAT chunk
    uint8_t *z = malloc(zn); if (!z) return 0;
    size_t o = 0; z[o++] = 0x78; z[o++] = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < rn; i++) { a += raw[i]; if (a >= 65521) a -= 65521; b += a; if (b >= 65521) b -= 65521; }
    for (size_t s = 0; s < rn || s == 0; ) {   // an empty image is one empty final block
        size_t k = rn - s > 65535 ? 65535 : rn - s; int last = s + k == rn;
        z[o++] = (uint8_t)last; z[o++] = (uint8_t)k; z[o++] = (uint8_t)(k >> 8); z[o++] = (uint8_t)~k; z[o++] = (uint8_t)(~k >> 8);
        memcpy(z + o, raw + s, k); o += k; s += k;
        if (last) break;
    }
    be32(z + o, b << 16 | a); o += 4;
    uint8_t ih[13]; be32(ih, w); be32(ih + 4, h); ih[8] = 8; ih[9] = 2; ih[10] = ih[11] = ih[12] = 0;   // 8-bit RGB, no interlace
    const uint8_t srgb = 0;   // perceptual: the colour space Metal's CGImage is tagged with
    FILE *f = fopen(path, "wb");
    int ok = f && fwrite("\x89PNG\r\n\x1a\n", 1, 8, f) == 8 && png_chunk(f, "IHDR", ih, 13) && png_chunk(f, "sRGB", &srgb, 1) && png_chunk(f, "IDAT", z, (uint32_t)o) && png_chunk(f, "IEND", NULL, 0);
    if (f && fclose(f)) ok = 0;
    free(z); return ok;
}
static void shot_one(const pq_ent *q) {
    if (g_shot_dead) return;   // said
    if (atomic_load(&vkn_lost)) { VRED_ONCE("screenshot %s: the GPU is lost to this run, no screenshots", q->shot); return; }
    if (!g_shot_pool) {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, vd.qfam};
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vk.vkCreateCommandPool(vd.dev, &pi, NULL, &g_shot_pool) || (ai.commandPool = g_shot_pool, vk.vkAllocateCommandBuffers(vd.dev, &ai, &g_shot_cb))) {
            g_shot_dead = 1; VRED_ONCE("screenshot %s: the readback command buffer cannot be made: no screenshots", q->shot); return; }
    }
    if (!shot_buf((VkDeviceSize)q->w * q->h * 4)) { VRED_ONCE("screenshot %s: no host-visible readback buffer of %ux%u: not taken (the next ones try again)", q->shot, q->w, q->h); return; }
    pthread_mutex_lock(&vkn_qmu); pthread_mutex_lock(&vkn_tmu);   // the layout read here is the queue's (see present_one), and the image stays until its use_pres has completed
    vkn_img *e = vkn_img_get(q->serial); VkImageLayout L = e ? e->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    VkFormatProperties fp = {0}; if (e) vk.vkGetPhysicalDeviceFormatProperties(vd.pd, e->fmt, &fp);
    int bgra = e && (e->fmt == VK_FORMAT_B8G8R8A8_UNORM || e->fmt == VK_FORMAT_B8G8R8A8_SRGB), rgba = e && (e->fmt == VK_FORMAT_R8G8B8A8_UNORM || e->fmt == VK_FORMAT_R8G8B8A8_SRGB), a2b10 = e && e->fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32, b10 = e && e->fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32, dep = e && (e->fmt == VK_FORMAT_D24_UNORM_S8_UINT || e->fmt == VK_FORMAT_X8_D24_UNORM_PACK32 || e->fmt == VK_FORMAT_D32_SFLOAT);   // dep: a depth target (DM_VK_SHOT_SER), its depth as grey   // a2b10: render targets taken with DM_VK_SHOT_SER, top 8 bits of each channel
    const char *no = !e ? "the window texture has no host image" : L == VK_IMAGE_LAYOUT_UNDEFINED ? "the window texture was never written" : !bgra && !rgba && !a2b10 && !b10 && !dep ? "its format is not RGBA8 / BGRA8 / A2B10G10R10 / B10G11R11 / D24 / D32"
                   : !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ? "its format cannot be copied from" : (VkDeviceSize)e->w * e->h * 4 > g_shot_cap ? "it is larger than the present said" : NULL;
    uint32_t w = e ? e->w : 0, h = e ? e->h : 0; VkFormat fmt = e ? e->fmt : VK_FORMAT_UNDEFINED; VkResult r = VK_SUCCESS; uint64_t val = 0;
    if (!no) {
        VkCommandBuffer cb = g_shot_cb;
        vk.vkResetCommandPool(vd.dev, g_shot_pool, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vk.vkBeginCommandBuffer(cb, &bi);
        VkImageAspectFlags ax = dep ? e->aspect : VK_IMAGE_ASPECT_COLOR_BIT;
        img_barrier(cb, e->img, ax, L, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy rg = {0, 0, 0, {dep ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
        vk.vkCmdCopyImageToBuffer(cb, e->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_shot_buf, 1, &rg);
        img_barrier(cb, e->img, ax, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, L, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);   // back as the next NVN submission expects it
        VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &hb, 0, NULL, 0, NULL};
        vk.vkCmdPipelineBarrier2(cb, &di);
        vk.vkEndCommandBuffer(cb);
        val = atomic_load(&g_pval) + 1; e->use_pres = val;
        if (!(r = submit_one(cb, VK_NULL_HANDLE, VK_NULL_HANDLE, val))) atomic_store(&g_pval, val);   // failed: the GPU lost (said there)
    }
    pthread_mutex_unlock(&vkn_tmu); pthread_mutex_unlock(&vkn_qmu);
    if (no) { VLOG("[RED] screenshot %s: not taken: %s", q->shot, no); return; }
    if (r) return;
    if (!pres_wait(val)) { g_shot_dead = 1; VRED_ONCE("screenshot %s: the readback has not finished on the GPU: no more screenshots (its buffer may still be written)", q->shot); return; }
    VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, g_shot_mem, 0, VK_WHOLE_SIZE};
    if (!g_shot_coh && (r = vk.vkInvalidateMappedMemoryRanges(vd.dev, 1, &mr))) { VLOG("[RED] screenshot %s: not taken: the readback memory could not be invalidated (VkResult %d)", q->shot, r); return; }
    size_t row = 1 + 3 * (size_t)w; uint8_t *raw = malloc(row * h ? row * h : 1); int ll = vkn_origin_ll(), ok = 0;
    if (raw) {
        const uint8_t *src = g_shot_map;
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t *s = src + (size_t)(ll ? h - 1 - y : y) * w * 4; uint8_t *d = raw + y * row; *d++ = 0;
            if (dep) {   // depth as grey: the range of this image's values (said in the log) stretched to 0-255, depth 1.0 (cleared) black
                static float lo, hi; if (y == 0) { lo = 2.0f; hi = -1.0f; const uint8_t *qq = src; for (size_t i = 0; i < (size_t)w * h; i++, qq += 4) { float f; uint32_t u; memcpy(&u, qq, 4);
                    if (fmt == VK_FORMAT_D32_SFLOAT) memcpy(&f, &u, 4); else f = (float)(u & 0xffffff) / 16777215.0f; if (f < 1.0f) { if (f < lo) lo = f; if (f > hi) hi = f; } }
                    VLOG("screenshot %s: depth below 1.0 spans %.6f-%.6f", q->shot, lo, hi); }
                for (uint32_t x = 0; x < w; x++, s += 4) { float f; uint32_t u; memcpy(&u, s, 4); if (fmt == VK_FORMAT_D32_SFLOAT) memcpy(&f, &u, 4); else f = (float)(u & 0xffffff) / 16777215.0f;
                    uint8_t g = f >= 1.0f || hi <= lo ? 0 : (uint8_t)(255.0f - 255.0f * (f - lo) / (hi - lo)); d[0] = d[1] = d[2] = g; d += 3; } }
            else if (b10) for (uint32_t x = 0; x < w; x++, s += 4) {   // B10G11R11 float: each channel's mantissa bits and exponent, clamped to 1.0, as 8 bits
                uint32_t v; memcpy(&v, s, 4); uint32_t c[3] = {v & 0x7ff, (v >> 11) & 0x7ff, (v >> 22) & 0x3ff}; int mb[3] = {6, 6, 5};
                for (int k = 0; k < 3; k++) { uint32_t ex = c[k] >> mb[k], m = c[k] & ((1u << mb[k]) - 1); float f = ex ? ldexpf(1.0f + (float)m / (float)(1u << mb[k]), (int)ex - 15) : ldexpf((float)m / (float)(1u << mb[k]), -14);
                    d[k] = (uint8_t)(f >= 1.0f ? 255 : f * 255.0f); }
                d += 3; }
            else if (a2b10) for (uint32_t x = 0; x < w; x++, s += 4) { uint32_t v; memcpy(&v, s, 4); d[0] = (uint8_t)((v >> 2) & 0xff); d[1] = (uint8_t)((v >> 12) & 0xff); d[2] = (uint8_t)((v >> 22) & 0xff); d += 3; }
            else for (uint32_t x = 0; x < w; x++, s += 4) { d[0] = s[bgra ? 2 : 0]; d[1] = s[1]; d[2] = s[bgra ? 0 : 2]; d += 3; }
        }
        ok = png_write(q->shot, w, h, raw); free(raw);
    }
    VLOG("%sscreenshot %ux%u format %d%s → %s %s", ok ? "" : "[RED] ", w, h, fmt, ll ? " (rows flipped: lower-left origin)" : "", q->shot, ok ? "ok" : "failed (memory, or the file cannot be written)");
}

static void *present_main(void *arg) {
    (void)arg;
    pthread_setname_np(
#ifdef __APPLE__
        "vkn-present");
#else
        pthread_self(), "vkn-present");
#endif
    int ok = objs_make();
    if (!ok) VRED_ONCE("present: the present objects could not be made: nothing is shown");
    else { const char *t = getenv("DM_VK_PRESENT_TEST"); if (t && atoi(t) > 0 && !test_make()) VRED_ONCE("DM_VK_PRESENT_TEST=1: the pattern could not be made: the game's frames are shown"); }
    pthread_mutex_lock(&g_pmu);
    if (!ok) { g_dead = 1; g_lost_req = 0; g_pq_done = g_pq_in; pthread_cond_broadcast(&g_pcv); pthread_mutex_unlock(&g_pmu); return NULL; }   // a waiting window loss returns (no surface was made)
    for (;;) {
        while (!g_lost_req && g_pq_read == g_pq_in) pthread_cond_wait(&g_pcv, &g_pmu);
        if (g_lost_req) {   // a new window starts afresh: its surface and swapchain lines are listed
            pthread_mutex_unlock(&g_pmu); surface_free("the front end's window is going"); pthread_mutex_lock(&g_pmu);
            g_lost_req = 0; g_going = 0; g_bad_run = 0; pthread_cond_broadcast(&g_pcv);
            if (g_wedged) break;
            continue;
        }
        pq_ent q = g_pq[g_pq_read % PQ]; g_pq_read++;
        pthread_mutex_unlock(&g_pmu);
        present_one(&q);
        if (q.shot[0] && !g_wedged) shot_one(&q);   // before the ticket is done (see shot_one); a wedged queue cannot be waited on
        pthread_mutex_lock(&g_pmu); g_pq_done = q.ticket; pthread_cond_broadcast(&g_pcv);
        if (g_wedged) break;
    }
    // Wedged (queue_idle, said in red): the thread ends as one whose objects could not be made does. Every ticket is retired at once from here, and a window loss does not wait (this
    // thread's window reference went in wedge; the surface and swapchain stay, leaked, and nothing draws into the window again)
    g_dead = 1; g_lost_req = 0; g_pq_done = g_pq_in; pthread_cond_broadcast(&g_pcv); pthread_mutex_unlock(&g_pmu);
    return NULL;
}

uint64_t vkn_present_enqueue(uint64_t serial, uint32_t w, uint32_t h, const int crop[4], int log, const char *shot) {
    if (!vd.dev) return 0;
    pthread_mutex_lock(&g_pmu);
    if (!g_thread_on) {
        pthread_t t; if (pthread_create(&t, NULL, present_main, NULL)) { pthread_mutex_unlock(&g_pmu); VRED_ONCE("present: the present thread cannot start"); return 0; }
        pthread_detach(t); g_thread_on = 1;
    }
    while (!g_dead && g_pq_in - g_pq_done >= PQ) pthread_cond_wait(&g_pcv, &g_pmu);   // a full queue holds the game here, as nvn_mtl.m's
    if (g_dead) { pthread_mutex_unlock(&g_pmu); if (shot) VRED_ONCE("screenshot %s: the present thread is not running: no screenshots", shot); return 0; }   // after the wait too: the present thread can end while a producer waits, and nothing would retire its ticket
    uint64_t tk = ++g_pq_in; pq_ent *e = &g_pq[(tk - 1) % PQ];   // the present thread reads entry g_pq_read (= tk - 1) next
    e->ticket = tk; e->serial = serial; e->w = w; e->h = h; if (crop) memcpy(e->crop, crop, sizeof e->crop); else memset(e->crop, 0, sizeof e->crop);
    snprintf(e->shot, sizeof e->shot, "%s", shot ? shot : "");
    if (log) atomic_store(&g_plog, 1);
    pthread_cond_broadcast(&g_pcv); pthread_mutex_unlock(&g_pmu);
    return tk;
}
void vkn_present_wait(uint64_t ticket) { pthread_mutex_lock(&g_pmu); while (g_pq_done < ticket && !g_dead) pthread_cond_wait(&g_pcv, &g_pmu); pthread_mutex_unlock(&g_pmu); }
void vkn_present_flush(void) { pthread_mutex_lock(&g_pmu); uint64_t t = g_pq_in; pthread_mutex_unlock(&g_pmu); vkn_present_wait(t); pres_wait(atomic_load(&g_pval)); }
// vkn_present_flush within ns (nvn_vk.c mtl_drain's bound): 1 = every ticket so far retired and the GPU through their present work (or the GPU lost: nothing more of it runs)
int vkn_present_drain(uint64_t ns) {
    struct timespec t0, dl; clock_gettime(CLOCK_MONOTONIC, &t0); clock_gettime(CLOCK_REALTIME, &dl);   // g_pcv waits on the realtime clock
    uint64_t e = (uint64_t)dl.tv_nsec + ns; dl.tv_sec += (time_t)(e / 1000000000ull); dl.tv_nsec = (long)(e % 1000000000ull);
    int ok = 1;
    pthread_mutex_lock(&g_pmu); uint64_t t = g_pq_in;
    while (ok && g_pq_done < t && !g_dead) if (pthread_cond_timedwait(&g_pcv, &g_pmu, &dl) == ETIMEDOUT) ok = g_pq_done >= t || g_dead;
    pthread_mutex_unlock(&g_pmu);
    uint64_t v = atomic_load(&g_pval);
    VkSemaphore tl = ptl();
    if (!ok || !tl || !v) return ok;
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t used = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)t1.tv_nsec - (uint64_t)t0.tv_nsec;
    if (used >= ns) return 0;
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &tl, &v};
    VkResult r = vk.vkWaitSemaphores(vd.dev, &wi, ns - used);
    if (r != VK_SUCCESS && r != VK_TIMEOUT) vkn_gpu_lost("the present's vkWaitSemaphores", r);
    return r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST;   // another error (out of memory) is no answer: not drained (mtl_drain says it)
}

// ---------- The front end's window hooks (front_android.c's weak ones are the null build's) ----------
int dm_gfx_owns_window(void) { return 1; }   // the swapchain draws the window: the front end must not lock it for a CPU fill, or the swapchain cannot connect
// The window is going (onNativeWindowDestroyed, on the main thread): returns once the surface and swapchain made on it are gone, which takes one present at most (the acquire has a timeout).
// Bounded at 4 s all the same, in red: a GPU that hangs a present must not hang the main thread into an ANR (the surface keeps its own window reference until the present thread lets it go).
// g_lost_req stays set until the present thread has let the surface go, so after a timeout too nothing new is started on the window (present_one: win_going)
void dm_gfx_window_lost(void) {
    pthread_mutex_lock(&g_pmu);
    if (g_thread_on && !g_dead) {
        g_lost_req = 1; pthread_cond_broadcast(&g_pcv);
        struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_sec += 4;   // g_pcv waits on the realtime clock (PTHREAD_COND_INITIALIZER)
        while (g_lost_req && !g_dead) if (pthread_cond_timedwait(&g_pcv, &g_pmu, &dl) == ETIMEDOUT) { VLOG("[RED] window loss: the present thread has not let the surface go in 4 s; the window goes regardless"); break; }
    }
    pthread_mutex_unlock(&g_pmu);
}
