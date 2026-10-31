// Vulkan GPU backend, first slice (docs/android/PLAN.md Phase 2): the mtl_* set hle_nvn.c calls, begun as a copy of the null backend (dm/gfx/null/nvn_null.c), whose symbol list it keeps.
// Real here: the device (vk_dev.c), NVN render targets as VkImages in plain device memory (PLAN.md 2.2: render-target pools have no shadow), SET_RT, colour and depth-stencil clears (folded
// into the dynamic-rendering load ops when they come before the pass's first draw, as nvn_mtl.m's rp_begin_ex_ does; vkCmdClearAttachments after it), barriers, submission with one timeline
// value per submit and a completion thread that performs NVN's completion effects (event values, counter reports, then the done fence: nvn_mtl.m's completion handlers), and the present of the
// window texture (vk_present.c), and memory strategy B (PLAN.md 2.2 "As built, slice 2a"): the pool pages the GPU reads as buffers are copied into shadow VkBuffers at each submission, kept current
// by nvn_dirty.c's write protection, and sampled textures and samplers (PLAN.md 2.3 "As built, slice 2b"): the textures the replay's binds name get host images, uploaded from guest memory (CPU
// deswizzle into the same staging, copied by the same upload command buffer) and uploaded again when the guest writes them, and the samplers they name get VkSamplers, and draws (PLAN.md 2.5
// "As built, slice 2c"): indexed draws with slice 0's SPIR-V modules, one push-descriptor set and monolithic pipelines; other draw kinds are said in red and skipped. As the null backend: buffer
// copies / fills / uniform updates run on the CPU at replay, texture reads come from guest memory. The ordering convention of nvn_mtl.h holds: a submission's event
// values and counter records are written before its fence counts as done, and waits look only at that.
// No Vulkan device (no library, no 1.3 driver): everything runs as the null backend does, said in red
#include "vkn.h"
#include "vk_desc.h"   // descriptor delivery: push descriptors or per-slot pools (Mali)
#include "../nvn/nvn_obj.h"
#include "../nvn/nvn_mtl.h"
#include "../nvn/nvn_tex.h"
#include "../nvn/nvn_dirty.h"
#include "../nvn/nvn_hmap.h"
#include "nvn_fmt_vk.h"
#include "../tex/nvn_deswizzle.h"
#include "../../host/dm.h"
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define NLOG VLOG
#define NRED_ONCE VRED_ONCE
int nvn_game_fps(void);   // hle_nvn.c
static int g_vk;          // the device is up (mtl_init); 0 = the null backend's behaviour
static _Atomic int g_origin_ll = 1;   // NVN's default window origin, lower left (hle_nvn.c sets it at device init and on SetWindowOriginMode)
int vkn_origin_ll(void) { return atomic_load_explicit(&g_origin_ll, memory_order_relaxed); }

// ---------- Surface statistics: what the game asks of a GPU backend (the Vulkan backend's work list, PLAN.md Phase 2), counted where the records are walked anyway ----------
// Kept from the null backend. Cumulative; one [vkn] line at presents 60, 600 and every 1800, every DM_NVN_NULL_EVERY=<n> presents as well (to tell a run's phases apart: a phase is the difference of two lines),
// and at mtl_drain (graceful exit, or the app sent to the background). Never before present 60 (STATS_FROM, DM_NVN_NULL_EVERY and mtl_drain too): the parity cut ends at the first present
enum { S_DRAW_INST = OP_COUNT_, S_DRAW_BINST, S_DRAW_IND, S_DRAW_MDI, S_DRAW_VTX, S_DIDX_INST, S_DIDX_BVTX, S_DIDX_BINST, S_DIDX_IND, S_DIDX_MDI, S_DIDX_IDX,
       S_PRIM, S_ITYPE = S_PRIM + 16, S_CLR_START = S_ITYPE + 3, S_CLR_MID, S_CLR_PART, S_DS_START, S_DS_MID, S_DS_PART,
       S_B2B_B, S_FILL_B, S_UB_B, S_B2T_B, S_T2B_B, S_T2T_RES, S_DISP_IND, S_CNT_TS, S_CNT_SP, S_CNT_OTH, S_TF_WHAT, S_RC_MODE = S_TF_WHAT + 4, S_BAR = S_RC_MODE + 4,
       S_RT_NC = S_BAR + 8, S_RT_DEPTH = S_RT_NC + 9, S_RT_ZONLY, S_RT_MSAA, S_RTX, S_RTX_B, S_RTX_GPU, S_GW_B, S_VPSW, S_STATE, S_N = S_STATE + 6 };
static _Atomic uint64_t g_st[S_N];
#define ST(i, n) atomic_fetch_add_explicit(&g_st[i], (uint64_t)(n), memory_order_relaxed)
static const char *const g_opname[OP_COUNT_] = {[OP_END] = "END", [OP_JUMP] = "JUMP", [OP_BEGIN] = "BEGIN", [OP_CALL] = "CALL", [OP_SIGNAL_EVENT] = "SIGNAL_EVENT", [OP_REPORT_COUNTER] = "REPORT_COUNTER",
    [OP_COPY_B2B] = "COPY_B2B", [OP_UPDATE_UB] = "UPDATE_UB", [OP_SET_RT] = "SET_RT", [OP_CLEAR_COLOR] = "CLEAR_COLOR", [OP_CLEAR_DS] = "CLEAR_DS", [OP_VIEWPORTS] = "VIEWPORTS", [OP_SCISSOR] = "SCISSOR",
    [OP_DEPTH_RANGE] = "DEPTH_RANGE", [OP_BIND_PROGRAM] = "BIND_PROGRAM", [OP_BIND_STATE] = "BIND_STATE", [OP_BIND_VATTR] = "BIND_VATTR", [OP_BIND_VSTREAM] = "BIND_VSTREAM", [OP_BIND_VBUF] = "BIND_VBUF",
    [OP_BIND_UBO] = "BIND_UBO", [OP_BIND_SSBO] = "BIND_SSBO", [OP_BIND_TEX] = "BIND_TEX", [OP_BIND_SAMP] = "BIND_SAMP", [OP_STENCIL] = "STENCIL", [OP_POLY_OFFSET] = "POLY_OFFSET", [OP_DRAW] = "DRAW",
    [OP_DRAW_IDX] = "DRAW_IDX", [OP_DISPATCH] = "DISPATCH", [OP_BARRIER] = "BARRIER", [OP_SET_TEX_POOL] = "SET_TEX_POOL", [OP_SET_SAMP_POOL] = "SET_SAMP_POOL", [OP_COPY_B2T] = "COPY_B2T",
    [OP_COPY_STRIDE] = "COPY_STRIDE", [OP_BLEND_COLOR] = "BLEND_COLOR", [OP_COPY_T2B] = "COPY_T2B", [OP_BIND_IMG] = "BIND_IMG", [OP_COPY_T2T] = "COPY_T2T", [OP_RESET_COUNTER] = "RESET_COUNTER",
    [OP_CLEAR_TEX] = "CLEAR_TEX", [OP_CLEAR_BUF] = "CLEAR_BUF", [OP_DEPTH_CLAMP] = "DEPTH_CLAMP", [OP_RENDER_COND] = "RENDER_COND", [OP_FENCE_SYNC] = "FENCE_SYNC", [OP_TF_BIND] = "TF_BIND",
    [OP_TF_STATE] = "TF_STATE", [OP_RAST_DISCARD] = "RAST_DISCARD", [OP_VP_SWIZZLE] = "VP_SWIZZLE", [OP_BIND_TEX1] = "BIND_TEX1", [OP_PRIM_RESTART] = "PRIM_RESTART"};
// Formats seen, as NVN enums (all below 256): 0 render targets, 1 registered textures (and their views), 2 vertex attributes; texture targets as bits
static _Atomic uint64_t g_fmt[3][4]; static _Atomic uint32_t g_tgt, g_vpsw;   // g_vpsw: viewport swizzles other than the identity, bit 8 * component + NVNviewportSwizzle
static void fmt_note(int k, unsigned f) { if (f < 256) atomic_fetch_or_explicit(&g_fmt[k][f >> 6], 1ull << (f & 63), memory_order_relaxed); }
// Live texture serials (TextureInitialize until Finalize or re-initialization: mtl_tex_init / mtl_tex_release), each with whether the GPU writes it (render targets, copy and clear destinations:
// a ReadTexels of one is a GPU → CPU readback in a real backend, of any other the CPU's own bytes) and whether the CPU does (mtl_tex_cpu_written: WriteTexels, FlushTexels, packaged data: its
// content is uploaded from guest memory where it is sampled). Value bits: 1 live, 2 the GPU writes it, 4 the CPU does. Records snapshot the texture object (hle_nvn.c tex_snap), so the
// replay meets serials already finalized: those are not added, nothing would take them out again (serials are never reused, nvn_new_serial). A leaf lock (taken under vkn_tmu too)
static nvn_hmap g_tser = {.klen = 8}; static pthread_mutex_t g_tsmu = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t tser_get(uint64_t s) { pthread_mutex_lock(&g_tsmu); uintptr_t v = s ? (uintptr_t)nvn_hm_get(&g_tser, &s) : 0; pthread_mutex_unlock(&g_tsmu); return v; }
static int tex_live(uint64_t s) { return tser_get(s) != 0; }
static void gpu_wrote(const tex_t *t) {
    static uint64_t last; if (!t || !t->serial || t->serial == last) return;   // submit thread only; the same few targets come back every pass
    uint64_t s = t->serial;
    pthread_mutex_lock(&g_tsmu); uintptr_t v = (uintptr_t)nvn_hm_get(&g_tser, &s); if (v && !(v & 2)) nvn_hm_put(&g_tser, &s, (void *)(v | 2)); pthread_mutex_unlock(&g_tsmu);
    last = s;
}
static int gpu_written(uint64_t s) { return (tser_get(s) & 2) != 0; }
static uint64_t region_bytes(const tex_t *t, const NVNcopyRegion *r) { int bw, bh, bpb = t ? nvn_fmt_bpb(t->d.format, &bw, &bh) : 0; return bpb ? (uint64_t)((r->width + bw - 1) / bw) * (uint64_t)((r->height + bh - 1) / bh) * (uint64_t)(r->depth > 1 ? r->depth : 1) * (uint64_t)bpb : 0; }
static _Atomic uint64_t g_present_n;   // presents so far = hle_nvn.c's frame index (init = 0), the frame things below are labelled with
static uint64_t g_every;               // DM_NVN_NULL_EVERY
static void shot_parse(void);          // DM_NVN_SHOT (the present section)
static uintptr_t g_pgsz, g_sh;              // the host page (4 KB on Android, 16 KB on the Mac) and its shift: nvn_dirty.c protects by it, so B tracks and uploads by it (vkn_hello)

// ---------- Submission state (below) that mtl_init makes ----------
// Submission slots, used round-robin: each its own command pool, and the completion effects (post ops) of the submission it holds until that completes
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // kind 1 event, 2 counter report
#define NSUB 4
// up: B's uploads into the shadows, submitted ahead of cb (b_commit); stg: the slot's staging buffer for them, host-visible and mapped, reused once the slot's last submission has completed
// ring: the uniform bytes of bindings at an address the device cannot bind (an offset not a multiple of minUniformBufferOffsetAlignment), copied at the draw (draw_ubo), host-visible and coherent
typedef struct { VkCommandPool pool; VkCommandBuffer cb, up; uint64_t value; int failed; post_op *post; int npost, postcap; VkBuffer stg; VkDeviceMemory stg_mem; void *stg_map; uint64_t stg_cap; int stg_coh;
                 VkBuffer ring; VkDeviceMemory ring_mem; uint8_t *ring_map; uint64_t ring_used; int ring_no;
                 struct { VkBuffer b; VkDeviceMemory m; } *tmp; int ntmp, tmpcap; } sub_slot;   // tmp: host buffers of this submission's CPU-decoded copies (op_copy_b2t), freed when the slot comes round again
static sub_slot g_sub[NSUB];
static sub_slot *g_cur;                    // the submission being replayed (submit thread: hle_nvn.c serializes submissions, g_sub_mu)
static uint64_t g_nbegun, g_ncommit, g_ncomp;   // submissions begun (submit thread), handed to the completion thread and completed (both under g_mu)
static VkSemaphore g_tl;                   // the NVN timeline: submission k signals k (= its fence)
static _Atomic uint64_t g_fence, g_done;   // g_fence: last committed; g_done: completed, effects written (g_mu / g_cv for waiters)
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER, g_ccv = PTHREAD_COND_INITIALIZER;
static void *comp_main(void *arg);
static _Atomic uint64_t g_stat_pass, g_stat_fold, g_stat_clr_mid, g_stat_bar, g_stat_own;   // passes begun, clears folded into load ops / made with vkCmdClearAttachments, barriers, oversize clears in a pass of their own
// A submission retired without the GPU's answer (comp_main: a lost GPU whose wait keeps timing out, or a wait that failed other than with DEVICE_LOST): the GPU may still use what it named, so from then on nothing is destroyed (vkn_sweep)
// and no command pool is reset (mtl_sub_begin: every later submission is failed). A leak after a broken GPU, never a use after free
static _Atomic int g_leak;

static void state_reset(void); static void draw_init(void);
static void vkn_hello(void) {
    NLOG("Vulkan GPU backend (dm/gfx/vkn): render targets, clears, barriers, submission and present are real, buffer memory the GPU reads is kept in shadow buffers (strategy B), bound textures and samplers get host images (uploaded from guest memory) and VkSamplers, and draws are drawn with slice 0's SPIR-V shaders (DM_VK_SHADERS); buffer copies, fills and uniform updates run on the CPU; texture reads come from guest memory");
    state_reset();
    const char *e = getenv("DM_NVN_NULL_EVERY"); g_every = e ? strtoull(e, NULL, 10) : 0;   // the null backend's surface lines, every <n> presents too
    shot_parse();
    g_pgsz = (uintptr_t)getpagesize(); g_sh = (uintptr_t)__builtin_ctzl(g_pgsz);
    if (!vkn_dev_init()) { NLOG("[RED] no Vulkan device: running as the null backend does, nothing is drawn or shown"); return; }
    VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    if (vk.vkCreateSemaphore(vd.dev, &si, NULL, &g_tl)) { NLOG("[RED] the timeline semaphore cannot be made: running as the null backend does"); return; }
    for (int i = 0; i < NSUB; i++) {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, vd.qfam};
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2}; VkCommandBuffer two[2];
        if (vk.vkCreateCommandPool(vd.dev, &pi, NULL, &g_sub[i].pool) || (ai.commandPool = g_sub[i].pool, vk.vkAllocateCommandBuffers(vd.dev, &ai, two))) {
            NLOG("[RED] the submission command pools cannot be made: running as the null backend does"); return; }
        g_sub[i].cb = two[0]; g_sub[i].up = two[1];
    }
    pthread_t t; if (pthread_create(&t, NULL, comp_main, NULL)) { NLOG("[RED] the completion thread cannot start: running as the null backend does"); return; }
    pthread_detach(t);
    draw_init();   // the draws' dummies (none: nothing is drawn, said)
    g_vk = 1;
}
void mtl_init(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, vkn_hello); }

// ---------- Memory pools and strategy B (PLAN.md 2.2 "As built, slice 2a"): the address ranges, and a shadow VkBuffer for each pool the GPU reads as buffers ----------
// Begun as the null backend's B simulation (nvn_null.c DM_NVN_NULL_BSIM), made real. Each pool has a page-class table. A replayed record that reads guest memory as a buffer (vertex, index, uniform,
// storage, indirect arguments, a copy's source) marks those pages PF_BUF and gives the pool its shadow: one VkBuffer over the pool's page-rounded range, device-local where offered (b_mark). A record's own
// pages are PF_CMD: command memory, which the game rewrites all the time, is never protected. At each commit (b_commit) the PF_BUF pages written since the last one (nvn_dirty.c's write protection
// sees them), read for the first time (PF_NEW), or never protected (PF_CMD, PF_ALW) go through the slot's staging buffer into the shadows, by a command buffer that runs ahead of the submission's own.
// mtl_pool_add gets no pool flags, so pools no record reads as a buffer get no shadow and nothing is protected in them: render-target pools (CPU_NO_ACCESS, which their textures tell mtl_tex_init:
// a buffer read there is said in red) and shader-code pools (hle_nvn.c reaches those through mtl_addr only, which makes nothing)
enum { PF_BUF = 1, PF_CMD = 4, PF_NEW = 8, PF_ALW = 16, PF_DUE = 32, PF_HOT = 64 };   // ALW: copied at every submission, never protected (see b_commit); DUE: picked by the commit under way; HOT: see b_hot
// The hot set (DM_VK_HOT=1: on; off by default, see the end of this comment). A page written again within HOT_WIN commits of its last write is left unprotected and copied at every commit for HOT_TTL commits, then protected again
// (it heats up again at its next writes; a frame sent in parts, DM_VK_SPLIT, has several commits). Copying a page that did not change is always right; a write fault and the mprotect calls around it cost far more than copying the page: Paper Mario: The Thousand-Year
// Door writes about 180 of its 4 KB buffer pages every frame (uniform rings, dynamic vertices), and on the RG 476H the faults, the mprotect calls and the commit's drain of them took
// several ms of the game thread a frame. Off by default: on the RG 476H it gained nothing (the faults are on guest threads, which are not the limit; submit+present rose 9.8 -> 11.0 ms)
// and one of two runs hung at a loading curtain. A hot page is copied even in a commit the CPU did not write it in, which can overwrite what a GPU pass wrote into the shadow page earlier
// in the frame; a sound version copies a hot page only when its bytes changed (keep the last copy, compare)
#define HOT_WIN 4
#define HOT_TTL 120
static int b_hot(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_HOT"); on = e && *e == '1'; if (on) NLOG("DM_VK_HOT=1: buffer pages written again within %d submissions are copied at every one, unprotected (the hot set)", HOT_WIN); } return on; }
typedef struct b_shadow { VkBuffer buf; VkDeviceMemory mem; VkDeviceAddress gpu; uint64_t size, use_nvn; struct b_shadow *next; } b_shadow;   // use_nvn: the last submission that used it (vkn_sweep)
// lo, hi: page-rounded, the shadow's extent (offsets in it are from lo); key, end: the pool's own bytes, the only ones an address resolves to (mtl_addr, mtl_gpu_addr: a page it shares with a
// neighbour holds the neighbour's bytes too, and this pool's shadow never has those uploaded); span: its nvn_dirty range, cached before any protect (mtl_pool_del keys on it);
// noacc: a CPU_NO_ACCESS pool (mtl_tex_init); nosh: its shadow could not be made (said once, not tried again)
typedef struct { uintptr_t lo, hi; const void *key; uintptr_t end; uint8_t *pf, *ht; const void *span; b_shadow *sh; uint8_t noacc, nosh; } pool_ent;   // ht: per page, 1 + commits since its last write (0: none within HOT_WIN), or a hot page's age (b_commit)
static pool_ent *g_pools; static int g_npool, g_cap, g_plast = -1;
// The pools by address (under g_pmu): their indices sorted by lo, rebuilt when a pool comes or goes. Pools hold disjoint bytes, so in this order their page-rounded ends rise too, and the
// first pool that can hold an address is found by a binary search on hi. The replay looked a range up by walking every pool, for every record (b_mark, b_use): Paper Mario: The
// Thousand-Year Door's in-game frames spent 18% of the process's CPU there on the RG 476H. DM_VK_POOL_INDEX=0: the linear walks, as before
static int *g_psort, g_psort_cap, g_psort_dirty = 1;
static int psort_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_POOL_INDEX"); on = !(e && *e == '0'); } return on; }
static int psort_cmp(const void *a, const void *b);
static void psort_build(void) {
    if (!g_psort_dirty) return;
    if (g_npool > g_psort_cap) { g_psort_cap = g_npool * 2 + 16; g_psort = realloc(g_psort, sizeof *g_psort * (size_t)g_psort_cap); if (!g_psort) die("vkn backend: pool index allocation failed"); }
    for (int k = 0; k < g_npool; k++) g_psort[k] = k;
    qsort(g_psort, (size_t)g_npool, sizeof *g_psort, psort_cmp);
    g_psort_dirty = 0;
}
static int psort_first(uint64_t addr) {   // the position of the first pool (in address order) whose page-rounded end lies past addr
    psort_build();
    int lo = 0, hi = g_npool;
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pools[g_psort[m]].hi > addr) hi = m; else lo = m + 1; }
    return lo;
}
// The pool whose own bytes hold [addr, addr+len) (in_pool), with a shadow when need_sh; -1 for none
static int in_pool(int k, uint64_t addr, uint64_t len);
static int pool_find(uint64_t addr, uint64_t len, int need_sh) {
    if (!psort_on()) { for (int k = 0; k < g_npool; k++) if ((!need_sh || g_pools[k].sh) && in_pool(k, addr, len)) return k; return -1; }
    for (int p = psort_first(addr); p < g_npool && g_pools[g_psort[p]].lo <= addr; p++) { int k = g_psort[p]; if ((!need_sh || g_pools[k].sh) && in_pool(k, addr, len)) return k; }
    return -1;
}
static pthread_mutex_t g_pmu = PTHREAD_MUTEX_INITIALIZER;
// Every pool made and freed, with its size and frame (the surface lines list each once); under g_pmu
typedef struct { uintptr_t mem; uint64_t size, born, died, id; uint8_t said; } pool_log;   // died ~0 = live; id: the order made (the number the lines show); said: 1 its creation listed, 2 its end too
static pool_log *g_plog; static int g_nplog, g_plogcap;
static uint64_t g_pmade, g_pmade_b, g_pfreed, g_pfreed_b, g_plive_b, g_ppeak_b;
static uint64_t g_nshadow, g_shadow_b, g_shadow_made;   // shadows in the table and their bytes, shadows made (g_pmu)
static _Atomic uint64_t g_shadow_freed;     // destroyed by vkn_sweep
static b_shadow *g_dead_b;                  // shadows of freed pools waiting for their last submission (under vkn_tmu, vkn_sweep)
void mtl_pool_add(const void *mem, uint64_t size) {
    mtl_init();
    uintptr_t pg = (uintptr_t)getpagesize(), lo = (uintptr_t)mem, hi = lo + size;   // the shadow spans whole host pages (lo, hi), as the Metal layer's zero-copy buffers do
    pthread_mutex_lock(&g_pmu);
    if (g_npool == g_cap) { g_cap = g_cap ? g_cap * 2 : 64; g_pools = realloc(g_pools, (size_t)g_cap * sizeof *g_pools); if (!g_pools) die("vkn backend: pool table allocation failed"); }
    pool_ent e = {lo & ~(pg - 1), (hi + pg - 1) & ~(pg - 1), mem, hi, NULL, NULL, NULL, 0, 0};
    uintptr_t np = (e.hi - e.lo) / pg;
    if (g_vk && (!(e.pf = calloc(np ? np : 1, 1)) || !(e.ht = calloc(np ? np : 1, 1)))) die("vkn backend: page class table allocation failed (%llu pages)", (unsigned long long)np);   // no device: no shadows, nothing tracked
    // Edge pages, those the pool's own bytes do not fill (16 KB host pages over 4 KB-aligned pools): a neighbour's nvn_dirty range can hold the same page, and two ranges protecting one page
    // break the tracker (a write unprotected and stamped through one leaves the other's state and generations stale). So no pool protects them: copied at every submission (PF_ALW).
    // With 4 KB host pages and 4 KB-aligned pools there are none. Slice 2b: texture protection must leave these pages alone too
    if (e.pf && np) { if (lo & (pg - 1)) e.pf[0] |= PF_ALW; if (hi & (pg - 1)) e.pf[np - 1] |= PF_ALW; }
    g_pools[g_npool++] = e; g_psort_dirty = 1;
    if (g_nplog == g_plogcap && g_nplog >= 4096) {   // bounded: sparse pools re-add their runs at every remap (vrun_set), so freed entries go, listed or not (the totals keep them)
        int k = 0; for (int i = 0; i < g_nplog; i++) if (g_plog[i].died == ~0ull) g_plog[k++] = g_plog[i];
        g_nplog = k;
    }
    if (g_nplog == g_plogcap) { g_plogcap = g_plogcap ? g_plogcap * 2 : 64; g_plog = realloc(g_plog, (size_t)g_plogcap * sizeof *g_plog); if (!g_plog) die("vkn backend: pool log allocation failed"); }
    g_plog[g_nplog++] = (pool_log){lo, size, atomic_load(&g_present_n), ~0ull, g_pmade, 0};
    g_pmade++; g_pmade_b += size; g_plive_b += size; if (g_plive_b > g_ppeak_b) g_ppeak_b = g_plive_b;
    pthread_mutex_unlock(&g_pmu);
}
static void b_shadow_free(b_shadow *s) { vk.vkDestroyBuffer(vd.dev, s->buf, NULL); vk.vkFreeMemory(vd.dev, s->mem, NULL); free(s); atomic_fetch_add(&g_shadow_freed, 1); }
void mtl_pool_del(const void *mem) {
    b_shadow *dead = NULL;
    pthread_mutex_lock(&g_pmu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].key == mem) {
        pool_ent *e = &g_pools[i];
        // Condition 3: nvn_dirty_remove_range ran just before (hle_nvn.c) and released the pool's read-only pages, but a commit protecting them then (b_commit, the submit thread) can take pages
        // that pass had already gone by; the range is dead, so nothing would release those (a write there reaches the crash handler). Commits protect under g_pmu with the span cached first,
        // so none runs now and a pool they may have protected has one: other ranges' pages first (a page shared with a neighbour), then all writable (nvn_null.c's fix)
        if (e->pf && e->span) {
            dm_guest_write_prepare((void *)e->lo, e->hi - e->lo);
            if (mprotect((void *)e->lo, e->hi - e->lo, PROT_READ | PROT_WRITE)) NRED_ONCE("B: a freed pool's pages could not be made writable again (errno %d)", errno);
        }
        if ((dead = e->sh)) { g_nshadow--; g_shadow_b -= dead->size; }
        free(e->pf); free(e->ht); *e = g_pools[--g_npool]; g_plast = -1; g_psort_dirty = 1; break;
    }
    for (int i = g_nplog - 1; i >= 0; i--) if (g_plog[i].mem == (uintptr_t)mem && g_plog[i].died == ~0ull) { g_plog[i].died = atomic_load(&g_present_n); g_pfreed++; g_pfreed_b += g_plog[i].size; g_plive_b -= g_plog[i].size; break; }
    pthread_mutex_unlock(&g_pmu);
    // Out of the table: no later record or commit reaches it. Destroyed once every submission that used it has completed (its use_nvn, final now), through the images' dead list and sweep;
    // never after g_leak. g_pmu and vkn_tmu are never held together
    if (dead) { pthread_mutex_lock(&vkn_tmu); dead->next = g_dead_b; g_dead_b = dead; vkn_sweep(); pthread_mutex_unlock(&vkn_tmu); }
}
void mtl_alias_drop_pool(const void *pool) { (void)pool; }
// buf: the pool's shadow VkBuffer, NULL while it has none (callers that only test success, hle_nvn.c's shader code and the CPU-run ops below, see the pool); off from the page-rounded start.
// A range resolves to the pool whose own bytes hold it (key, end), never through the page rounding: of two pools sharing a page, the other one's shadow never has this one's bytes there
static int psort_cmp(const void *a, const void *b) { uintptr_t x = g_pools[*(const int *)a].lo, y = g_pools[*(const int *)b].lo; return x < y ? -1 : x > y; }
static int in_pool(int k, uint64_t addr, uint64_t len) { uint64_t lo = (uintptr_t)g_pools[k].key, hi = g_pools[k].end; return addr >= lo && addr <= hi && len <= hi - addr; }   // no addr + len: a guest-supplied range near 2^64 wraps
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int ok = 0;
    pthread_mutex_lock(&g_pmu);
    int i = g_plast >= 0 && g_plast < g_npool && in_pool(g_plast, addr, len) ? g_plast : -1;
    if (i < 0 && (i = pool_find(addr, len, 0)) >= 0) g_plast = i;
    if (i >= 0) { *buf = g_pools[i].sh ? (void *)g_pools[i].sh->buf : NULL; *off = addr - g_pools[i].lo; ok = 1; }
    pthread_mutex_unlock(&g_pmu);
    return ok;
}
// The GPU's view (nvn_mtl.m mtl_gpu_addr): the shadow, the offset in it and its device address (0 without bufferDeviceAddress). Only a pool with a shadow answers, by its own bytes (in_pool;
// the cached g_plast too). 0 = the pool holding the range has no shadow (a CPU_NO_ACCESS pool, a shadow that could not be made: said there), or no pool holds it
int mtl_gpu_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off, uint64_t *gpu) {
    int ok = 0;
    pthread_mutex_lock(&g_pmu);
    int i = g_plast >= 0 && g_plast < g_npool && g_pools[g_plast].sh && in_pool(g_plast, addr, len) ? g_plast : -1;
    if (i < 0 && (i = pool_find(addr, len, 1)) >= 0) g_plast = i;
    if (i >= 0) { b_shadow *s = g_pools[i].sh; *buf = (void *)s->buf; *off = addr - g_pools[i].lo; *gpu = s->gpu ? s->gpu + *off : 0; ok = 1; }
    pthread_mutex_unlock(&g_pmu);
    return ok;
}
static int cpu_target(uint64_t addr, uint64_t n, const char *what) {   // 1 = [addr, addr+n) is pool memory the CPU may write; the dirty trackers hear of the write first, as for any host write into guest memory
    void *b; uint64_t o;
    if (!mtl_addr(addr, n, &b, &o)) { NLOG("[RED] %s outside every memory pool dst=%#llx n=%#llx: skipped", what, (unsigned long long)addr, (unsigned long long)n); return 0; }
    dm_guest_write_prepare((void *)(uintptr_t)addr, (size_t)n);   // B: the pages it writes are picked at the commit (not ordered before the submission's earlier draws: PLAN.md 2.2)
    ST(S_GW_B, n);   // a GPU write into pool memory: what a backend would have to bring back if the CPU reads it
    return 1;
}

// ---------- B: marks (the replay), shadows, the upload at commit, the statistics line ----------
// A shadow over the pool's page-rounded range (under g_pmu, at the first buffer read): every buffer use a later slice binds it for, transfers both ways, and a device address. NULL = not made (said)
static b_shadow *b_shadow_make(const pool_ent *e) {
    uint64_t n = e->hi - e->lo; const char *no = NULL; VkMemoryRequirements mr = {0}; int mt = -1;
    b_shadow *s = calloc(1, sizeof *s); if (!s) die("vkn backend: shadow entry allocation failed");
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | (vd.bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    if (vk.vkCreateBuffer(vd.dev, &bi, NULL, &s->buf)) { s->buf = VK_NULL_HANDLE; no = "vkCreateBuffer"; }
    else {
        vk.vkGetBufferMemoryRequirements(vd.dev, s->buf, &mr);
        if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, 0);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, vd.bda ? &fi : NULL, mr.size, (uint32_t)mt};
        if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &s->mem)) { s->mem = VK_NULL_HANDLE; no = "device memory"; }
        else if (vk.vkBindBufferMemory(vd.dev, s->buf, s->mem, 0)) no = "vkBindBufferMemory";
    }
    if (no) {
        NRED_ONCE("B: memory pool %#llx..%#llx (%llu B) has no shadow buffer (%s failed): the GPU reads nothing of it (later pools are not said)", (unsigned long long)e->lo, (unsigned long long)e->hi, (unsigned long long)n, no);
        if (s->mem) vk.vkFreeMemory(vd.dev, s->mem, NULL);
        if (s->buf) vk.vkDestroyBuffer(vd.dev, s->buf, NULL);
        free(s); return NULL;
    }
    if (vd.bda) { VkBufferDeviceAddressInfo di = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, s->buf}; s->gpu = vk.vkGetBufferDeviceAddress(vd.dev, &di); }
    s->size = mr.size; g_nshadow++; g_shadow_b += mr.size;
    if (g_shadow_made++ < 32) NLOG("B: shadow buffer for memory pool %#llx..%#llx: %llu B of %s memory (%llu shadows, %llu B)", (unsigned long long)e->lo, (unsigned long long)e->hi, (unsigned long long)mr.size,
                                   vd.mem.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "device-local" : "host", (unsigned long long)g_nshadow, (unsigned long long)g_shadow_b);
    return s;
}
// The pages of [addr, addr+len) get flag in every pool holding them (submit thread, during the replay). A buffer read gives a pool its shadow, and only pools whose own bytes it touches (two pools
// share a page-rounded page: the neighbour gets nothing for it); the shadow's last use is this submission. Command pages (PF_CMD) are marked in any pool
static void b_mark(uint64_t addr, uint64_t len, uint8_t flag) {
    if (!len) return;
    uint64_t end = addr + len < addr ? UINT64_MAX : addr + len;
    pthread_mutex_lock(&g_pmu);
    int idx = psort_on();
    for (int p = idx ? psort_first(addr) : 0; p < g_npool; p++) {
        int k = idx ? g_psort[p] : p;
        pool_ent *e = &g_pools[k]; if (idx && e->lo >= end) break;   // in address order: no later pool reaches the range
        uint64_t lo = addr > e->lo ? addr : e->lo, hi = end < e->hi ? end : e->hi;
        if (!e->pf || lo >= hi) continue;
        if (flag == PF_BUF) {
            if (addr >= e->end || end <= (uintptr_t)e->key) continue;   // only the shared page, none of this pool's bytes
            if (e->noacc) { NRED_ONCE("B: a record reads %#llx+%#llx as a buffer in a CPU-no-access (render-target) pool: such pools have no shadow, the GPU reads nothing there", (unsigned long long)addr, (unsigned long long)len); continue; }
            if (!e->sh && !e->nosh && !(e->sh = b_shadow_make(e))) e->nosh = 1;
            if (!e->sh) continue;
            e->sh->use_nvn = atomic_load(&g_fence) + 1;   // this submission's value: only this thread moves g_fence (mtl_sub_commit)
        }
        for (uint64_t i = (lo - e->lo) >> g_sh, j = (hi - 1 - e->lo) >> g_sh; i <= j; i++) if (!(e->pf[i] & flag)) e->pf[i] |= flag | PF_NEW;   // NEW: copied (and protected) at this commit
    }
    pthread_mutex_unlock(&g_pmu);
}
static void b_cmd(const uint8_t *p, uint32_t n) {   // a record's pages hold command memory (records follow each other, so only a page change looks the pools up)
    static uintptr_t last = ~(uintptr_t)0; uintptr_t a = (uintptr_t)p >> g_sh, b = ((uintptr_t)p + (n ? n : 1) - 1) >> g_sh;
    if (a == last && b == last) return;
    last = b; b_mark((uint64_t)a << g_sh, (uint64_t)(b - a + 1) << g_sh, PF_CMD);
}
// What a record reads as buffers, as the B simulation marks it (nvn_null.c surf_rec): the draw and dispatch slices widen it where they read more (vertex ranges, multi-draw arrays)
static void b_rec(const op_hdr *h, const uint8_t *p, const uint8_t *a) {
    b_cmd(p, h->size);
    switch (h->op) {
        case OP_DRAW: { p_draw t; const p_draw *d = draw_rec(a, h->size, &t); b_mark(d->ind, d->ind ? 16 : 0, PF_BUF); b_mark(d->mdi_cnt, d->mdi_cnt ? 4 : 0, PF_BUF); break; }
        case OP_DRAW_IDX: { p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h->size, &t); int it = d->itype >= 0 && d->itype < 3 ? d->itype : 2;
            b_mark(d->ibuf, d->count > 0 ? (uint64_t)d->count << it : 0, PF_BUF); b_mark(d->ind, d->ind ? 20 : 0, PF_BUF); b_mark(d->mdi_cnt, d->mdi_cnt ? 4 : 0, PF_BUF); break; }
        case OP_DISPATCH: { const p_dispatch *d = (const p_dispatch *)a; b_mark(d->ind, d->ind ? 12 : 0, PF_BUF); break; }
        case OP_COPY_B2B: { uint64_t s_, n; memcpy(&s_, a, 8); memcpy(&n, a + 16, 8); b_mark(s_, n, PF_BUF); break; }
        case OP_COPY_B2T: { const p_copy_b2t *c = (const p_copy_b2t *)a; b_mark(c->src, region_bytes(c->tex, &c->r), PF_BUF); break; }
        case OP_BIND_VBUF: { const p_bind_vbuf *v = (const p_bind_vbuf *)a; for (int i = 0; i < v->n && i < 16; i++) b_mark(v->r[i].address, v->r[i].size, PF_BUF); break; }
        case OP_BIND_UBO: case OP_BIND_SSBO: { const p_bind_buf *u = (const p_bind_buf *)a; b_mark(u->addr, u->size, PF_BUF); break; }
        default: break;
    }
}
// Cumulative statistics (b_report): uploads made, pages by why (0 written since the last commit, 1 read for the first time, 2 copied every time), bytes, B's CPU time at commit; tracked pages now;
// late: the pages of why 0 that only the re-scan after the drain picked (b_commit), and the drain's and re-scan's share of the commit time
static _Atomic uint64_t g_bs_up, g_bs_pg[3], g_bs_b, g_bs_ns, g_bs_trk, g_bs_alw, g_bs_max, g_bs_late, g_bs_dns, g_bs_hot;
static uint64_t g_bstamp;   // the dirty generation at the last commit's scan (under g_pmu)
static VkBufferCopy *g_upc; static b_shadow **g_ups; static int *g_upk, g_nup, g_upcap;   // the upload being built: regions, their shadows and pools (submit thread, under g_pmu)
static int stg_ensure(sub_slot *s, uint64_t n) {   // the slot's staging buffer holds n bytes (its last submission has completed: mtl_sub_begin waited). 0 = none (said)
    if (s->stg_cap >= n) return 1;
    if (s->stg_map) vk.vkUnmapMemory(vd.dev, s->stg_mem);
    if (s->stg) vk.vkDestroyBuffer(vd.dev, s->stg, NULL);
    if (s->stg_mem) vk.vkFreeMemory(vd.dev, s->stg_mem, NULL);
    s->stg = VK_NULL_HANDLE; s->stg_mem = VK_NULL_HANDLE; s->stg_map = NULL; s->stg_cap = 0;
    uint64_t cap = 256 * 1024; while (cap < n) cap *= 2;   // grows by doubling: the dirty set settles in a few frames
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkMemoryRequirements mr; int mt = -1; void *map = NULL;
    if (vk.vkCreateBuffer(vd.dev, &bi, NULL, &s->stg)) { s->stg = VK_NULL_HANDLE; return 0; }
    vk.vkGetBufferMemoryRequirements(vd.dev, s->stg, &mr);
    if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &s->stg_mem)) { s->stg_mem = VK_NULL_HANDLE; vk.vkDestroyBuffer(vd.dev, s->stg, NULL); s->stg = VK_NULL_HANDLE; return 0; }
    if (vk.vkBindBufferMemory(vd.dev, s->stg, s->stg_mem, 0) || vk.vkMapMemory(vd.dev, s->stg_mem, 0, VK_WHOLE_SIZE, 0, &map)) {
        vk.vkDestroyBuffer(vd.dev, s->stg, NULL); vk.vkFreeMemory(vd.dev, s->stg_mem, NULL); s->stg = VK_NULL_HANDLE; s->stg_mem = VK_NULL_HANDLE; return 0; }
    s->stg_map = map; s->stg_cap = cap; s->stg_coh = (vd.mem.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return 1;
}
static int b_protectable(uint8_t f) { return (f & PF_BUF) && !(f & (PF_CMD | PF_ALW | PF_HOT)); }   // a tracked page the commit protects (b_commit's drain), due or not
// A pool's nvn_dirty range, found before any protect (mtl_pool_del keys on it): every place that takes it comes here (under g_pmu). NULL: none yet (hle_nvn.c adds it just after mtl_pool_add, so a
// commit can come between) or the tracker's table is full; the pool is copied every time meanwhile. Found now: its buffer pages were copied with no protection and the fresh range's generations
// say nothing of the writes since (0 <= g_bstamp), so every one is taken as read for the first time (NEW): picked, protected by the drain, then copied at the next commit's scan (slice 2c's fix
// of a write lost in that gap). Whichever finds it first marks them: b_commit's buffer pass, or a texture's upload in the same commit (tex_prep_all), which finds it first when the range is
// added on another thread between the two
static const void *pool_span(pool_ent *e) {
    if (e->span || !(e->span = nvn_dirty_span((const void *)e->lo, e->hi - e->lo)) || !e->pf) return e->span;
    for (uintptr_t i = 0, np = (e->hi - e->lo) >> g_sh; i < np; i++) if (e->pf[i] & PF_BUF) e->pf[i] |= PF_NEW;
    return e->span;
}
// Texture jobs (the replay's tex_ready, at an image's first use in the submission): what the upload command buffer does to it ahead of the submission, which starts from its layout then. Kinds:
// an upload from guest memory (the first, again after a guest write, again because it is never protected), a zero fill (never written, nvn_mtl.m tex_zero_fill), the one-time fill from its
// storage's initial content (TJ_INIT: nvn_mtl.m tex_init_fill, no write tracking), or only the layout change to shader-read. An upload that cannot be made (no pool, no staging, no deswizzle:
// said there) falls back to a zero fill or the layout change (tj_fallback) and is tried again at the next submission
enum { TJ_NONE, TJ_UP1, TJ_RE, TJ_REALW, TJ_EDGE, TJ_ZERO, TJ_LAYOUT, TJ_INIT };   // TJ_EDGE: written since, maybe only beside it (decided at the commit: TJ_RE or nothing)
static int tj_copies(int k) { return (k >= TJ_UP1 && k <= TJ_REALW) || k == TJ_INIT; }   // the kinds that copy bytes from guest memory through the staging buffer
typedef struct { vkn_img *e; VkImageLayout old; int kind, alw, r0, nr; uint64_t off, bytes; const void *span; } tjob;   // r0, nr: its copy regions in g_tcr; off: in the staging buffer
static tjob *g_tj; static int g_ntj, g_tjcap;   // the submission being replayed (submit thread)
static uint64_t tex_prep_all(uint64_t off, uint64_t stamp);   // under g_pmu, in b_commit (the texture section below)
static void tex_fill_all(uint8_t *stg);
static void tex_done_all(uint64_t stamp);
static void tex_unprep_all(void);
static void tex_record(VkCommandBuffer cb, VkBuffer stg);
// The commit's upload (submit thread, before the submission is queued; v its timeline value): 1 = s->up holds copies into the shadows, to run ahead of s->cb. Under g_pmu, in this order:
// the dirty generation is stamped (S); the pages to copy are picked (PF_BUF pages written since the last stamp B, NEW ones, and those never protected); every protectable tracked page is
// protected, due or not (the drain); the ones not picked are looked at again against B (the re-scan) and those newer are picked too; then the picked ones are copied to the slot's staging.
// A write after the protect faults and is newer than S, so the next commit picks it; one before it lands in this copy. A run the drain could not protect goes back to NEW (and its range is
// always written now, so the re-scan picks it): copied now, and copied and tried again at the next commit.
// Why the drain covers every tracked page and the re-scan is needed (nvn_dirty.c's store order in an unprotect U of page P, by a write fault or a host pre-write: (c) state PROT -> BUSY,
// (m) mprotect writable, (f) g = fetch_add(g_now) + 1, (s) gen[P] = g, (n) state NONE; the write W it was for comes after (n)): (f) moves g_now before (s) publishes g. A commit whose stamp
// read falls between them gets S >= g and its scan may still read P's old generation: P is not picked, and the next commit's "newer than S" misses g forever, P staying writable. Sound now:
// - any (f) at or before S belongs to a page that was BUSY at S or has already published its gen: let C be the first commit whose stamp read comes after U's (f) (so S >= g, and B, read
//   before (f), is < g). U's (c) precedes its (f), which precedes C's stamp read, so C's drain reads P's state at (c) or later: BUSY (U under way: it waits, nvn_dirty_protect_wait),
//   NONE (U done: it protects P) or PROT (protected since by another caller of nvn_dirty_protect). It reads that state after (n) in each case, so (s) is visible to the re-scan, which
//   finds gen[P] >= g > B and copies P after its protect; W came before that protect (and is in the copy) or faults again. (U's mprotect failing leaves P PROT and its range always
//   written, as does the drain's wait timing out: nvn_dirty_since_span says 1 there, so the re-scan picks P all the same);
// - that fault, or any unprotect whose (c) comes after the drain read P, has its (f) after S: g > S, and the next commit's scan picks it (the drain after the stamp is what orders the two).
// No caller other than this one protects these pages at present (only hle_nvn.c's ft_add under the tex_dirty_upload profile, off for Stardew); the argument holds for one anyway, as above.
// A page the drain finds PROT costs one failed exchange: beyond the picked pages it protects only those written since the scan or caught in that window (the re-scan's, g_bs_late).
// Pages that hold commands, edge pages (a page shared with a neighbouring pool: mtl_pool_add) and a pool the tracker does not cover (no span) are copied every time and never protected.
// A run the tracker finds in another range than the pool's own (find_span takes the first range holding it: only pools overlapping in bytes, two pools over the same guest memory, none on
// this route; edge pages no longer get here) is copied every time too, since protecting it there would leave this pool's generations blind; it is checked per run, so such an overlap that
// covers only part of a run is not caught.
// Naive: the whole dirty set is copied and protected again at every submission. The hot-set scheme (stardew-surface.md: copy the pages written k submissions running without protecting them,
// protect again once their bytes stop changing) is for the gameplay sub-slice
static int b_commit(sub_slot *s, uint64_t v) {
    uint64_t t0 = now_ns(), bytes = 0, pg[3] = {0}, trk = 0, alw = 0, late = 0, dns = 0, hot = 0; int ok = 1, hon = b_hot();
    pthread_mutex_lock(&g_pmu);
    uint64_t stamp = nvn_dirty_now();
    g_nup = 0;
    for (int k = 0; k < g_npool; k++) {
        pool_ent *e = &g_pools[k]; if (!e->sh || !e->pf) continue;
        uintptr_t np = (e->hi - e->lo) >> g_sh;
        pool_span(e);   // the range, found before any protect; found now: every buffer page NEW (pool_span)
        for (uintptr_t i = 0; i < np; i++) {
            uint8_t f = e->pf[i]; if (!(f & PF_BUF)) continue;
            if (f & PF_HOT) {   // copied; at HOT_TTL protected again by this commit's drain (copied after it, as any due page)
                trk++; alw++; hot++; pg[2]++;
                if (++e->ht[i] >= HOT_TTL || !hon) { f &= (uint8_t)~PF_HOT; e->ht[i] = 1; }
                e->pf[i] = (uint8_t)((f & ~PF_NEW) | PF_DUE); continue;
            }
            int fixed = (f & (PF_CMD | PF_ALW)) || !e->span; trk++; alw += fixed;
            if (!fixed && !(f & PF_NEW) && !nvn_dirty_since_span(e->span, (const void *)(e->lo + (i << g_sh)), g_pgsz, g_bstamp)) { e->ht[i] = e->ht[i] && e->ht[i] < HOT_WIN ? e->ht[i] + 1 : 0; continue; }
            pg[fixed ? 2 : f & PF_NEW ? 1 : 0]++;
            if (!fixed && !(f & PF_NEW) && hon && e->ht[i]) { f |= PF_HOT; e->ht[i] = 0; }   // written again soon: left writable from now on (its fault made it so)
            else e->ht[i] = fixed ? 0 : 1;
            e->pf[i] = (uint8_t)((f & ~PF_NEW) | PF_DUE);
        }
        uint64_t td = now_ns();
        for (uintptr_t i = 0; e->span && i < np; ) {   // the drain: every protectable tracked page, one call per run (see above)
            if (!b_protectable(e->pf[i])) { i++; continue; }
            uintptr_t j = i; while (j < np && b_protectable(e->pf[j])) j++;
            const void *ra = (const void *)(e->lo + (i << g_sh)); size_t rl = (size_t)(j - i) << g_sh;
            if (nvn_dirty_span(ra, rl) != e->span)   // copied every time from now on, this time included
                for (uintptr_t q = i; q < j; q++) { pg[2] += !(e->pf[q] & PF_DUE); alw++; e->pf[q] |= PF_ALW | PF_DUE; }
            else if (!nvn_dirty_protect_wait(ra, rl)) for (uintptr_t q = i; q < j; q++) e->pf[q] |= PF_NEW;   // not all protected (said there): copied now (the re-scan), and copied and tried again next time
            i = j;
        }
        for (uintptr_t i = 0; e->span && i < np; i++) {   // the re-scan against B (g_bstamp is still the last commit's): NEW stays as the drain left it
            uint8_t f = e->pf[i];
            if (!b_protectable(f) || (f & PF_DUE) || !nvn_dirty_since_span(e->span, (const void *)(e->lo + (i << g_sh)), g_pgsz, g_bstamp)) continue;
            pg[0]++; late++; e->pf[i] = f | PF_DUE;
        }
        dns += now_ns() - td;
        for (uintptr_t i = 0; i < np; ) {   // the copies, one region per run
            if (!(e->pf[i] & PF_DUE)) { i++; continue; }
            uintptr_t j = i; while (j < np && (e->pf[j] & PF_DUE)) e->pf[j++] &= (uint8_t)~PF_DUE;
            if (g_nup == g_upcap) { g_upcap = g_upcap ? g_upcap * 2 : 64; g_upc = realloc(g_upc, (size_t)g_upcap * sizeof *g_upc); g_ups = realloc(g_ups, (size_t)g_upcap * sizeof *g_ups);
                g_upk = realloc(g_upk, (size_t)g_upcap * sizeof *g_upk); if (!g_upc || !g_ups || !g_upk) die("vkn backend: upload list allocation failed"); }
            g_upc[g_nup] = (VkBufferCopy){bytes, (VkDeviceSize)i << g_sh, (VkDeviceSize)(j - i) << g_sh}; g_ups[g_nup] = e->sh; g_upk[g_nup++] = k; bytes += (uint64_t)(j - i) << g_sh;
            i = j;
        }
        e->sh->use_nvn = v;
    }
    // Textures (tex_jobs): each upload's pool found, its pages protected (after this stamp, as the buffers' pages), its share of the staging buffer after the buffers'
    uint64_t tb0 = (bytes + 15) & ~15ull, tb = tex_prep_all(tb0, stamp);
    int need = g_nup || tb > tb0, skip = need && !stg_ensure(s, tb);
    if (skip) NRED_ONCE("B: no staging buffer of %llu B: this submission's uploads are skipped (taken at the next one)", (unsigned long long)tb);
    else if (need) {
        for (int r = 0; r < g_nup; r++) memcpy((uint8_t *)s->stg_map + g_upc[r].srcOffset, (const void *)(g_pools[g_upk[r]].lo + g_upc[r].dstOffset), (size_t)g_upc[r].size);
        tex_fill_all(s->stg_map);   // the textures deswizzled into it (still under g_pmu: a pool cannot go while its bytes are read)
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, s->stg_mem, 0, VK_WHOLE_SIZE}; VkResult fr;
        if (!s->stg_coh && (fr = vk.vkFlushMappedMemoryRanges(vd.dev, 1, &mr))) { skip = 1; NRED_ONCE("B: flushing the staging buffer failed (VkResult %d): this submission's uploads are skipped (taken at the next one)", fr); }
    }
    if (skip) {   // not copied: the pages go back to NEW, so the next commit takes them whatever their protection says now (under g_pmu: g_upk's pool indices hold only here)
        for (int r = 0; r < g_nup; r++) for (uintptr_t i = g_upc[r].dstOffset >> g_sh, c = g_upc[r].size >> g_sh; c--; i++) g_pools[g_upk[r]].pf[i] |= PF_NEW;
        g_nup = 0; ok = 0;
        tex_unprep_all();   // and the textures are uploaded at their next use (their stamps stay those of their last upload)
    } else tex_done_all(stamp);
    g_bstamp = stamp;
    int n = g_nup, nt = g_ntj;
    pthread_mutex_unlock(&g_pmu);
    // Recorded outside the lock: the shadows named stay alive (use_nvn = v, not completed; a pool freed meanwhile only queues its shadow for vkn_sweep), and so do the images (use_mark)
    if (n || nt) {
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vk.vkBeginCommandBuffer(s->up, &bi);
    }
    if (n) {
        // Before: every earlier submission's GPU work is done with the shadows (reads: write-after-read; an earlier upload's copies: write-after-write)
        VkMemoryBarrier2 b0 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
        VkDependencyInfo d0 = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &b0, 0, NULL, 0, NULL};
        vk.vkCmdPipelineBarrier2(s->up, &d0);
        for (int r = 0; r < n; ) { int q = r; while (q < n && g_ups[q] == g_ups[r]) q++; vk.vkCmdCopyBuffer(s->up, s->stg, g_ups[r]->buf, (uint32_t)(q - r), g_upc + r); r = q; }
        // After: what reads buffers in the submission (and later ones) sees the copies
        VkMemoryBarrier2 b1 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo d1 = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &b1, 0, NULL, 0, NULL};
        vk.vkCmdPipelineBarrier2(s->up, &d1);
    }
    if (nt) tex_record(s->up, s->stg);   // the textures' layout changes, copies and fills, after the buffers'
    if (n || nt) vk.vkEndCommandBuffer(s->up);
    g_ntj = 0;
    atomic_fetch_add(&g_bs_up, n != 0); for (int i = 0; i < 3; i++) atomic_fetch_add(&g_bs_pg[i], ok ? pg[i] : 0); atomic_fetch_add(&g_bs_b, n ? bytes : 0);
    atomic_store(&g_bs_trk, trk); atomic_store(&g_bs_alw, alw); atomic_store(&g_bs_hot, hot); if (ok && pg[0] + pg[1] + pg[2] > atomic_load(&g_bs_max)) atomic_store(&g_bs_max, pg[0] + pg[1] + pg[2]);
    atomic_fetch_add(&g_bs_late, ok ? late : 0); atomic_fetch_add(&g_bs_dns, dns); atomic_fetch_add(&g_bs_ns, now_ns() - t0);
    return n || nt;   // 1: s->up goes ahead of s->cb (buffer copies, texture jobs, or both)
}
// The B line (periodic as the surface lines, and at mtl_drain): per frame since the last line, and the state now. Faults and pre-write unprotects are the process's (nvn_dirty.c counts them;
// on this backend only B protects). Compare stardew-surface.md section 4: the title screen at 4 KB pages (the phone) 4.13 pages a frame, 31 tracked; at 16 KB (the Mac) about 3.3 a frame, 52 KiB
static void b_report(const char *why) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static uint64_t p_n, p_up, p_pg[3], p_b, p_ns, p_fa, p_pr, p_mc, p_late, p_dns;
    pthread_mutex_lock(&mu);
    uint64_t n = atomic_load(&g_present_n), up = atomic_load(&g_bs_up), pg[3], b = atomic_load(&g_bs_b), ns = atomic_load(&g_bs_ns), late = atomic_load(&g_bs_late), dns = atomic_load(&g_bs_dns), fa, pr, mc, mp, mn;
    for (int i = 0; i < 3; i++) pg[i] = atomic_load(&g_bs_pg[i]);
    nvn_dirty_stats(&fa, &pr); nvn_dirty_mp_stats(&mc, &mp, &mn);
    pthread_mutex_lock(&g_pmu); uint64_t nsh = g_nshadow, shb = g_shadow_b, made = g_shadow_made; pthread_mutex_unlock(&g_pmu);
    double f = n > p_n ? (double)(n - p_n) : 1.0, all = (double)(pg[0] + pg[1] + pg[2] - p_pg[0] - p_pg[1] - p_pg[2]);
    NLOG("B (shadow buffers) up to frame %llu (%s), per frame over the last %llu: pages uploaded %.2f (written since the last submission %.2f, of them picked only by the re-scan %.2f, read for the first time %.2f, copied every time %.2f), %.1f KiB, "
         "submissions with an upload %.2f; write faults %.2f, host pre-write unprotects %.2f (process-wide), mprotect calls %.2f; B at commit %.3f ms (the drain and re-scan %.3f ms); now %llu pages tracked (%llu B pages), %llu of them copied every time (%llu hot), "
         "%llu shadows %llu B; totals: %llu pages, %llu B, at most %llu pages in one submission, shadows made %llu destroyed %llu",
         (unsigned long long)n, why, (unsigned long long)(n - p_n), all / f, (double)(pg[0] - p_pg[0]) / f, (double)(late - p_late) / f, (double)(pg[1] - p_pg[1]) / f, (double)(pg[2] - p_pg[2]) / f, (double)(b - p_b) / 1024.0 / f,
         (double)(up - p_up) / f, (double)(fa - p_fa) / f, (double)(pr - p_pr) / f, (double)(mc - p_mc) / f, (double)(ns - p_ns) / 1e6 / f, (double)(dns - p_dns) / 1e6 / f,
         (unsigned long long)atomic_load(&g_bs_trk), (unsigned long long)g_pgsz, (unsigned long long)atomic_load(&g_bs_alw), (unsigned long long)atomic_load(&g_bs_hot), (unsigned long long)nsh, (unsigned long long)shb,
         (unsigned long long)(pg[0] + pg[1] + pg[2]), (unsigned long long)b, (unsigned long long)atomic_load(&g_bs_max), (unsigned long long)made, (unsigned long long)atomic_load(&g_shadow_freed));
    p_n = n; p_up = up; for (int i = 0; i < 3; i++) p_pg[i] = pg[i]; p_b = b; p_ns = ns; p_fa = fa; p_pr = pr; p_mc = mc; p_late = late; p_dns = dns;
    pthread_mutex_unlock(&mu);
}

// ---------- Host images (vkn.h vkn_img), by serial: VkImages in device memory, made when a texture in a CPU-no-access pool is initialized (render-target pools) or first set as a target ----------
pthread_mutex_t vkn_tmu = PTHREAD_MUTEX_INITIALIZER, vkn_qmu = PTHREAD_MUTEX_INITIALIZER;
#define IT_B 1024
static vkn_img *g_it[IT_B], *g_dead_l;   // buckets by serial; released images waiting for their last uses (vkn_sweep)
static uint64_t g_nimg, g_img_b, g_img_made;
// Ghosts: host images of serials already finalized when a record names them (tex_live: hle_nvn.c snapshots the texture object at record time, and Breath of the Wild finalizes its
// transient targets before it submits, nvn_mtl.m's g_ghost). The table would keep them for good (no second release comes), so they live in a small ring instead: the oldest goes to the
// pending-destruction list when a new one comes, and a record replayed again soon still finds its image
#define NGHOST 16
static vkn_img *g_ghost[NGHOST]; static unsigned g_ghost_next; static uint64_t g_ghost_puts;
static vkn_img **it_slot(uint64_t s) { vkn_img **p = &g_it[(s * 0x9e3779b97f4a7c15ull) >> 54]; while (*p && (*p)->serial != s) p = &(*p)->next; return p; }
vkn_img *vkn_img_get(uint64_t s) {
    if (!s) return NULL;
    vkn_img *e = *it_slot(s); if (e || !g_ghost_puts) return e;
    for (int i = 0; i < NGHOST; i++) if (g_ghost[i] && g_ghost[i]->serial == s) return g_ghost[i];
    return NULL;
}
static void img_dead(vkn_img *e) { e->dead = 1; e->next = g_dead_l; g_dead_l = e; }   // destroyed by vkn_sweep once its last uses have completed
static VkFormatFeatureFlags fmt_feat(VkFormat f) { VkFormatProperties p; vk.vkGetPhysicalDeviceFormatProperties(vd.pd, f, &p); return p.optimalTilingFeatures; }
static VkImageAspectFlags fmt_aspect(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32: return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT: return VK_IMAGE_ASPECT_STENCIL_BIT;
        case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default: return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}
// NVN format → the VkFormat of its host image: nvn_fmt_vk.c's first candidate the device has for the use. want: NVK_RT a render target (an attachment), NVK_UP a texture uploaded from guest
// memory (sampled, a copy's destination), both: a texture sampled before anything wrote it (an attachment where its format is a target format, else as NVK_UP). sf: the format the present samples
// it as (UNORM for an sRGB one: bytes pass through); asp: its aspects. 0 = none: not in the table, not for this use, or no candidate the device has. Depth falls back to D32_SFLOAT_S8_UINT where
// D24S8 is missing (KosmicKrisp has no D24_UNORM_S8_UINT)
static VkFormat fmt_map(unsigned nf, int want, VkFormat *sf, VkImageAspectFlags *asp) {
    const nvk_fmt *m = nvn_fmt_vk((int)nf);
    if (!m || !(m->flags & want)) return 0;
    for (int k = 0; k < 3 && m->f[k]; k++) {
        VkFormat f = m->f[k]; VkImageAspectFlags a = fmt_aspect(f);
        VkFormatFeatureFlags need = (want & m->flags & NVK_RT) ? (a == VK_IMAGE_ASPECT_COLOR_BIT ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT : VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                                                           : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((fmt_feat(f) & need) != need) continue;
        *sf = m->lin ? m->lin : f; *asp = a; return f;
    }
    return 0;
}
// A host image for t (under vkn_tmu), with every usage its format offers (attachment, sampled, both transfers): one image per serial, whichever use made it. want as fmt_map's. why: NULL at
// DM_VK_DIAG=afbc (diagnostics): the driver's answer (VkImageCompressionPropertiesEXT) on whether a host image is compressed, once per kind (format, usage, flags), and for the same create
// info without STORAGE usage, without MUTABLE_FORMAT and its format list, and with STORAGE added: what each costs (on Mali: AFBC on or off). Flags 0 = compressed (default), 4 = disabled
static uint32_t comp_of(VkImage img, VkImageAspectFlags asp) {
    VkImageCompressionPropertiesEXT cp = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    VkImageSubresource2KHR sr = {VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR, NULL, {asp & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : asp & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : asp, 0, 0}};
    VkSubresourceLayout2KHR lo = {VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR, &cp};
    vd.layout2(vd.dev, img, &sr, &lo); return cp.imageCompressionFlags;
}
static int comp_probe(const VkImageCreateInfo *ii, VkImageAspectFlags asp) {   // -1 = could not be made
    VkImage im; VkDeviceMemory m = VK_NULL_HANDLE; VkMemoryRequirements mr; int r = -1;
    if (vk.vkCreateImage(vd.dev, ii, NULL, &im)) return -1;
    vk.vkGetImageMemoryRequirements(vd.dev, im, &mr); int mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt >= 0 && !vk.vkAllocateMemory(vd.dev, &ai, NULL, &m) && !vk.vkBindImageMemory(vd.dev, im, m, 0)) r = (int)comp_of(im, asp);
    vk.vkDestroyImage(vd.dev, im, NULL); if (m) vk.vkFreeMemory(vd.dev, m, NULL); return r;
}
static void img_comp(const VkImageCreateInfo *ii, VkImage img, VkImageAspectFlags asp, uint64_t ser, VkFormatFeatureFlags ff) {
    static struct { VkFormat f; VkImageUsageFlags u; VkImageCreateFlags c; } seen[96]; static int ns, nimg, ncomp;
    uint32_t got = comp_of(img, asp); nimg++; ncomp += got == 0;
    for (int i = 0; i < ns; i++) if (seen[i].f == ii->format && seen[i].u == ii->usage && seen[i].c == ii->flags) goto count;
    if (ns == (int)(sizeof seen / sizeof *seen)) goto count;
    seen[ns].f = ii->format; seen[ns].u = ii->usage; seen[ns].c = ii->flags; ns++;
    { VkImageCreateInfo a = *ii; int nost = -2, nomut = -2, addst = -2;
      if (a.usage & VK_IMAGE_USAGE_STORAGE_BIT) { a.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT; nost = comp_probe(&a, asp); a = *ii; }
      if (a.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) { a.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; a.pNext = NULL; nomut = comp_probe(&a, asp); a = *ii; }
      if (!(a.usage & VK_IMAGE_USAGE_STORAGE_BIT) && ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) { a.usage |= VK_IMAGE_USAGE_STORAGE_BIT; addst = comp_probe(&a, asp); }
      NLOG("[afbc] ser=%llu %ux%u x%u levels %u format %d usage %#x flags %#x%s: compression %#x; without STORAGE %d, without MUTABLE_FORMAT %d, with STORAGE added %d (-2: n/a, -1: not made)",
           (unsigned long long)ser, ii->extent.width, ii->extent.height, ii->arrayLayers, ii->mipLevels, ii->format, ii->usage, ii->flags, ii->pNext ? " +format list" : "", got, nost, nomut, addst); }
count:
    if (!(nimg & 255)) NLOG("[afbc] %d host images so far, %d compressed", nimg, ncomp);
}
// DM_VK_STORAGE_SER=<ser>[,<ser>…] (an experiment): only those images get STORAGE usage (the ones the dispatches write, logged as "writes ser=… as a storage image"); every other colour
// image keeps its compression (AFBC on Mali: STORAGE usage turns it off). A dispatch writing an image not in the list is skipped (said in red)
// STORAGE usage turns compression (AFBC) off on Mali, so by default no image is made with it: a dispatch that binds one as a storage image first remakes it with STORAGE (cs_storage_up).
// DM_VK_STORAGE_IMG=1: every colour image whose format allows it gets STORAGE when made (as before, with DM_VK_STORAGE_SER narrowing it); =0: none ever (its dispatches are skipped)
static int g_mk_storage;   // img_make for a dispatch's storage image: it gets STORAGE now
static int storage_mode(void) { static int m = -1; if (m < 0) { const char *x = getenv("DM_VK_STORAGE_IMG"); m = x && *x == '0' ? 0 : x && *x == '1' ? 1 : 2; } return m; }
static int storage_ser(uint64_t ser);
static int storage_want(uint64_t ser) { int m = storage_mode(); return m == 1 ? storage_ser(ser) : m == 2 && g_mk_storage; }
static int storage_ser(uint64_t ser) {
    static const char *l = (const char *)1; if (l == (const char *)1) l = getenv("DM_VK_STORAGE_SER");
    if (!l || !*l) return 1;
    for (const char *p = l; *p; ) { char *e; unsigned long long v = strtoull(p, &e, 10); if (e == p) break; if (v == ser) return 1; p = *e == ',' ? e + 1 : e; }
    return 0;
}
// TextureInitialize (a texture this slice cannot render to is passed over quietly: not every texture in such a pool is a target), else the use that wants it (said in red when it cannot have one)
static vkn_img *img_make(const tex_t *t, const char *why, int want) {
    VkFormat sf = 0, f = 0; VkImageAspectFlags asp = 0; const char *no = NULL; int tg = t->d.target;
    static int img3d = -1; if (img3d < 0) { const char *v = getenv("DM_VK_3D_IMG"); img3d = !(v && *v == '0'); }   // 3D textures: DM_VK_3D_IMG=0 leaves them without host images (read as zeros)
    int is3d = tg == NVN_TEXTURE_TARGET_3D && img3d && !(want & NVK_RT && !(want & NVK_UP));
    static int cube_on = -1; if (cube_on < 0) { const char *v = getenv("DM_VK_CUBE"); cube_on = !(v && *v == '0'); }   // cube maps: DM_VK_CUBE=0 leaves them out (read as zeros)
    int iscube = (tg == NVN_TEXTURE_TARGET_CUBEMAP || tg == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) && cube_on;
    if (tg != NVN_TEXTURE_TARGET_2D && tg != NVN_TEXTURE_TARGET_2D_ARRAY && tg != NVN_TEXTURE_TARGET_RECTANGLE && !is3d && !iscube) no = "its target";
    else if (t->d.samples > 1) no = "multisampling";
    else if (!t->d.width || !t->d.height) no = "a zero size";
    else if (!(f = fmt_map((unsigned)t->d.format, want, &sf, &asp))) no = "its format";
    if (no) {
        if (why && want == NVK_RT) NRED_ONCE("%s: texture ser=%llu %ux%u format %#x target %d samples %d has no host image (%s is not handled yet): its passes are skipped", why, (unsigned long long)t->serial,
                                             t->d.width, t->d.height, t->d.format, tg, t->d.samples, no);
        else if (why) NRED_ONCE("%s: texture ser=%llu %ux%u format %#x target %d samples %d has no host image (%s is not handled yet): it is sampled as nothing", why, (unsigned long long)t->serial,
                                t->d.width, t->d.height, t->d.format, tg, t->d.samples, no);
        return NULL;
    }
    uint32_t w = t->d.width, h = t->d.height, layers = (tg == NVN_TEXTURE_TARGET_2D_ARRAY || tg == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) && t->d.depth ? t->d.depth : tg == NVN_TEXTURE_TARGET_CUBEMAP ? 6 : 1, lv = t->d.levels ? t->d.levels : 1, mx = 1, dz = is3d && t->d.depth ? t->d.depth : 1;
    for (uint32_t m = w > h ? (w > dz ? w : dz) : (h > dz ? h : dz); m > 1; m >>= 1) mx++;
    if (lv > mx) lv = mx;
    VkFormatFeatureFlags ff = fmt_feat(f); int col = asp == VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageUsageFlags u = (!is3d && ff & (col ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT : VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ? (col ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) : 0)
                        | (ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT ? VK_IMAGE_USAGE_SAMPLED_BIT : 0) | (ff & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0)
                        | (ff & VK_FORMAT_FEATURE_TRANSFER_DST_BIT ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0)
                        | (col && sf == f && (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && storage_want(t->serial) ? VK_IMAGE_USAGE_STORAGE_BIT : 0);   // compute shaders write it as an image (op_dispatch: see storage_mode); not with a second view format (its views would inherit the usage)
    VkFormat fl[2] = {f, sf}; VkImageFormatListCreateInfo li = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, fl};   // the two formats it is viewed as (a driver may keep its compression)
    int cubec = cube_on && !is3d && w == h && layers >= 6 && layers % 6 == 0;   // square with whole sets of six layers: a game may view it as a cube (TTYD's lighting probes)
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, sf != f ? &li : NULL, (sf != f ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0) | (cubec ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0), is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, f, {w, h, dz}, lv, layers, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    vkn_img *e = calloc(1, sizeof *e); if (!e) die("vkn backend: image entry allocation failed");
    VkMemoryRequirements mr; int mt;
    if (vk.vkCreateImage(vd.dev, &ii, NULL, &e->img)) { NRED_ONCE("vkCreateImage %ux%u format %d failed: texture ser=%llu has no host image", w, h, f, (unsigned long long)t->serial); free(e); return NULL; }
    vk.vkGetImageMemoryRequirements(vd.dev, e->img, &mr);
    if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &e->mem) || vk.vkBindImageMemory(vd.dev, e->img, e->mem, 0)) {
        NRED_ONCE("device memory for a %ux%u host image (%llu B) failed: texture ser=%llu has none", w, h, (unsigned long long)mr.size, (unsigned long long)t->serial);
        if (e->mem) vk.vkFreeMemory(vd.dev, e->mem, NULL); vk.vkDestroyImage(vd.dev, e->img, NULL); free(e); return NULL; }
    if (vd.layout2) img_comp(&ii, e->img, asp, t->serial, ff);
    e->serial = t->serial; e->fmt = f; e->sfmt = sf; e->aspect = asp; e->w = w; e->h = h; e->depth = dz; e->cube = cubec; e->layers = layers; e->levels = lv; e->nvn_fmt = (uint32_t)t->d.format;
    e->usage = u; e->gpu = want == NVK_RT; e->gmem = t->mem; e->size = t->L.size; e->d = t->d;
    e->layout = e->rec = VK_IMAGE_LAYOUT_UNDEFINED; g_img_b += mr.size;
    if (tex_live(e->serial)) { vkn_img **p = it_slot(e->serial); e->next = *p; *p = e; g_nimg++; }   // checked under vkn_tmu: a release after this finds it in the table
    else {
        vkn_img **g = &g_ghost[g_ghost_next++ % NGHOST]; if (*g) img_dead(*g);
        *g = e; e->next = NULL;
        if (!g_ghost_puts++) NLOG("texture ser=%llu was finalized before a record naming it was replayed: its host image is a ghost (the last %d are kept)", (unsigned long long)e->serial, NGHOST);
    }
    if (g_img_made++ < 24) NLOG("host image ser=%llu %ux%u x%u levels %u, NVN format %#x as %d, %llu B of device memory (%s; %llu images, %llu B)", (unsigned long long)e->serial, w, h, layers, lv,
                                e->nvn_fmt, f, (unsigned long long)mr.size, why ? why : "a texture in a CPU-no-access pool", (unsigned long long)g_nimg, (unsigned long long)g_img_b);
    return e;
}
static int att_ok(const vkn_img *e) { return (e->usage & (e->aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0; }   // it can be a target
VkImageView vkn_img_view(vkn_img *e, uint32_t level, uint32_t layer, int present) {
    uint32_t key = present ? ~0u : level << 16 | layer;
    for (int i = 0; i < e->nviews; i++) if (e->views[i].key == key) return e->views[i].v;
    if (level >= e->levels || layer >= e->layers) { NRED_ONCE("an attachment view of level %u layer %u, beyond a host image's %u / %u: skipped", level, layer, e->levels, e->layers); return VK_NULL_HANDLE; }
    if (e->nviews == (int)(sizeof e->views / sizeof *e->views)) { NRED_ONCE("a host image has more than %d views: the rest are skipped", e->nviews); return VK_NULL_HANDLE; }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, e->img, VK_IMAGE_VIEW_TYPE_2D, present ? e->sfmt : e->fmt, {0},
        {present ? VK_IMAGE_ASPECT_COLOR_BIT : e->aspect, present ? 0 : level, 1, present ? 0 : layer, 1}};
    VkImageView v; if (vk.vkCreateImageView(vd.dev, &vi, NULL, &v)) { NRED_ONCE("vkCreateImageView failed (host image ser=%llu)", (unsigned long long)e->serial); return VK_NULL_HANDLE; }
    e->views[e->nviews].key = key; e->views[e->nviews++].v = v;
    return v;
}
static void img_free(vkn_img *e) {
    if (e->fbv) vk.vkDestroyImageView(vd.dev, e->fbv, NULL);
    if (e->fbsv) vk.vkDestroyImageView(vd.dev, e->fbsv, NULL);
    if (e->fbi) { vk.vkDestroyImage(vd.dev, e->fbi, NULL); vk.vkFreeMemory(vd.dev, e->fbm, NULL); }
    for (int i = 0; i < e->nviews; i++) vk.vkDestroyImageView(vd.dev, e->views[i].v, NULL);
    for (int i = 0; i < e->nsviews; i++) vk.vkDestroyImageView(vd.dev, e->sviews[i].v, NULL);
    free(e->edge[0]); free(e->edge[1]);
    VkMemoryRequirements mr; vk.vkGetImageMemoryRequirements(vd.dev, e->img, &mr); g_img_b -= mr.size;
    vk.vkDestroyImage(vd.dev, e->img, NULL); vk.vkFreeMemory(vd.dev, e->mem, NULL); free(e);
}
// Under vkn_tmu. g_leak is read after the completed values: comp_main sets it (seq_cst) before it stores the g_done of a submission retired without the GPU's answer, so a g_done that counts
// such a submission is never paired with a g_leak read as 0 (read first, g_leak could be 0, then the retirement, then a g_done past an image the GPU still uses). The present timeline is
// the GPU's own counter (vk_present.c retires nothing without it; vkn_pres_done reads one it cannot get as 0: nothing done)
void vkn_sweep(void) {
    if (!g_dead_l && !g_dead_b) return;
    uint64_t dn = atomic_load(&g_done), dp = vkn_pres_done();
    if (atomic_load(&g_leak)) return;   // a submission retired without the GPU's answer: g_done says nothing of what the GPU still uses
    for (vkn_img **p = &g_dead_l; *p; ) { vkn_img *e = *p; if (e->use_nvn <= dn && e->use_pres <= dp) { *p = e->next; img_free(e); } else p = &e->next; }
    for (b_shadow **p = &g_dead_b; *p; ) { b_shadow *s = *p; if (s->use_nvn <= dn) { *p = s->next; b_shadow_free(s); } else p = &s->next; }   // freed pools' shadows (mtl_pool_del): NVN submissions only
}


// One record's share of the surface statistics. g_pass_drawn: a clear before the pass's first draw folds into its load action
static int g_pass_drawn;
static void surf_rec(const op_hdr *h, const uint8_t *p, const uint8_t *a) {
    if (h->op < OP_COUNT_) ST(h->op, 1);
    switch (h->op) {
        case OP_SET_RT: { const p_set_rt *r = (const p_set_rt *)a; int nc = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n, ms = 0; g_pass_drawn = 0; ST(S_RT_NC + nc, 1);
            for (int i = 0; i < nc; i++) if (r->color[i]) { fmt_note(0, (unsigned)r->color[i]->d.format); gpu_wrote(r->color[i]); ms |= r->color[i]->d.samples > 1; }
            if (r->depth) { ST(S_RT_DEPTH, 1); if (!nc) ST(S_RT_ZONLY, 1); fmt_note(0, (unsigned)r->depth->d.format); gpu_wrote(r->depth); ms |= r->depth->d.samples > 1; }
            if (ms) ST(S_RT_MSAA, 1);
            break; }
        case OP_CLEAR_COLOR: { const p_clear_color *c = (const p_clear_color *)a; ST(g_pass_drawn ? S_CLR_MID : S_CLR_START, 1); if ((c->mask & 0xF) != 0xF) ST(S_CLR_PART, 1); break; }   // a partial mask needs a draw (vkCmdClearAttachments writes every channel)
        case OP_CLEAR_DS: { const p_clear_ds *c = (const p_clear_ds *)a; int sm = c->stencil_mask & 0xff; ST(g_pass_drawn ? S_DS_MID : S_DS_START, 1); if (sm && sm != 0xff) ST(S_DS_PART, 1); break; }
        case OP_DRAW: { p_draw t; const p_draw *d = draw_rec(a, h->size, &t); g_pass_drawn = 1; ST(S_PRIM + (d->prim & 15), 1); ST(S_DRAW_VTX, d->count > 0 ? d->count : 0);
            if (d->instances > 1) ST(S_DRAW_INST, 1); if (d->base_instance) ST(S_DRAW_BINST, 1); if (d->ind) ST(S_DRAW_IND, 1); if (d->mdi_cnt) ST(S_DRAW_MDI, 1);
            break; }
        case OP_DRAW_IDX: { p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h->size, &t); int it = d->itype >= 0 && d->itype < 3 ? d->itype : 2; g_pass_drawn = 1;
            ST(S_PRIM + (d->prim & 15), 1); ST(S_ITYPE + it, 1); ST(S_DIDX_IDX, d->count > 0 ? d->count : 0);
            if (d->instances > 1) ST(S_DIDX_INST, 1); if (d->base_vertex) ST(S_DIDX_BVTX, 1); if (d->base_instance) ST(S_DIDX_BINST, 1); if (d->ind) ST(S_DIDX_IND, 1); if (d->mdi_cnt) ST(S_DIDX_MDI, 1);
            break; }
        case OP_DISPATCH: { const p_dispatch *d = (const p_dispatch *)a; if (d->ind) ST(S_DISP_IND, 1); break; }
        case OP_COPY_B2B: { uint64_t n; memcpy(&n, a + 16, 8); ST(S_B2B_B, n); break; }
        case OP_UPDATE_UB: { uint64_t n; memcpy(&n, a + 8, 8); ST(S_UB_B, n); break; }
        case OP_CLEAR_BUF: ST(S_FILL_B, ((const p_clear_buf *)a)->size); break;
        case OP_COPY_B2T: { const p_copy_b2t *c = (const p_copy_b2t *)a; uint64_t n = region_bytes(c->tex, &c->r); ST(S_B2T_B, n); gpu_wrote(c->tex); break; }
        case OP_COPY_T2B: { const p_copy_b2t *c = (const p_copy_b2t *)a; ST(S_T2B_B, region_bytes(c->tex, &c->r)); break; }   // src holds the destination buffer: a GPU write into pool memory
        case OP_COPY_T2T: { const p_copy_t2t *c = (const p_copy_t2t *)a; if (c->flags & DM_COPY_RESOLVE) ST(S_T2T_RES, 1); gpu_wrote(c->dst); break; }
        case OP_CLEAR_TEX: gpu_wrote(((const p_clear_tex *)a)->tex); break;
        case OP_REPORT_COUNTER: { uint64_t t; memcpy(&t, a, 8); ST(t == NVN_COUNTER_TYPE_TIMESTAMP || t == NVN_COUNTER_TYPE_TIMESTAMP_TOP ? S_CNT_TS : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? S_CNT_SP : S_CNT_OTH, 1); break; }
        case OP_TF_STATE: ST(S_TF_WHAT + (((const p_tf_state *)a)->what & 3), 1); break;
        case OP_RENDER_COND: ST(S_RC_MODE + (((const p_render_cond *)a)->mode & 3), 1); break;
        case OP_BARRIER: { int32_t bits; memcpy(&bits, a, 4); for (int k = 0; k < 8; k++) if (bits >> k & 1) ST(S_BAR + k, 1); break; }
        case OP_VP_SWIZZLE: { const p_vp_swizzle *w = (const p_vp_swizzle *)a; uint32_t m = 0;   // Adreno has no viewport swizzle: each one needs another way (a flipped viewport for NEGATIVE_Y)
            for (int i = 0; i < w->n && i < 16; i++) for (int k = 0; k < 4; k++) if (w->sw[i][k]) m |= 1u << (8 * k + ((w->sw[i][k] ^ 2 * k) & 7));
            if (m) { ST(S_VPSW, 1); atomic_fetch_or_explicit(&g_vpsw, m, memory_order_relaxed); }
            break; }
        case OP_BIND_STATE: ST(S_STATE + (((const p_bind_state *)a)->kind % 6), 1); break;
        case OP_BIND_VATTR: { const p_bind_vattr *v = (const p_bind_vattr *)a; for (int i = 0; i < v->n && i < 16; i++) fmt_note(2, v->st[i] & 0xff); break; }   // va_st.fmt, the low byte
        default: break;
    }
}

// ---------- Passes: the replay's render-target state (submit thread) ----------
// Targets are kept by serial with their view's level and layer (the game may copy or reuse the tex_t), resolved to host images at SET_RT and again at each submission's start (an image
// released between submissions is gone). A clear before the pass's first draw is recorded and lands as that attachment's load op when the pass begins (rp_begin), as nvn_mtl.m's rp_begin_ex_
// folds it; at SET_RT, a barrier or the submission's end, recorded clears with no draw get a pass of their own (rp_flush, nvn_mtl.m's). A clear after the first draw, or under a scissor that does
// not cover the target, is a vkCmdClearAttachments in the open pass (PLAN.md 2.7: nvn_mtl.m draws a triangle there)
static struct {
    int has_rt, n; uint64_t cser[8], dser; uint32_t clv[8], cly[8], dlv, dly;
    vkn_img *ce[8], *de;                           // resolved for this submission
    int clear_c[8], clear_d, clear_s; VkClearColorValue cc[8]; float cd; uint32_t cs;
    int has_sc; p_scissor sc;
    int on; VkExtent2D area; int att_c[8], att_ds;   // a pass is open (vkCmdBeginRendering), its area and what it attached (vkCmdClearAttachments names those only)
    uint32_t nc; int hd, hs; VkFormat cf[8], dsf;   // the open pass's colour attachment count and formats (UNDEFINED where none), its depth / stencil (a pipeline must name the same: draw_pso)
    unsigned act;                                  // the colour slots the last SET_RT named: one naming some of the open pass's targets keeps the pass, the others are not written (rt_sub)
} R;
static uint64_t g_sub_id;                          // this submission (vkn_img.rec_sub)
static vkn_img **g_touch; static int g_ntouch, g_touchcap;   // images whose layout this submission tracks (published at commit)
static VkCommandBuffer cb_now(void) { return g_cur->cb; }
#define SR_STAGES (VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
static VkImageLayout rec_layout(vkn_img *e) {
    if (e->rec_sub != g_sub_id) {
        e->rec_sub = g_sub_id; e->rec = e->layout; e->sr_st = SR_STAGES;   // the submit thread is the only writer of layout (at commit), so this read needs no lock
        if (g_ntouch == g_touchcap) { g_touchcap = g_touchcap ? g_touchcap * 2 : 64; g_touch = realloc(g_touch, (size_t)g_touchcap * sizeof *g_touch); if (!g_touch) die("vkn backend: touch list allocation failed"); }
        g_touch[g_ntouch++] = e;
    }
    return e->rec;
}
static void use_mark(vkn_img *e) { if (e) e->use_nvn = atomic_load(&g_fence) + 1; }   // under vkn_tmu: this submission's fence; destruction waits for it
static void rt_resolve(const p_set_rt *r) {   // SET_RT: the targets by serial, made here when TextureInitialize did not (a target outside a CPU-no-access pool)
    pthread_mutex_lock(&vkn_tmu);
    for (int i = 0; i < R.n; i++) {
        const tex_t *t = r->color[i]; R.ce[i] = NULL; R.cser[i] = t ? t->serial : 0;
        if (!t || !t->serial) continue;
        R.clv[i] = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0; R.cly[i] = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0;
        if (r->has_cview[i] && r->cview[i].set_format && r->cview[i].format != t->d.format)
            NRED_ONCE("SET_RT: a colour target viewed as format %#x, the texture's is %#x (ser=%llu): rendered in the texture's own format (views of another format are a later slice)", r->cview[i].format, t->d.format, (unsigned long long)t->serial);
        int dup = 0; for (int k = 0; k < i; k++) dup |= R.cser[k] == R.cser[i] && R.clv[k] == R.clv[i] && R.cly[k] == R.cly[i];
        if (dup) { NRED_ONCE("SET_RT attaches one target view to two colour slots: the second is skipped"); R.cser[i] = 0; continue; }
        if (!(R.ce[i] = vkn_img_get(t->serial))) R.ce[i] = img_make(t, "SET_RT", NVK_RT);
        else if (!att_ok(R.ce[i])) { NRED_ONCE("SET_RT: colour target ser=%llu format %#x was first sampled, and its host image cannot be a target: its passes are skipped", (unsigned long long)t->serial, t->d.format); R.ce[i] = NULL; }
        use_mark(R.ce[i]);
    }
    const tex_t *d = r->depth; R.de = NULL; R.dser = d ? d->serial : 0;
    if (d && d->serial) {
        R.dlv = r->has_dview && r->dview.set_levels ? r->dview.base_level : 0; R.dly = r->has_dview && r->dview.set_layers ? r->dview.min_layer : 0;
        if (r->has_dview && r->dview.set_format && r->dview.format != d->d.format)
            NRED_ONCE("SET_RT: a depth target viewed as format %#x, the texture's is %#x (ser=%llu): rendered in the texture's own format (views of another format are a later slice)", r->dview.format, d->d.format, (unsigned long long)d->serial);
        if (!(R.de = vkn_img_get(d->serial))) R.de = img_make(d, "SET_RT", NVK_RT);
        else if (!att_ok(R.de)) { NRED_ONCE("SET_RT: depth target ser=%llu format %#x was first sampled, and its host image cannot be a target: its passes are skipped", (unsigned long long)d->serial, d->d.format); R.de = NULL; }
        use_mark(R.de);
    }
    pthread_mutex_unlock(&vkn_tmu);
}
static void rt_reresolve(void) {   // a submission's start: the targets the last SET_RT left, as they are now
    if (!R.has_rt) return;
    pthread_mutex_lock(&vkn_tmu);
    for (int i = 0; i < R.n; i++) { R.ce[i] = vkn_img_get(R.cser[i]); if (R.ce[i] && !att_ok(R.ce[i])) R.ce[i] = NULL; use_mark(R.ce[i]); }   // not a target: said at SET_RT
    R.de = vkn_img_get(R.dser); if (R.de && !att_ok(R.de)) R.de = NULL; use_mark(R.de);
    pthread_mutex_unlock(&vkn_tmu);
}
static VkExtent2D lv_ext(const vkn_img *e, uint32_t lv) { VkExtent2D x = {e->w >> lv, e->h >> lv}; if (!x.width) x.width = 1; if (!x.height) x.height = 1; return x; }
static int sc_full(VkExtent2D x) { return !R.has_sc || (R.sc.x <= 0 && R.sc.y <= 0 && (int64_t)R.sc.x + R.sc.w >= (int64_t)x.width && (int64_t)R.sc.y + R.sc.h >= (int64_t)x.height); }
static void mem_barrier(VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, ss, sa, ds, da};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &b, 0, NULL, 0, NULL};
    vk.vkCmdPipelineBarrier2(cb_now(), &di);
}
// An attachment's barrier at a pass's start: into its attachment layout (the whole image: one tracked layout each), after everything before it wrote (an earlier pass on it included)
static void att_barrier(VkImageMemoryBarrier2 *b, vkn_img *e) {
    int col = e->aspect == VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout to = col ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    *b = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        col ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        col ? VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        rec_layout(e), to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    e->rec = to;
}
static void clears_reset(void) { memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0; }
// Deferred clears (DM_VK_CLEAR_FOLD=0: off). A full clear of a target that no draw follows in its pass (SET_RT X, CLEAR_DS, SET_RT of others) and a CLEAR_TEX of a whole level
// are kept on the image (pc) instead of being recorded: the next pass that attaches that level and layer takes it as its load op, any other use records it first (pend_resolve:
// tex_lookup, copies, buffer-to-texture copies, feedback copies), and a dispatch and the submission's end record them all (pend_all). On a tiler a clear recorded alone writes the whole
// image and the next pass loads it back; folded, the pass starts from the clear colour with no read. Paper Mario: The Thousand-Year Door clears each next target with CLEAR_TEX right
// before its SET_RT (about 25 a frame, its 960x540 G-buffer among them) and clears 18 depth targets a frame in passes with no draw
static int g_fold_on = -1;
static int fold_on(void) { if (g_fold_on < 0) { const char *e = getenv("DM_VK_CLEAR_FOLD"); g_fold_on = !(e && *e == '0'); if (!g_fold_on) NLOG("DM_VK_CLEAR_FOLD=0: clears are recorded where they come"); } return g_fold_on; }
static uint64_t g_pend[64]; static int g_npend;   // the serials of the images with a deferred clear
static uint64_t g_stat_pend[4];   // deferred, taken as a load op, recorded between passes, recorded after ending a pass
static uint64_t g_stat_dsn[2];   // loaded depth / stencil attachments not stored (rp_begin: no draw of the pass writes them)
static void pend_rec(vkn_img *e, int locked);
// Opens the pass on the targets. clear_only: attach only the targets with a recorded clear (rp_flush). 0 = nothing to attach (the pass is skipped, as nvn_mtl.m skips one with no texture)
typedef struct { unsigned w; int64_t x0, y0, x1, y1; } pass_look;   // what the pass may write: depth / stencil bits (ds_writes), and its draws' pixels (x0 > x1: none; full: unknown)
static pass_look pass_scan(uint32_t fw, uint32_t fh);
static uint64_t g_stat_area[2];   // passes given a render area smaller than their targets, and the pixels it left out (load and store skip them)
static int area_on(void) { static int on = -1; if (on < 0) { const char *x = getenv("DM_VK_AREA"); on = !(x && *x == '0'); } return on; }
static int rp_begin(int clear_only) {
    if (R.on) return 1;
    if (!R.has_rt) { NRED_ONCE("a draw or clear with no render target set: skipped"); clears_reset(); return 0; }
    VkRenderingAttachmentInfo ca[8], da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkExtent2D ex[9]; VkImageMemoryBarrier2 b[9]; uint32_t nb = 0, nc = 0, w = UINT32_MAX, h = UINT32_MAX; int any = 0, has_d = 0, has_s = 0;
    for (int i = 0; i < 8; i++) { ca[i] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; R.att_c[i] = 0; R.cf[i] = VK_FORMAT_UNDEFINED; }
    R.att_ds = 0; R.dsf = VK_FORMAT_UNDEFINED;
    pthread_mutex_lock(&vkn_tmu);   // views are made under it (the present thread makes its own on the same images)
    for (int i = 0; i < R.n; i++) {
        vkn_img *e = R.ce[i]; VkImageView v;
        if (!e || (clear_only && !R.clear_c[i]) || !(v = vkn_img_view(e, R.clv[i], R.cly[i], 0))) continue;
        if (e->pc && !R.clear_c[i]) {   // a deferred clear of this level and layer becomes the load op; of another, it is recorded first
            if ((e->pc & 1) && e->pc_lv == R.clv[i] && e->pc_ly == R.cly[i]) { R.clear_c[i] = 1; R.cc[i] = e->pcv.color; e->pc = 0; g_stat_pend[1]++; } else pend_rec(e, 1); }
        ca[i].imageView = v; ca[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ca[i].loadOp = R.clear_c[i] ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE; ca[i].clearValue.color = R.cc[i];
        ex[i] = lv_ext(e, R.clv[i]); if (ex[i].width < w) w = ex[i].width; if (ex[i].height < h) h = ex[i].height;
        att_barrier(&b[nb++], e); nc = (uint32_t)i + 1; any = 1; R.att_c[i] = 1; R.cf[i] = e->fmt;
        if (R.clear_c[i]) { memcpy(e->clear, R.cc[i].float32, sizeof e->clear); e->cleared = 1; g_stat_fold++; }
    }
    vkn_img *de = R.de; VkImageView dv = VK_NULL_HANDLE;
    if (de && (!clear_only || R.clear_d || R.clear_s) && (dv = vkn_img_view(de, R.dlv, R.dly, 0))) {
        if (de->pc) {   // as for colour: per aspect, the recorded clear wins
            if (de->pc_lv == R.dlv && de->pc_ly == R.dly) {
                if ((de->pc & 2) && !R.clear_d) { R.clear_d = 1; R.cd = de->pcv.depthStencil.depth; }
                if ((de->pc & 4) && !R.clear_s) { R.clear_s = 1; R.cs = de->pcv.depthStencil.stencil; }
                de->pc = 0; g_stat_pend[1]++;
            } else pend_rec(de, 1);
        }
        VkClearValue cv; cv.depthStencil.depth = R.cd; cv.depthStencil.stencil = R.cs;
        if (de->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { has_d = 1; da.imageView = dv; da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; da.loadOp = R.clear_d ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
                                                      da.storeOp = VK_ATTACHMENT_STORE_OP_STORE; da.clearValue = cv; g_stat_fold += R.clear_d; }
        if (de->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { has_s = 1; sa.imageView = dv; sa.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; sa.loadOp = R.clear_s ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
                                                        sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE; sa.clearValue = cv; g_stat_fold += R.clear_s; }
        // A loaded depth or stencil that no draw or clear of the pass can write is not stored (STORE_OP_NONE, core in 1.3: the store leaves it as it is). On a tiler the store is a
        // full write of the image back to memory: Paper Mario: The Thousand-Year Door's 960x540 depth-stencil is loaded by ~20 passes a frame that only test against it (the
        // characters' shadows, lighting). Opt-in, DM_VK_DS_STORE=1: on the RG 476H ~16 depth and ~3.5 stencil loads a frame were not stored, yet the frame stayed 39.8 → 39.6-39.8 ms (Mali seems to skip
        // unchanged tiles itself). ds_scan reads ahead to the pass's end. DM_VK_DIAG=dsnone (wrong on purpose, a bound): a loaded depth / stencil is never stored (39.7 → 37.0 ms, the
        // characters vanish: the gain is in passes that write depth)
        { static int on = -1, dsn; if (on < 0) { const char *x = getenv("DM_VK_DS_STORE"); on = x && *x == '1'; x = getenv("DM_VK_DIAG"); dsn = x && strstr(x, "dsnone"); }
          unsigned w = dsn ? 0 : !on ? 3 : clear_only ? 0 : (has_d && !R.clear_d) || (has_s && !R.clear_s) ? pass_scan(1, 1).w : 3;
          if (has_d && !R.clear_d && !(w & 1)) { da.storeOp = VK_ATTACHMENT_STORE_OP_NONE; g_stat_dsn[0]++; }
          if (has_s && !R.clear_s && !(w & 2)) { sa.storeOp = VK_ATTACHMENT_STORE_OP_NONE; g_stat_dsn[1]++; } }
        ex[8] = lv_ext(de, R.dlv); if (ex[8].width < w) w = ex[8].width; if (ex[8].height < h) h = ex[8].height;
        att_barrier(&b[nb++], de); any = 1; R.att_ds = 1; R.dsf = de->fmt;
    }
    pthread_mutex_unlock(&vkn_tmu);
    clears_reset();
    if (!any) return 0;
    VkCommandBuffer cb = cb_now();
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, nb, b};
    vk.vkCmdPipelineBarrier2(cb, &di);
    // A pass covers only its smallest attachment, and so does a clear folded into its load op, while NVN clears the whole target (nvn_mtl.m: Bayonetta 2's corner-only depth clear):
    // a cleared attachment larger than the pass is cleared in a pass of its own first, then loaded
    int own = 0;
    for (int i = 0; i < 9; i++) {
        int ds = i == 8; VkRenderingAttachmentInfo *c = ds ? NULL : &ca[i];
        if (ds ? !R.att_ds || !((has_d && da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) || (has_s && sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)) : !R.att_c[i] || c->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR) continue;
        if (ex[i].width <= w && ex[i].height <= h) continue;
        VkRenderingAttachmentInfo d1 = da, s1 = sa;
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, ex[i]}, 1, 0, ds ? 0 : 1, ds ? NULL : c, ds && has_d ? &d1 : NULL, ds && has_s ? &s1 : NULL};
        vk.vkCmdBeginRendering(cb, &ri); vk.vkCmdEndRendering(cb);
        if (ds) { da.loadOp = sa.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; } else c->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        own = 1; g_stat_own++;
    }
    if (own) mem_barrier(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    R.area = (VkExtent2D){w, h}; R.nc = nc; R.hd = has_d; R.hs = has_s;
    // The render area: the part of the targets the pass's draws can reach (their viewport and scissor: pass_scan), rounded out to 32 pixels, where no attachment is cleared by its load
    // op (that clears the area only) and the scan saw the pass's end. Load and store ops touch only the area, the rest of each target is left as it is: on a tiler a pass drawing a
    // corner loads and stores only that corner. Paper Mario: The Thousand-Year Door's reflections blur at 320x180 in 960x540 targets, ~7 passes a frame. DM_VK_AREA=0: the whole targets
    VkRect2D ra = {{0, 0}, R.area};
    if (area_on() && !clear_only) {
        int clr = has_d && da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR; clr |= has_s && sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
        for (uint32_t i = 0; i < nc; i++) clr |= R.att_c[i] && ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
        if (!clr) {
            pass_look L = pass_scan(w, h);
            if (L.x0 > L.x1) L.x0 = L.y0 = L.x1 = L.y1 = 0;   // no draw: an empty pass loads and stores nothing
            int64_t x0 = L.x0 & ~31, y0 = L.y0 & ~31, x1 = (L.x1 + 31) & ~31, y1 = (L.y1 + 31) & ~31;
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 > w) x1 = w; if (y1 > h) y1 = h;
            if (x1 <= x0 || y1 <= y0) { x0 = y0 = 0; x1 = y1 = 1; }   // an area must not be empty
            if ((uint64_t)(x1 - x0) * (uint64_t)(y1 - y0) < (uint64_t)w * h) {
                ra = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
                g_stat_area[0]++; g_stat_area[1] += (uint64_t)w * h - (uint64_t)ra.extent.width * ra.extent.height; }
        }
    }
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, ra, 1, 0, nc, ca, has_d ? &da : NULL, has_s ? &sa : NULL};
    vk.vkCmdBeginRendering(cb, &ri);
    R.on = 1; g_stat_pass++;
    return 1;
}
static int g_pl_on;   // DM_VK_PASSLOG's frame is being replayed: pass ends are logged with the line that ended them
static void rp_end_at(int line) { if (R.on) { vk.vkCmdEndRendering(cb_now()); R.on = 0; if (g_pl_on) NLOG("passlog   pass of c0=ser%llu ended at nvn_vk.c:%d", (unsigned long long)(R.n ? R.cser[0] : 0), line); } }
#define rp_end() rp_end_at(__LINE__)
static int clears_defer(void);
static void rp_flush_at(int line) {   // recorded clears with no draw are deferred (clears_defer) or land in a pass of their own; an open pass ends
    int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
    if (pend && !R.on && !clears_defer()) rp_begin(1);
    rp_end_at(line);
}
#define rp_flush() rp_flush_at(__LINE__)
// ---------- Texture copies and clears outside passes (COPY_T2T, CLEAR_TEX) ----------
// Paper Mario: The Thousand-Year Door renders at 1600x900 and copies the frame into its 1920x1080 window texture: without these the window stayed all zeros. A copy is a
// vkCmdCopyImage when the regions are the same size, unmirrored, of one format; otherwise a vkCmdBlitImage (linear when the game asks, NVN_COPY_FLAGS_LINEAR_FILTER_BIT; MIRROR_X / Y
// walk the source from its far edge, as nvn_mtl.m's copy_t2t_draw). A clear is vkCmdClearColorImage over whole levels. Each end's layout becomes a transfer layout in the submission's
// tracking (rec); the next use moves it on as it does after a pass. DM_VK_COPY=0: neither is done (counted only, as before)
static int g_copy_on = -1;
static int copy_on(void) { if (g_copy_on < 0) { const char *e = getenv("DM_VK_COPY"); g_copy_on = !(e && *e == '0'); if (!g_copy_on) NLOG("DM_VK_COPY=0: texture copies and clears are not done"); } return g_copy_on; }
static void xfer_barrier(vkn_img *e, VkImageLayout to, VkAccessFlags2 da, uint32_t lv, uint32_t ly, uint32_t nly) {
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, da,
        rec_layout(e), to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    (void)lv; (void)ly; (void)nly;   // the whole image: one tracked layout each
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &b};
    vk.vkCmdPipelineBarrier2(cb_now(), &di); e->rec = to;
}
// Records e's deferred clear (outside a pass: an open one ends first); locked: the caller holds vkn_tmu
static void pend_rec(vkn_img *e, int locked) {
    if (!e || !e->pc) return;
    uint8_t pc = e->pc; e->pc = 0;
    for (int i = 0; i < g_npend; i++) if (g_pend[i] == e->serial) { g_pend[i] = g_pend[--g_npend]; break; }
    if (R.on) { rp_end(); g_stat_pend[3]++; } else g_stat_pend[2]++;
    if (locked) use_mark(e); else { pthread_mutex_lock(&vkn_tmu); use_mark(e); pthread_mutex_unlock(&vkn_tmu); }
    xfer_barrier(e, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT, e->pc_lv, e->pc_ly, 1);
    VkImageSubresourceRange r = {0, e->pc_lv, 1, e->pc_ly, 1};
    if (pc & 1) { r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; vk.vkCmdClearColorImage(cb_now(), e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &e->pcv.color, 1, &r); }
    else { r.aspectMask = (pc & 2 ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (pc & 4 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0); vk.vkCmdClearDepthStencilImage(cb_now(), e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &e->pcv.depthStencil, 1, &r); }
}
static void pend_resolve(vkn_img *e) { pend_rec(e, 0); }
static void pend_all(void) {   // every deferred clear (images released since are dropped)
    while (g_npend) {
        uint64_t ser = g_pend[g_npend - 1];
        pthread_mutex_lock(&vkn_tmu); vkn_img *e = vkn_img_get(ser); pthread_mutex_unlock(&vkn_tmu);
        if (e && e->pc && !e->dead) pend_resolve(e); else { if (e) e->pc = 0; g_npend--; }
    }
}
// Keeps a full clear of e's level lv layer ly (bits as pc) for later; 0 = not kept (the caller records it). A deferred clear of another level or layer is recorded first
static int pend_set(vkn_img *e, uint8_t bits, uint32_t lv, uint32_t ly, const VkClearValue *v) {
    if (!fold_on() || !e || !(e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    if (e->pc && (e->pc_lv != lv || e->pc_ly != ly || ((e->pc | bits) & 1) != (bits & 1))) pend_resolve(e);
    if (!e->pc) {
        if (g_npend == (int)(sizeof g_pend / sizeof *g_pend)) pend_all();
        g_pend[g_npend++] = e->serial; e->pc_lv = lv; e->pc_ly = ly;
    }
    if (bits & 1) e->pcv.color = v->color;
    if (bits & 2) e->pcv.depthStencil.depth = v->depthStencil.depth;
    if (bits & 4) e->pcv.depthStencil.stencil = v->depthStencil.stencil;
    e->pc |= bits; g_stat_pend[0]++;
    return 1;
}
// rp_flush with recorded clears and no pass: each target's clear is deferred (all or none: 0 leaves them recorded)
static int clears_defer(void) {
    if (!fold_on()) return 0;
    for (int i = 0; i < R.n && i < 8; i++) if (R.clear_c[i] && (!R.ce[i] || !(R.ce[i]->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))) return 0;
    if ((R.clear_d || R.clear_s) && R.de && !(R.de->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    for (int i = 0; i < R.n && i < 8; i++) if (R.clear_c[i]) { VkClearValue v; v.color = R.cc[i]; pend_set(R.ce[i], 1, R.clv[i], R.cly[i], &v);
                                                               pthread_mutex_lock(&vkn_tmu); memcpy(R.ce[i]->clear, R.cc[i].float32, sizeof R.ce[i]->clear); R.ce[i]->cleared = 1; pthread_mutex_unlock(&vkn_tmu); }
    if ((R.clear_d || R.clear_s) && R.de) { VkClearValue v; v.depthStencil.depth = R.cd; v.depthStencil.stencil = R.cs; pend_set(R.de, (uint8_t)((R.clear_d ? 2 : 0) | (R.clear_s ? 4 : 0)), R.dlv, R.dly, &v); }
    clears_reset();
    return 1;
}
static int buf_host(VkDeviceSize n, VkBufferUsageFlags u, VkBuffer *b, VkDeviceMemory *m, void **map);
static int b_use(uint64_t addr, VkBuffer *buf, uint64_t *off, uint64_t *avail);
static vkn_img *xfer_img(const tex_t *t, int dst) {   // under vkn_tmu: the host image of a copy's end (a destination is made as a target would be; a source must exist)
    if (!t || !t->serial) return NULL;
    vkn_img *e = vkn_img_get(t->serial);
    if (!e && dst) e = img_make(t, "a copy or clear destination", NVK_RT | NVK_UP);   // a render target where the format can be one, else a sampled image
    use_mark(e);
    return e;
}
static uint64_t g_fb_ep = 1;   // the write epoch of feedback copies (fb_copy): a dispatch or buffer-to-texture copy starts another
static void op_copy_t2t(const p_copy_t2t *c) {
    if (!copy_on()) return;
    if (c->flags & DM_COPY_RESOLVE) { NRED_ONCE("a multisample resolve (Downsample): not done (multisampled host images are a later slice)"); return; }
    rp_flush();
    pthread_mutex_lock(&vkn_tmu);
    vkn_img *s = xfer_img(c->src, 0), *d = xfer_img(c->dst, 1);
    if (d) d->fb_ok = 0;   // its feedback copy is no longer current (fb_copy)
    pthread_mutex_unlock(&vkn_tmu);
    pend_resolve(s); pend_resolve(d);
    if (!s || !d) { NRED_ONCE("COPY_T2T: the %s (ser=%llu) has no host image: the copy is skipped", !s ? "source" : "destination", (unsigned long long)(!s ? (c->src ? c->src->serial : 0) : (c->dst ? c->dst->serial : 0))); return; }
    if (!(s->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(d->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || s->aspect != d->aspect) {
        NRED_ONCE("COPY_T2T ser=%llu → ser=%llu: formats %d → %d cannot be copied on this device (usage or aspect): skipped", (unsigned long long)s->serial, (unsigned long long)d->serial, s->fmt, d->fmt); return; }
    uint32_t sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    uint32_t sy = (c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0) + (uint32_t)(c->sr.zoffset > 0 ? c->sr.zoffset : 0), dy = (c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0) + (uint32_t)(c->dr.zoffset > 0 ? c->dr.zoffset : 0);
    uint32_t n = (uint32_t)(c->sr.depth > 1 ? c->sr.depth : 1);
    if (sl >= s->levels || dl >= d->levels || sy + n > s->layers || dy + n > d->layers) { NRED_ONCE("COPY_T2T: a level or layer beyond a host image: skipped"); return; }
    int same = s == d; VkImageLayout sL = same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dL = same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    xfer_barrier(s, sL, VK_ACCESS_2_TRANSFER_READ_BIT | (same ? VK_ACCESS_2_TRANSFER_WRITE_BIT : 0), sl, sy, n);
    if (!same) xfer_barrier(d, dL, VK_ACCESS_2_TRANSFER_WRITE_BIT, dl, dy, n);
    VkImageSubresourceLayers ss = {s->aspect, sl, sy, n}, ds = {d->aspect, dl, dy, n};
    int mir = c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT);
    if (!mir && c->sr.width == c->dr.width && c->sr.height == c->dr.height && s->fmt == d->fmt && c->sr.width > 0 && c->sr.height > 0) {
        VkImageCopy r = {ss, {c->sr.xoffset, c->sr.yoffset, 0}, ds, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        vk.vkCmdCopyImage(cb_now(), s->img, sL, d->img, dL, 1, &r);
    } else {
        if (!(fmt_feat(s->fmt) & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(fmt_feat(d->fmt) & VK_FORMAT_FEATURE_BLIT_DST_BIT) || s->aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
            NRED_ONCE("COPY_T2T ser=%llu → ser=%llu: a scaled, mirrored or converting copy of formats %d → %d, which this device cannot blit: skipped", (unsigned long long)s->serial, (unsigned long long)d->serial, s->fmt, d->fmt); return; }
        int x0 = c->sr.xoffset, y0 = c->sr.yoffset, x1 = x0 + c->sr.width, y1 = y0 + c->sr.height, t;
        if (c->flags & NVN_COPY_FLAGS_MIRROR_X_BIT) { t = x0; x0 = x1; x1 = t; }
        if (c->flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) { t = y0; y0 = y1; y1 = t; }
        VkImageBlit r = {ss, {{x0, y0, 0}, {x1, y1, 1}}, ds, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
        int lin = (c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && (fmt_feat(s->fmt) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
        vk.vkCmdBlitImage(cb_now(), s->img, sL, d->img, dL, 1, &r, lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    { static int nlog; if (nlog++ < 4) NLOG("COPY_T2T ser=%llu %ux%u → ser=%llu %ux%u (region %dx%d → %dx%d, flags %#x): done", (unsigned long long)s->serial, s->w, s->h, (unsigned long long)d->serial, d->w, d->h,
                                      c->sr.width, c->sr.height, c->dr.width, c->dr.height, (unsigned)c->flags); }
}
static void op_clear_tex(const p_clear_tex *c) {
    if (!copy_on()) return;
    rp_flush();
    pthread_mutex_lock(&vkn_tmu); vkn_img *e = xfer_img(c->tex, 1); pthread_mutex_unlock(&vkn_tmu);
    if (e) e->fb_ok = 0;
    if (!e) return;   // said by img_make
    if (e->aspect != VK_IMAGE_ASPECT_COLOR_BIT || !(e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) { NRED_ONCE("CLEAR_TEX: texture ser=%llu (format %d) is not a colour image this device can clear: skipped", (unsigned long long)e->serial, e->fmt); return; }
    uint32_t lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, ly = (c->has_view && c->view.set_layers ? c->view.min_layer : 0) + (uint32_t)(c->r.zoffset > 0 ? c->r.zoffset : 0);
    uint32_t n = (uint32_t)(c->r.depth > 1 ? c->r.depth : 1);
    if (lv >= e->levels || ly + n > e->layers) { NRED_ONCE("CLEAR_TEX: a level or layer beyond a host image: skipped"); return; }
    VkExtent2D x = lv_ext(e, lv);
    int whole = (c->mask & 0xF) == 0xF && c->r.xoffset <= 0 && c->r.yoffset <= 0 && (uint32_t)c->r.width >= x.width && (uint32_t)c->r.height >= x.height;
    if (whole && n == 1) { VkClearValue cv; memcpy(&cv.color, c->c.u, sizeof cv.color); if (pend_set(e, 1, lv, ly, &cv)) return; }   // float or integer bits alike
    pend_resolve(e);
    if (!whole)
        NRED_ONCE("CLEAR_TEX of part of a level or of some channels (mask %#x, region %dx%d at %d,%d of %ux%u): the whole level is cleared", (unsigned)c->mask, c->r.width, c->r.height, c->r.xoffset, c->r.yoffset, x.width, x.height);
    xfer_barrier(e, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT, lv, ly, n);
    VkClearColorValue v; memcpy(&v, c->c.u, sizeof v);   // float or integer bits alike (is_uint: the integer form)
    VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, ly, n};
    vk.vkCmdClearColorImage(cb_now(), e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &r);
}
// vkCmdClearAttachments in the open pass: over its area, or the scissor's part of it
static void clear_in_pass(const VkClearAttachment *a, VkExtent2D x) {
    VkClearRect r = {{{0, 0}, R.area}, 0, 1};
    if (!sc_full(x)) {
        int64_t x0 = R.sc.x > 0 ? R.sc.x : 0, y0 = R.sc.y > 0 ? R.sc.y : 0, x1 = (int64_t)R.sc.x + R.sc.w, y1 = (int64_t)R.sc.y + R.sc.h;
        if (x1 > R.area.width) x1 = R.area.width; if (y1 > R.area.height) y1 = R.area.height;
        if (x1 <= x0 || y1 <= y0) return;
        r.rect = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    }
    vk.vkCmdClearAttachments(cb_now(), 1, a, 1, &r); g_stat_clr_mid++;
}
// A full clear in the open pass of an attachment larger than the pass: vkCmdClearAttachments reaches the pass's area only, NVN clears the whole target. The pass ends and the clear moves to the
// start of the next one, as nvn_mtl.m does with a mid-pass depth-stencil clear: rp_begin clears an oversize attachment in a pass of its own, then loads it (the attachments not cleared load too).
// A scissored clear stays in the pass, clipped to its area (nvn_mtl.m's triangle is too). 1 = the pass was ended, record the clear
static int oversize_mid(VkExtent2D x) {
    if (x.width <= R.area.width && x.height <= R.area.height) return 0;
    rp_end();
    static int once; if (!once++) NLOG("mid-pass full clear of a %ux%u attachment in a %ux%u pass: the pass ends, the clear moves to the start of the next (first case pass %llu)", x.width, x.height, R.area.width, R.area.height, (unsigned long long)g_stat_pass);
    return 1;
}
// The channels a target's format has (mask bits R 1, G 2, B 4, A 8): a clear whose mask covers them all writes the whole texel. The X of RGBX8 and its kind (NVK_NOA) is undefined, so they count as RGB
static unsigned chan_mask(const vkn_img *e) {
    switch (e->fmt) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R32_SFLOAT: return 1;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16G16_SFLOAT: return 3;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return 7;
        default: { const nvk_fmt *m = nvn_fmt_vk((int)e->nvn_fmt); return m && (m->flags & NVK_NOA) ? 7 : 0xf; }
    }
}
static void op_clear_color(const p_clear_color *c) {
    if (c->index < 0 || c->index >= R.n || !R.cser[c->index] || !(R.act >> c->index & 1)) { NRED_ONCE("colour clear of slot %d, which has no target: skipped", c->index); return; }
    vkn_img *e = R.ce[c->index]; if (!e) return;   // no host image: said at SET_RT
    e->fb_ok = 0;
    unsigned all = chan_mask(e); if ((c->mask & all) != all) { NRED_ONCE("colour clear with channel mask %#x of a target with channels %#x: needs a draw (a later slice), skipped", c->mask, all); return; }
    VkExtent2D x = lv_ext(e, R.clv[c->index]);
    if (sc_full(x) && (!R.on || oversize_mid(x))) { R.clear_c[c->index] = 1; memcpy(R.cc[c->index].float32, c->rgba, sizeof c->rgba); return; }   // folded into the next pass's load op
    if (!rp_begin(0) || !R.att_c[c->index]) return;
    VkClearAttachment a = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {{{c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]}}}};
    clear_in_pass(&a, x);
    pthread_mutex_lock(&vkn_tmu); memcpy(e->clear, c->rgba, sizeof e->clear); e->cleared = 1; pthread_mutex_unlock(&vkn_tmu);   // the present thread reads it under the lock
}
static void op_clear_ds(const p_clear_ds *c) {
    vkn_img *e = R.de; if (!R.dser || !e) return;   // no depth target: nothing to clear (none made: said at SET_RT)
    e->fb_ok = 0;
    unsigned sm = (unsigned)c->stencil_mask & 0xff; int cd = c->depth_mask && (e->aspect & VK_IMAGE_ASPECT_DEPTH_BIT), cs = sm && (e->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
    if (cs && sm != 0xff) { NRED_ONCE("depth-stencil clear with stencil mask %#x: needs a draw (a later slice), its stencil part is skipped", sm); cs = 0; }
    if (!cd && !cs) return;
    VkExtent2D x = lv_ext(e, R.dlv);
    float dv = c->depth >= 0.0f ? (c->depth <= 1.0f ? c->depth : 1.0f) : 0.0f;   // Vulkan wants [0, 1] (no VK_EXT_depth_range_unrestricted); NaN goes to 0
    if (sc_full(x) && (!R.on || oversize_mid(x))) { if (cd) { R.clear_d = 1; R.cd = dv; } if (cs) { R.clear_s = 1; R.cs = (uint32_t)c->stencil & 0xff; } return; }
    if (!rp_begin(0) || !R.att_ds) return;
    VkClearAttachment a = {(cd ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (cs ? VK_IMAGE_ASPECT_STENCIL_BIT : 0), 0, {{{0}}}};
    a.clearValue.depthStencil.depth = dv; a.clearValue.depthStencil.stencil = (uint32_t)c->stencil & 0xff;
    clear_in_pass(&a, x);
}
// NVN barrier bits order earlier GPU work before later work (and invalidate caches): one global memory barrier. Inside a pass the pass ends first (a pipeline barrier inside dynamic rendering
// is only for framebuffer-local dependencies); the next draw opens it again with load ops. Clears still recorded stay recorded: they land after the barrier, as on NVN
// Passes kept open (DM_VK_PASS_KEEP=0: every SET_RT and barrier ends the pass, as before). On a tiler each pass loads and stores its whole targets: Paper Mario: The Thousand-Year Door
// opens about 50 passes a frame, 29 on its 1920x1080 composite (RGB10A2 + D24S8), most of them a SET_RT of the same targets or an ordering barrier between draws, and the RG 476H's
// GPU took 54 ms a frame. Inside a pass Vulkan keeps draw and blend order itself, and a texture written earlier is made ready where it is sampled (tex_ready ends the pass then)
static int pass_keep(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_PASS_KEEP"); on = !(e && *e == '0'); if (!on) NLOG("DM_VK_PASS_KEEP=0: every SET_RT and barrier ends the pass"); } return on; }
static uint64_t g_stat_keep_rt, g_stat_keep_bar;   // SET_RTs and barriers that left the pass open
static uint64_t g_split_draws, g_split_n;   // draws since the last part of the frame went to the GPU, and parts sent early (DM_VK_SPLIT, exec_run)
// Whether SET_RT r names the open pass's targets or some of them, each in its slot (colour serials, levels, layers), and the same depth (serial, level, layer): → the colour slots it
// names, else -1. Paper Mario: The Thousand-Year Door's storybook pages switch from their G-buffer (three colour targets and depth) to its depth alone and back about ten times a frame;
// each pass ended there loaded and stored 960x540 x3 and depth again, and the RG 476H's GPU took 45 ms a frame
static int rt_sub(const p_set_rt *r) {
    int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n; unsigned m = 0;
    for (int i = 0; i < n; i++) {
        const tex_t *t = r->color[i]; uint64_t ser = t ? t->serial : 0;
        if (!ser) continue;
        uint32_t lv = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0, ly = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0;
        if (i >= R.n || ser != R.cser[i] || lv != R.clv[i] || ly != R.cly[i]) return -1;
        m |= 1u << i;
    }
    const tex_t *d = r->depth; uint64_t ds = d ? d->serial : 0;
    uint32_t dl = d && r->has_dview && r->dview.set_levels ? r->dview.base_level : 0, dy = d && r->has_dview && r->dview.set_layers ? r->dview.min_layer : 0;
    return ds == R.dser && (!ds || (dl == R.dlv && dy == R.dly)) ? (int)m : -1;
}
static void op_barrier(int32_t bits) {
    if (!bits) return;
    if (R.on && pass_keep() && !(bits & NVN_BARRIER_ORDER_INDIRECT_DATA_BIT)) { g_stat_keep_bar++; return; }
    rp_end();
    mem_barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    g_stat_bar++;
}

// ---------- Sampled textures and samplers (PLAN.md 2.3 "As built, slice 2b"): the registrations, the replay's bound slots, the host images they name and their uploads ----------
// The tables mirror nvn_mtl.m's (mtl_reg_tex / mtl_reg_samp, tex_by_id / samp_by_id): texture ID → the registered texture object and view, sampler ID → its parameters and VkSampler (deduplicated
// by parameters, never destroyed: Stardew Valley makes four). A draw with a program resolves only the slots its shaders sample (slice 2c: draw_view, draw_samp) and pushes them; a dispatch (the
// compute stage's) and a draw with no program resolve every bound slot (tex_bound). Either way only the slots bound or changed since the last resolve, unless something says all (a new submission,
// SET_RT, a registration). An ID not registered, or a texture already finalized, is only counted; a shader that samples one gets the dummy (the draws' line counts dummies).
// A texture's host image is the render target's when the GPU drew it (same serial: one image, its layout tracked from rendered to sampled), else made at its first lookup. A CPU-written texture
// (packaged layout, WriteTexels, a flush: mtl_tex_cpu_written) is uploaded from guest memory: deswizzled on the CPU (nvn_deswizzle.c) into the slot's staging buffer at the commit and copied by the
// upload command buffer ahead of the submission (b_commit, after strategy B's copies). A LINEAR texture in a pool the CPU can write counts as CPU-written from the start: nvn_mtl.m makes it a
// texture over the pool's own memory (tex_linear), so whatever the CPU stores there is what the GPU reads, and the front half leaves it out of the flush and WriteTexels notifications for that
// reason; here it is uploaded and tracked as the others (until the GPU draws it: then what was drawn is sampled). Its pages are then write-protected (after the commit's stamp); at its first use
// in a later submission a guest write since (nvn_dirty_since_span over its pages, at every such use) uploads it again, unless only the pages it shares with other data were written and its own bytes
// there are as uploaded (nvn_mtl.m pkg_edges_clean). A texture in a page no pool may protect (PF_ALW edge pages of two pools, command pages, an untracked pool) is uploaded at every use. A texture
// nothing wrote is filled once from its storage when nvn_mtl.m's tex_init_fill would (a CPU_NO_ACCESS pool's initial content, or storage the CPU flushed before the texture's first use: tex_init_ok),
// else cleared to zero (nvn_mtl.m tex_zero_fill). Jobs (tjob, above b_commit) carry all of it to the commit; the image's layout is SHADER_READ_ONLY_OPTIMAL in the submission from its first lookup
typedef struct { tex_t t; view_t v; uint8_t ok, has_v; } treg_t;
typedef struct { samp_par p; VkSampler s, si; uint8_t ok; } sreg_t;   // si: its integer twin (samp_lookup_as)
static treg_t *g_treg; static sreg_t *g_sreg; static int g_ntreg, g_nsreg, g_tpool_n;   // under g_rmu (a leaf: VkSamplers are made under it, nothing else is taken)
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_reg_gen;                 // moves at each change of a registration, the pool capacity, or a texture becoming CPU-written: every bound slot is resolved again
static nvn_hmap g_samps = {.klen = sizeof(samp_par)}, g_isamps = {.klen = sizeof(samp_par)};   // parameters → VkSampler, and → integer twin (under g_rmu)
enum { TS_UP1, TS_RE, TS_REALW, TS_UP1_B, TS_RE_B, TS_REALW_B, TS_EDGE, TS_ZERO, TS_LAY_UP, TS_LAY_CB, TS_TLOOK, TS_SLOOK, TS_MADE, TS_CPUW, TS_FAIL, TS_FEED, TS_UNREG, TS_STALE, TS_OOB, TS_SMADE,
       TS_INIT, TS_INIT_B, TS_CHK, TS_CHK_NS, TS_N };   // INIT: initial-content fills (tex_init_ok); CHK: the dirty checks of uploaded textures (tex_ready) and their time
static _Atomic uint64_t g_ts[TS_N];
#define TSA(i, n) atomic_fetch_add_explicit(&g_ts[i], (uint64_t)(n), memory_order_relaxed)
#define SR_LAYOUT VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL

// A view's component mapping from the NVN swizzle. Under the portability subset without imageViewFormatSwizzle (VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465) a view must map
// each component to itself: the identity, said once (the RGBX formats' alpha of one and a view's own swizzle are then the shader's to apply, which no slice does yet)
static void sview_comp(const uint8_t sw[4], VkComponentSwizzle c[4]) {
    int id = 1; for (int k = 0; k < 4; k++) { c[k] = sw[k] == 2 + k ? VK_COMPONENT_SWIZZLE_IDENTITY : (VkComponentSwizzle)(sw[k] + 1); id &= c[k] == VK_COMPONENT_SWIZZLE_IDENTITY; }
    if (id || !vd.port || vd.port_swz) return;
    NRED_ONCE("a sampled view swizzles its components (%u %u %u %u), and the portability subset has no imageViewFormatSwizzle: viewed unswizzled (applying it in the shader is not done yet)", sw[0], sw[1], sw[2], sw[3]);
    for (int k = 0; k < 4; k++) c[k] = VK_COMPONENT_SWIZZLE_IDENTITY;
}
// The aspect a sampled view of e has: colour, else the stencil one where asked and the image has it, else depth, else (a stencil-only image) stencil
static VkImageAspectFlags sview_aspect(const vkn_img *e, int stencil) {
    return e->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : stencil && (e->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT
         : e->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
}
// A sampled view of e (under vkn_tmu), kept with the image and destroyed with it: vf one of its formats (image or present), levels l0 + nl, layers s0 + ns, sw the NVN channel swizzle
// (0 zero, 1 one, 2-5 R G B A: Vulkan's ZERO .. A are one more), stencil: the stencil aspect of a depth-stencil image (sview_aspect)
static VkImageView img_sview(vkn_img *e, VkFormat vf, VkImageViewType vt, uint32_t l0, uint32_t nl, uint32_t s0, uint32_t ns, const uint8_t sw[4], int stencil) {
    uint64_t key = (uint64_t)(vf & 0xffff) << 48 | (uint64_t)(l0 & 15) << 44 | (uint64_t)(nl & 31) << 39 | (uint64_t)(s0 & 2047) << 28 | (uint64_t)(ns & 2047) << 17
                 | (uint64_t)(sw[0] | sw[1] << 3 | sw[2] << 6 | sw[3] << 9) << 5 | (uint64_t)(stencil != 0) << 1 | (vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY) | (uint64_t)(vt == VK_IMAGE_VIEW_TYPE_3D) << 2 | (uint64_t)(vt == VK_IMAGE_VIEW_TYPE_CUBE) << 3;
    for (int i = 0; i < e->nsviews; i++) if (e->sviews[i].key == key) return e->sviews[i].v;
    if (e->nsviews == (int)(sizeof e->sviews / sizeof *e->sviews)) { NRED_ONCE("a host image has more than %d sampled views: the rest are sampled as nothing", e->nsviews); return VK_NULL_HANDLE; }
    VkImageAspectFlags a = sview_aspect(e, stencil);
    VkComponentSwizzle c[4]; sview_comp(sw, c);
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, e->img, vt, vf, {c[0], c[1], c[2], c[3]}, {a, l0, nl, s0, ns}};
    VkImageView v; VkResult r = vk.vkCreateImageView(vd.dev, &vi, NULL, &v);
    if (r) { NRED_ONCE("vkCreateImageView of a sampled view failed (VkResult %d, host image ser=%llu): sampled as nothing", r, (unsigned long long)e->serial); return VK_NULL_HANDLE; }
    e->sviews[e->nsviews].key = key; e->sviews[e->nsviews++].v = v;
    return v;
}
// The view a registration samples (under vkn_tmu), as tex_by_id builds it: the view's levels, layers, target and format (the image's own or its present format only: another is a later slice),
// the channel swizzle from the view or else from the texture builder (no alpha for the NVK_NOA formats: a reads as 1, nvn_fmt_to_mtl's), the stencil aspect for a stencil-mode view or texture
// want: the view type a shader declares (VKN_VT_2D / 2D_ARRAY: a 2D view is of the view's first layer), -1 = the view's own (a draw with no program: slice 2b's).
// comp: the numeric class a shader reads the view as (fmt_comp's), from the view's own aspect and format: a stencil aspect is unsigned integers whatever the image's depth half is
static int fmt_comp(VkFormat f);
static VkImageView tex_view(vkn_img *e, const treg_t *r, int tid, int want, int *comp) {
    const view_t *v = r->has_v ? &r->v : NULL; VkFormat vf = e->fmt; int nf = (int)e->nvn_fmt;
    if (!(e->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) { NRED_ONCE("texture ID %d: its host image (format %d) cannot be sampled on this device: sampled as nothing", tid, e->fmt); return VK_NULL_HANDLE; }
    if (v && v->set_format && v->format != r->t.d.format) {
        const nvk_fmt *m = nvn_fmt_vk(v->format); VkFormat f = m ? m->f[0] : 0;
        if (f && (f == e->fmt || f == e->sfmt)) { vf = f; nf = v->format; }
        else NRED_ONCE("texture ID %d viewed as format %#x, its texture's is %#x: sampled in the texture's own (views of another format are a later slice)", tid, v->format, r->t.d.format);
    }
    uint32_t l0 = v && v->set_levels ? v->base_level : 0, s0 = v && v->set_layers ? v->min_layer : 0;
    if (l0 >= e->levels || s0 >= e->layers) { NRED_ONCE("texture ID %d: a view from level %u layer %u, beyond its %u levels / %u layers: sampled as nothing", tid, l0, s0, e->levels, e->layers); return VK_NULL_HANDLE; }
    uint32_t nl = v && v->set_levels && v->nlevels ? v->nlevels : e->levels - l0, ns = v && v->set_layers && v->nlayers ? v->nlayers : e->layers - s0;
    if (nl > e->levels - l0) nl = e->levels - l0;   // cut to the levels it has, as deko3d's descriptor writer clips them (nvn_mtl.m)
    if (ns > e->layers - s0) ns = e->layers - s0;
    int tg = v && v->set_target ? v->target : r->t.d.target;
    int vcube = e->cube && (want == VKN_VT_CUBE || (want < 0 && (tg == NVN_TEXTURE_TARGET_CUBEMAP || tg == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY)));
    if (tg != NVN_TEXTURE_TARGET_2D && tg != NVN_TEXTURE_TARGET_2D_ARRAY && tg != NVN_TEXTURE_TARGET_RECTANGLE && !vcube && tg != NVN_TEXTURE_TARGET_3D)
        NRED_ONCE("texture ID %d viewed with target %d: sampled as 2D (other view targets are a later slice)", tid, tg);
    VkImageViewType vt = want == VKN_VT_2D_ARRAY || (want < 0 && tg == NVN_TEXTURE_TARGET_2D_ARRAY) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if (e->depth > 1 || tg == NVN_TEXTURE_TARGET_3D) { if (want >= 0 && want != VKN_VT_3D) return VK_NULL_HANDLE; vt = VK_IMAGE_VIEW_TYPE_3D; s0 = 0; ns = 1; }   // a 3D image has 3D views only
    else if (want == VKN_VT_3D) return VK_NULL_HANDLE;
    if (vcube) { if (s0 + 6 > e->layers) return VK_NULL_HANDLE; vt = VK_IMAGE_VIEW_TYPE_CUBE; ns = 6; }   // six faces from the view's first layer
    if (vt == VK_IMAGE_VIEW_TYPE_2D) ns = 1;
    uint8_t sw[4] = {2, 3, 4, 5};
    if (v && v->set_swizzle) memcpy(sw, v->swz, 4);
    else if (r->t.set_swz) for (int c = 0; c < 4; c++) sw[c] = (uint8_t)(r->t.swz >> (3 * c) & 7);
    const nvk_fmt *m = nvn_fmt_vk(nf); int noa = m && (m->flags & NVK_NOA);
    for (int c = 0; c < 4; c++) { if (sw[c] > 5) sw[c] = (uint8_t)(2 + c); if (noa && sw[c] == 5) sw[c] = 1; }
    int st = v && v->set_dsmode ? v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL : r->t.stencil;
    if (st && !(e->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) NRED_ONCE("texture ID %d: a stencil-mode view of format %d, which has no stencil: depth is read", tid, e->fmt);
    *comp = sview_aspect(e, st) == VK_IMAGE_ASPECT_STENCIL_BIT ? 1 : fmt_comp(vf);
    return img_sview(e, vf, vt, l0, nl, s0, ns, sw, st);
}

// The bound slots (submit thread): handles as bound per stage (textures 0-127 combined, 128-255 separate, NVN_SEPTEX_SLOT0; separate samplers 0-127), which are bound (non-zero) and which
// changed since the last resolve, and what they resolved to. all: stages whose every bound slot is resolved at the next draw / dispatch
static struct {
    uint64_t tex[6][256], samp[6][128], tb[6][4], td[6][4], sb[6][2], sd[6][2];
    VkImageView tv[6][256]; VkSampler ts[6][128], ss[6][128];
    uint64_t gen; unsigned all;
    // A draw with a program (slice 2c: draw_tex_prep, draw_view, draw_samp) resolves only the slots its shaders sample: tsd the combined slots whose sampler half is still to resolve (td for their views); tvt: the view type
    // and component type + 1 that tv was resolved for (0: slice 2b's own, tex_bound)
    uint64_t tsd[6][2]; uint8_t tvt[6][256];
} T;
static void tex_bind(uint32_t op, const uint8_t *a) {
    p_bind_handles t1; const p_bind_handles *b = op == OP_BIND_SAMP ? (const p_bind_handles *)a : bind_rec(op, a, &t1);
    int lim = op == OP_BIND_SAMP ? 128 : 256, st = b->stage;
    if (st < 0 || st >= 6 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > lim) {   // the recorder writes none of these: a record overwritten after it was recorded
        NRED_ONCE("%s record binds stage %d slots %d + %d, outside the stage's %d: skipped", op == OP_BIND_SAMP ? "sampler" : "texture", st, b->first, b->n, lim); return; }
    uint64_t hd[32]; const uint64_t *h = b->h;
    if (b->defer && b->src && b->n) { memcpy(hd, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); h = hd; }   // BindTexturesDeferred: the handles as they are now (bh_fresh's, into a copy)
    uint64_t *dst = op == OP_BIND_SAMP ? T.samp[st] : T.tex[st], *bm = op == OP_BIND_SAMP ? T.sb[st] : T.tb[st], *dm = op == OP_BIND_SAMP ? T.sd[st] : T.td[st];
    for (int k = 0; k < b->n; k++) {
        int s = b->first + k; uint64_t bit = 1ull << (s & 63);
        if (dst[s] == h[k]) continue;
        dst[s] = h[k];
        int comb = op != OP_BIND_SAMP && s < NVN_SEPTEX_SLOT0;
        if (op != OP_BIND_SAMP) T.tvt[st][s] = 0;
        if (h[k]) { bm[s >> 6] |= bit; dm[s >> 6] |= bit; if (comb) T.tsd[st][s >> 6] |= bit; continue; }
        bm[s >> 6] &= ~bit; dm[s >> 6] &= ~bit; if (comb) T.tsd[st][s >> 6] &= ~bit;   // unbound: nothing to resolve
        if (op == OP_BIND_SAMP) T.ss[st][s] = VK_NULL_HANDLE; else { T.tv[st][s] = VK_NULL_HANDLE; if (comb) T.ts[st][s] = VK_NULL_HANDLE; }
    }
}

// Upload sizes: the linear bytes of each level and layer, as the copy regions take them (rows of whole blocks, tightly packed), each region 16-byte aligned in the staging buffer
static uint64_t tex_bytes(const vkn_img *e) {
    int bw, bh, bpb = nvn_fmt_bpb(e->d.format, &bw, &bh), dbpp; uint64_t n = 0;
    if (!bpb) return 0;
    if (vkn_bc_decoded((int)e->nvn_fmt, &dbpp)) { bw = bh = 1; bpb = dbpp; }   // decoded on the CPU (vk_bc.c): the staging holds the pixels
    for (uint32_t l = 0; l < e->levels; l++) { uint32_t w = e->w >> l ? e->w >> l : 1, h = e->h >> l ? e->h >> l : 1;
        uint32_t dz = e->depth >> l ? e->depth >> l : 1;   // a 3D level's slices, one after another
        n += (((uint64_t)((w + bw - 1) / bw) * (uint64_t)bpb * (uint64_t)((h + bh - 1) / bh) * dz + 15) & ~15ull) * e->layers; }
    return n;
}
static void tj_add(vkn_img *e, VkImageLayout old, int kind) {
    if (g_ntj == g_tjcap) { g_tjcap = g_tjcap ? g_tjcap * 2 : 64; g_tj = realloc(g_tj, (size_t)g_tjcap * sizeof *g_tj); if (!g_tj) die("vkn backend: texture job list allocation failed"); }
    g_tj[g_ntj++] = (tjob){e, old, kind, 0, 0, 0, 0, 0, NULL};
}
// An upload that cannot be made: the image gets what it would have without one (a zero fill if it holds nothing yet, else its layout), and is checked again at its next use
static void tj_fallback(tjob *q) {
    TSA(TS_FAIL, 1); q->nr = 0;
    q->kind = q->old == VK_IMAGE_LAYOUT_UNDEFINED && (q->e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ? TJ_ZERO : q->old != SR_LAYOUT ? TJ_LAYOUT : TJ_NONE;
}
// The replay's part (submit thread, at a draw or dispatch that has it bound): its layout becomes shader-read in this submission. At its first use in the submission, a job for the upload
// command buffer (an upload, a zero fill, the layout change; e->rec is SR from then on); later in the submission after it was a target, a barrier in the submission's command buffer (a pass
// ends first). cpu: its content comes from guest memory. A texture that is a target of the pass being drawn (a feedback loop) is left alone, counted (a draw whose shader samples it says so in red: draw_view).
// t: its registration's texture object (tex_init_ok reads its pool and storage)
static int tex_init_ok(const vkn_img *e, const tex_t *t);
// A draw sampling a target of its own pass (on the Switch it reads what the pass drew so far: Paper Mario: The Thousand-Year Door's pass over its 1920x1080 composite samples it,
// and read as zeros it wrote the frame black): the pass ends, level 0 / layer 0 is copied into the image's twin, and the draw samples the twin. DM_VK_FEEDBACK=0: zeros, as before
// A depth target sampled for its depth gets the same (stencil: none, zeros): Paper Mario: The Thousand-Year Door's decals (decal.bfsha, the characters' shadows) test against the
// 960x540 depth-stencil and sample its depth in the same pass; read as zeros, no decal drew. The twin's view is of the depth aspect. DM_VK_FEEDBACK_DEPTH=0: zeros, as before
// The reflection pass (reflection.bfsha) reads the stencil it tests against the same way: the twin's stencil view.
// The copy is made again only once the part sampled may have changed: a draw that writes it (colour; depth, or stencil, writes for a depth target: draw_fb_mark), a clear or copy into the
// image, a dispatch or buffer-to-texture copy (g_fb_ep), a new frame. The decals write stencil only, so they sample one depth copy and the pass stays open. DM_VK_FEEDBACK_REUSE=0: a copy at each lookup
static uint64_t g_stat_fb[3];   // feedback copies made: colour, depth; copies reused
static VkImageView fb_copy(vkn_img *e, int stencil) {
    static int on = -1, don = -1, reuse = -1; if (on < 0) { const char *v = getenv("DM_VK_FEEDBACK"); on = !(v && *v == '0'); v = getenv("DM_VK_FEEDBACK_DEPTH"); don = !(v && *v == '0');
                                                            v = getenv("DM_VK_FEEDBACK_REUSE"); reuse = !(v && *v == '0'); }
    int ds = e->aspect != VK_IMAGE_ASPECT_COLOR_BIT;
    if (!on || (ds && (!don || !(e->aspect & (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT)))) || !(e->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || e->depth > 1) return VK_NULL_HANDLE;
    pend_resolve(e);
    uint64_t ep = g_fb_ep ^ atomic_load(&g_present_n) << 32; uint8_t part = stencil && ds ? 2 : 1;
    if (reuse && e->fbi && (e->fb_ok & part) && e->fb_ep == ep) { g_stat_fb[2]++; use_mark(e); return stencil && ds ? e->fbsv : e->fbv; }
    { static int st = -1; if (st < 0) { const char *x = getenv("DM_VK_DIAG"); st = x && strstr(x, "fbstale"); }   // DM_VK_DIAG=fbstale (wrong on purpose, a bound): a twin once made is never copied again
      if (st && e->fbi && e->fb_ok) { g_stat_fb[2]++; use_mark(e); return stencil && ds ? e->fbsv : e->fbv; } }
    if (!e->fbi) {
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, e->fmt, {e->w, e->h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkMemoryRequirements mr; int mt;
        if (vk.vkCreateImage(vd.dev, &ii, NULL, &e->fbi)) { e->fbi = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
        vk.vkGetImageMemoryRequirements(vd.dev, e->fbi, &mr);
        if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, 0);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, e->fbi, VK_IMAGE_VIEW_TYPE_2D, e->fmt, {0}, {ds ? e->aspect & VK_IMAGE_ASPECT_DEPTH_BIT : e->aspect, 0, 1, 0, 1}},
                              si = vi; si.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &e->fbm) || vk.vkBindImageMemory(vd.dev, e->fbi, e->fbm, 0) || (vi.subresourceRange.aspectMask && vk.vkCreateImageView(vd.dev, &vi, NULL, &e->fbv)) ||
            ((e->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && vk.vkCreateImageView(vd.dev, &si, NULL, &e->fbsv))) {
            NRED_ONCE("a feedback copy of host image ser=%llu cannot be made: the draw reads zeros", (unsigned long long)e->serial);
            if (e->fbv) vk.vkDestroyImageView(vd.dev, e->fbv, NULL);
            if (e->fbm) vk.vkFreeMemory(vd.dev, e->fbm, NULL); vk.vkDestroyImage(vd.dev, e->fbi, NULL); e->fbi = VK_NULL_HANDLE; e->fbm = VK_NULL_HANDLE; e->fbv = VK_NULL_HANDLE; e->fbsv = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
        static int n; if (n++ < 4) NLOG("feedback copy: host image ser=%llu %ux%u%s gets a twin that the draws sampling their own target read", (unsigned long long)e->serial, e->w, e->h, ds ? " (depth)" : "");
    }
    g_stat_fb[ds]++; e->fb_ok = 3; e->fb_ep = ep;
    rp_end();
    VkImageMemoryBarrier2 b[2] = {
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
         rec_layout(e), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}},
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, SR_STAGES, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->fbi, {e->aspect, 0, 1, 0, 1}}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 2, b};
    vk.vkCmdPipelineBarrier2(cb_now(), &di); e->rec = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkImageCopy r = {{e->aspect, 0, 0, 1}, {0, 0, 0}, {e->aspect, 0, 0, 1}, {0, 0, 0}, {e->w, e->h, 1}};
    vk.vkCmdCopyImage(cb_now(), e->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, e->fbi, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    VkImageMemoryBarrier2 c = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, SR_STAGES, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, SR_LAYOUT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->fbi, {e->aspect, 0, 1, 0, 1}};
    di.pImageMemoryBarriers = &c; di.imageMemoryBarrierCount = 1;
    vk.vkCmdPipelineBarrier2(cb_now(), &di);
    use_mark(e);
    return stencil && ds ? e->fbsv : e->fbv;
}
static int g_in_dispatch;   // op_dispatch is resolving its bindings: no pass is open, so the last pass's targets are no feedback loop
// The stages a target drawn earlier in the submission is made ready for: the one that samples it (st: NVN's stage, 1 fragment, 5 compute, the others before rasterization; -1 all). Mali runs a pass's
// vertex and tiling work beside the previous pass's fragment work; a barrier into the vertex stage after a pass that drew the texture stops that. Paper Mario: The Thousand-Year Door's
// storybook alternates about 20 small blur passes, each sampling the last. A later use from a stage not yet covered adds a barrier for it. DM_VK_SRSTAGE=0: every sampling stage, as before;
// DM_VK_SRSTAGE=ab:<frames>: off and on in alternate phases of that many frames (phase 0 off), to compare both in one run, scene and thermal state
static VkPipelineStageFlags2 sr_stage(int st) {
    static int on = -1; static uint64_t ab;
    if (on < 0) { const char *e = getenv("DM_VK_SRSTAGE"); on = !(e && *e == '0'); if (e && !strncmp(e, "ab:", 3)) ab = strtoull(e + 3, NULL, 10);
                  NLOG("%s", ab ? "DM_VK_SRSTAGE=ab: sampling-stage readiness alternates off / on by phase" : on ? "targets sampled later are made ready for the sampling stage only (DM_VK_SRSTAGE=0: all)" : "DM_VK_SRSTAGE=0: targets sampled later are made ready for every sampling stage"); }
    int now = ab ? (int)(atomic_load(&g_present_n) / ab & 1) : on;
    return !now || st < 0 ? SR_STAGES : st == 1 ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : st == 5 ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT;
}
static uint64_t g_stat_sr_more, g_stat_sr_split;   // barriers added for a stage the first one did not cover, and those of them that ended an open pass
static const char *g_tl_from;     // DM_VK_DIAG=sr: which lookup asked (draw_view, draw_dyn, draw_texu, tex_bound)
static int tex_ready(vkn_img *e, int cpu, uintptr_t ts, const tex_t *t, int prog, int st) {   // 0 = a target of the pass: not sampled (prog: a draw with a program samples it, which says it)
    int tgt = !g_in_dispatch && R.de == e; for (int i = 0; i < R.n && !g_in_dispatch; i++) tgt |= R.ce[i] == e;
    if (tgt) {
        if (!TSA(TS_FEED, 1) && !prog) NLOG("texture ser=%llu is bound while it is a target of the pass (first case): left as the target (a draw whose shader samples it is said in red)", (unsigned long long)e->serial);
        return 0;
    }
    if (e->rec_sub == g_sub_id) {
        VkPipelineStageFlags2 need = sr_stage(st);
        if (e->rec == SR_LAYOUT && (e->sr_st & need) == need) return 1;
        int was_on = R.on; rp_end();
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, need, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            e->rec, SR_LAYOUT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &b};
        vk.vkCmdPipelineBarrier2(cb_now(), &di);
        if (e->rec == SR_LAYOUT) {
            { static int dg = -1, n; if (dg < 0) { const char *x = getenv("DM_VK_DIAG"); dg = x && strstr(x, "sr"); }
              if (dg && n++ < 40) NLOG("sr: ser=%llu %ux%u had stages %#llx, stage %d (%s) asks %#llx, pass %s, frame %llu", (unsigned long long)e->serial, e->w, e->h, (unsigned long long)e->sr_st, st, g_tl_from ? g_tl_from : "?", (unsigned long long)need, was_on ? "open" : "closed", (unsigned long long)atomic_load(&g_present_n)); }
            e->sr_st |= need; g_stat_sr_more++; g_stat_sr_split += was_on; } else e->sr_st = need;
        e->rec = SR_LAYOUT; TSA(TS_LAY_CB, 1);
        return 1;
    }
    VkImageLayout old = rec_layout(e); int kind = TJ_NONE;
    if (cpu) {
        const nvk_fmt *m = nvn_fmt_vk((int)e->nvn_fmt);
        if (ts & 2) NRED_ONCE("texture ser=%llu is both drawn by the GPU and written by the CPU: uploaded from guest memory over what was drawn, as nvn_mtl.m does", (unsigned long long)e->serial);
        if (!m || !(m->flags & NVK_UP) || !(e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !e->gmem)
            NRED_ONCE("CPU-written texture ser=%llu format %#x cannot be uploaded (format %d, usage %#x): what the GPU had is sampled", (unsigned long long)e->serial, e->nvn_fmt, e->fmt, e->usage);
        else if (e->once) kind = TJ_NONE;   // read once from a CPU-no-access pool (tex_prep_all)
        else if (!e->up) kind = TJ_UP1;
        else if (e->alw) kind = TJ_REALW;
        else {
            // Every use checks the pages (no "the generation has not moved since the last check" shortcut, as nvn_mtl.m's pkg path has). nvn_dirty.c's unprotect moves the generation
            // (atomic_fetch_add on g_now) before it stores the page's generation and sets it writable, so a check in between reads the new generation and finds the page clean; a cached
            // "nothing written up to that generation" would then hide the write that follows for good (it does not fault again, and the generation need not move again). Without the
            // shortcut: the write that matters to this submission comes after its unprotect returned (the faulting store re-executes then), so after the page's generation was stored,
            // and before the guest submitted; this check comes after that submission, so it sees the generation, which stays above e->stamp until the next upload. A write after this
            // check is for a later submission, whose own check sees it. Cost: one atomic load per page of the texture, once per submission it is used in (TS_CHK, the texture line)
            uint64_t c0 = now_ns();
            if (nvn_dirty_since_span(e->span, e->gmem, (size_t)e->size, e->stamp)) kind = e->edge_n[0] || e->edge_n[1] ? TJ_EDGE : TJ_RE;
            TSA(TS_CHK, 1); TSA(TS_CHK_NS, now_ns() - c0);
        }
    } else if (old == VK_IMAGE_LAYOUT_UNDEFINED && !e->up && tex_init_ok(e, t)) kind = TJ_INIT;
    if (!kind) kind = old == VK_IMAGE_LAYOUT_UNDEFINED ? (e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT ? TJ_ZERO : TJ_LAYOUT) : old != SR_LAYOUT ? TJ_LAYOUT : TJ_NONE;
    if (kind) tj_add(e, old, kind);
    e->rec = SR_LAYOUT; e->sr_st = SR_STAGES;
    return 1;
}
// The one-time fill from initial content (nvn_mtl.m tex_init_fill, the same tests): a texture nothing wrote, in a pool the CPU cannot access (what the storage held at
// nvnMemoryPoolInitialize is its content: the game cannot write it again), or in one it can whose storage the CPU flushed (nvnMemoryPoolFlushMappedRange, hle_nvn.c nvn_flushed) before the
// texture's first use, when the GPU has not written it yet (its image still holds nothing: tex_ready's caller). Not for a packaged layout, a sparse pool, multisample, COMPRESSIBLE (its bytes
// mean nothing without the compression tags) or depth / stencil, nor for storage that is all zeros (the zero fill does the same). DM_NVN_FLUSH_FILL=0 leaves the flushed ones out, as there
int nvn_flushed(const void *p, uint64_t n);   // hle_nvn.c
static int mem_any(const uint8_t *p, uint64_t n) {   // whether any byte is non-zero (stops at the first)
    uint64_t i = 0, v;
    for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) return 1; }
    for (; i < n; i++) if (p[i]) return 1;
    return 0;
}
static int tex_init_no(const vkn_img *e, const char *why) {   // a texture left to the zero fill: why (the first cases)
    static int n; if (n++ < 6) NLOG("initial content: texture ser=%llu %ux%u format %#x is zero-filled: %s (first cases)", (unsigned long long)e->serial, e->w, e->h, e->nvn_fmt, why);
    return 0;
}
static int tex_init_ok(const vkn_img *e, const tex_t *t) {
    if (!t->mem || !t->pool || !e->gmem) return tex_init_no(e, "no storage or pool");
    if (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) return tex_init_no(e, "a virtual (sparse) pool");
    if (t->has_layout) return tex_init_no(e, "a packaged layout");
    if (t->d.samples > 1 || (t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) || e->aspect != VK_IMAGE_ASPECT_COLOR_BIT) return tex_init_no(e, "multisample, COMPRESSIBLE or depth / stencil");
    int no_cpu = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) != 0;
    static int flush_fill = -1; if (flush_fill < 0) { const char *v = getenv("DM_NVN_FLUSH_FILL"); flush_fill = !v || atoi(v) != 0; }
    if (!no_cpu && (!flush_fill || !nvn_flushed(e->gmem, e->size))) return tex_init_no(e, "a CPU-accessible pool whose range the CPU never flushed");
    if (!mem_any(e->gmem, e->size)) return tex_init_no(e, "its storage is all zeros");
    const nvk_fmt *m = nvn_fmt_vk((int)e->nvn_fmt);
    if (!m || !(m->flags & NVK_UP) || !(e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        NRED_ONCE("texture ser=%llu format %#x has initial content (%s) this slice cannot upload: cleared to zero instead", (unsigned long long)e->serial, e->nvn_fmt, no_cpu ? "a CPU_NO_ACCESS pool's" : "flushed by the CPU");
        return 0; }
    static uint64_t n[2]; if (n[no_cpu]++ < 4) NLOG("initial content: texture ser=%llu %ux%u format %#x filled from its storage, %s (first cases)", (unsigned long long)e->serial, e->w, e->h, e->nvn_fmt,
                                                    no_cpu ? "a CPU_NO_ACCESS pool's" : "flushed by the CPU before its first use");
    return 1;
}
// Texture ID → its sampled view (submit thread), the image made at the first lookup (one image per serial, the render target's if the GPU drew it), its job given (tex_ready).
// want / comp: the view type and component type the shader declares (-1: a draw with no program, the view's own type); why: what an empty answer means when the texture exists
// (1 a target of the pass, 2 a cube the shader wants and no host image is one, 3 a component type the image's format has not)
enum { TL_FEED = 1, TL_CUBE, TL_COMP, TL_3D };
// What the last tex_lookup found (the red lines of draw_view say it): its fields kept, formatted by tl_what only when a line is written (a snprintf at every lookup
// was 3.5% of TTYD's game thread on the Mac)
static struct { int ok; unsigned long long ser; uint32_t w, h, nvn_fmt; int fmt, aspect, vc; } g_tlw;
static const char *tl_what(void) {
    static char b[128]; b[0] = 0;
    if (g_tlw.ok) snprintf(b, sizeof b, "ser=%llu %ux%u NVN format %#x, Vulkan %d, aspect %#x, view class %d", g_tlw.ser, g_tlw.w, g_tlw.h, g_tlw.nvn_fmt, g_tlw.fmt, g_tlw.aspect, g_tlw.vc);
    return b;
}
static VkImageView tex_lookup(int tid, int want, int comp, int *why, int st) {
    if (why) *why = 0;
    g_tlw.ok = 0;
    if (!tid) return VK_NULL_HANDLE;
    TSA(TS_TLOOK, 1);
    pthread_mutex_lock(&g_rmu);
    if (g_tpool_n && tid >= g_tpool_n) {   // an empty slot (a game binds GetSeparateTextureHandle(0xFFFFF) to unused slots, nvn_mtl.m tex_by_id): reads 0
        int np = g_tpool_n; pthread_mutex_unlock(&g_rmu);
        if (!TSA(TS_OOB, 1)) NLOG("texture ID %d outside the pool (capacity %d) bound: an empty slot", tid, np);
        return VK_NULL_HANDLE; }
    if (tid < 0 || tid >= g_ntreg || !g_treg[tid].ok) { pthread_mutex_unlock(&g_rmu); if (!TSA(TS_UNREG, 1)) NLOG("texture ID %d bound but not registered (first case): an empty slot", tid); return VK_NULL_HANDLE; }
    treg_t r = g_treg[tid];
    pthread_mutex_unlock(&g_rmu);
    uintptr_t ts = tser_get(r.t.serial);
    // LINEAR in a pool the CPU can write: CPU-written from the start (see the section's head), unless the GPU drew it (its drawing is the content: no GPU-to-CPU path in strategy B)
    int lin = (r.t.d.flags & NVN_TEXTURE_FLAGS_LINEAR) && r.t.mem && r.t.pool && !(r.t.pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    if (lin && (ts & 2) && !(ts & 4))
        NRED_ONCE("linear texture ser=%llu was drawn by the GPU and is sampled as drawn: CPU stores into its storage after that are not seen (nvn_mtl.m's texture over the pool memory shows both)", (unsigned long long)r.t.serial);
    int cpu = (r.t.has_layout || (ts & 4) || (lin && !(ts & 2))) && r.t.mem;
    VkImageView v = VK_NULL_HANDLE;
    pthread_mutex_lock(&vkn_tmu);
    vkn_img *e = vkn_img_get(r.t.serial);
    if (!e && ts && (e = img_make(&r.t, "a sampled texture", cpu ? NVK_UP : NVK_UP | NVK_RT))) TSA(TS_MADE, 1);
    if (e && want == VKN_VT_CUBE && !e->cube) { if (why) *why = TL_CUBE; e = NULL; }   // the shader wants a cube, the image cannot be viewed as one
    if (e && want == VKN_VT_3D && e->depth <= 1 && r.t.d.target != NVN_TEXTURE_TARGET_3D) { if (why) *why = TL_3D; e = NULL; }   // the shader wants 3D, the texture is not
    int vc = 0;
    if (e) { use_mark(e); v = tex_view(e, &r, tid, want, &vc); g_tlw.ok = 1; g_tlw.ser = (unsigned long long)e->serial; g_tlw.w = e->w; g_tlw.h = e->h; g_tlw.nvn_fmt = e->nvn_fmt; g_tlw.fmt = e->fmt; g_tlw.aspect = e->aspect; g_tlw.vc = vc; }
    // A stencil view the shader declares as floats: the translator has no integer texture type and the shader takes the raw bits, as nvn_mtl.m binds Metal's stencil-only view.
    // Paper Mario: The Thousand-Year Door's reflection pass (reflection.bfsha) reads its mask so. A uint view bound to a float image is undefined in Vulkan: on the RG 476H (Mali-G57) it
    // greyed Rogueport's walkway (screen luminance 158 -> 137) and drew no reflection, for ~2 ms a frame. Opt-in, DM_VK_STENCIL_RAW=1; by default such a slot reads zeros, as before
    static int sraw = -1; if (sraw < 0) { const char *x = getenv("DM_VK_STENCIL_RAW"); sraw = x && *x == '1'; }
    int raw = sraw && v && vc == 1 && comp == 0 && e && (e->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
    if (raw) { static int n; if (n++ < 2) NLOG("texture ID %d: a stencil view sampled as floats is bound as it is (the shader reads the stencil value's bits) (first cases)", tid); }
    if (v && want >= 0 && vc != comp && !raw) { if (why) *why = TL_COMP; v = VK_NULL_HANDLE; }   // the view's class (a stencil view reads unsigned integers), not the image format's
    pthread_mutex_unlock(&vkn_tmu);
    if (!e && !ts && !TSA(TS_STALE, 1)) NLOG("texture ID %d holds texture ser=%llu, already finalized (first case): an empty slot", tid, (unsigned long long)r.t.serial);
    if (v && e->pc) pend_resolve(e);   // a deferred clear lands before it is sampled
    if (v && !tex_ready(e, cpu, ts, &r.t, want >= 0, st)) { VkImageView f = want >= 0 ? fb_copy(e, vc == 1) : VK_NULL_HANDLE; if (!f && why) *why = TL_FEED; v = f; }   // its own pass's target: the twin (fb_copy), else zeros
    return v;
}
// Sampler parameters → a VkSampler (under g_rmu), as samp_by_id maps them: GL's filters (a min filter without mips: the base level, minLod 0 maxLod 0.25 so a minified sample stays minified),
// the wrap modes (clamp as clamp-to-edge, the mirror-clamps as mirror-clamp-to-edge and clamp-to-border), the LOD clamp and bias (clamped to the device's limit), anisotropy, the compare function
// (NVN's are Vulkan's plus one), min / max reduction, and the border colour where a wrap uses it: one of Vulkan's three when it is exactly that, else the nearest (nvn_mtl.m's choice, said)
// The LOD bias, clamped to the device's limit; 0 under the portability subset without samplerMipLodBias (VUID-VkSamplerCreateInfo-samplerMipLodBias-04467), said once
static float samp_bias(float b) {
    float mb = vd.props.limits.maxSamplerLodBias, r = b > mb ? mb : b < -mb ? -mb : b == b ? b : 0;
    if (r != 0 && vd.port && !vd.port_bias) { NRED_ONCE("a sampler with a LOD bias of %g, and the portability subset has no samplerMipLodBias: no bias", b); r = 0; }
    return r;
}
static VkSampler samp_make(const samp_par *p, int integer) {   // integer: an integer twin's (samp_lookup_as): its border colour is the integer one of the same three
    static const VkSamplerAddressMode wm[8] = {VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    if (atomic_load(&g_ts[TS_SMADE]) >= 4000) { NRED_ONCE("4000 samplers made (the least maxSamplerAllocationCount Vulkan allows): later parameters get none"); return VK_NULL_HANDLE; }
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSamplerAddressMode w[3]; const uint8_t ws[3] = {p->ws, p->wt, p->wr}; int border = 0;
    for (int i = 0; i < 3; i++) {
        if (ws[i] > 7) { NRED_ONCE("sampler wrap mode %u unknown: repeat", ws[i]); w[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT; } else w[i] = wm[ws[i]];
        if (w[i] == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE && !vd.mirror) { NRED_ONCE("a sampler wraps by mirror-clamp-to-edge, which the device lacks: mirrored repeat"); w[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; }
        border |= w[i] == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    }
    if (p->minf > 5 || p->magf > 1) NRED_ONCE("sampler filters min %u mag %u: one is unknown, taken as nvn_mtl.m does (nearest, linear mips)", p->minf, p->magf);
    ci.magFilter = p->magf == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = p->minf == 1 || p->minf == 3 || p->minf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = p->minf >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = w[0]; ci.addressModeV = w[1]; ci.addressModeW = w[2];
    float lo = p->lod_min > 0 ? p->lod_min : 0, hi = p->lod_max >= lo ? p->lod_max : lo;   // NaN: 0 and lo
    if (p->minf <= 1) { lo = 0; hi = 0.25f; }
    { static int tl = -1; if (tl < 0) { const char *x = getenv("DM_VK_DIAG"); tl = x && strstr(x, "texlod"); }   // DM_VK_DIAG=texlod (wrong on purpose, a bound on texture bandwidth): mips from level 4
      if (tl && hi > 4) lo = 4; }
    ci.minLod = lo; ci.maxLod = hi;
    ci.mipLodBias = samp_bias(p->lod_bias);
    static float cap = -1; if (cap < 0) { const char *x = getenv("DM_VK_ANISO"); cap = x && *x ? (float)atof(x) : 16; }   // DM_VK_ANISO=<n>: anisotropy at most n (1: none)
    if (p->aniso > 1 && cap > 1) {
        float ma = vd.props.limits.maxSamplerAnisotropy > cap ? cap : vd.props.limits.maxSamplerAnisotropy;
        if (vd.aniso && ma > 1) { ci.anisotropyEnable = VK_TRUE; ci.maxAnisotropy = p->aniso < ma ? p->aniso : ma; }
        else NRED_ONCE("an anisotropic sampler (%g), and the device has no anisotropic filtering: filtered without", p->aniso);
    }
    if (p->cmp_mode) {
        if (p->cmp_func >= 1 && p->cmp_func <= 8) { ci.compareEnable = VK_TRUE; ci.compareOp = (VkCompareOp)(p->cmp_func - 1); }
        else NRED_ONCE("sampler compare function %u unknown: no compare", p->cmp_func);
    }
    VkSamplerReductionModeCreateInfo rc = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL, p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX};
    if (p->reduction == 1 || p->reduction == 2) {
        if (!vd.minmax) NRED_ONCE("a min / max reduction sampler, and the device has no reduction: averaged");
        else if (ci.compareEnable) NRED_ONCE("a min / max reduction sampler with a depth compare, which Vulkan has not: averaged");
        else ci.pNext = &rc;
    } else if (p->reduction) NRED_ONCE("sampler reduction %u unknown: averaged", p->reduction);
    if (border) {
        const float *b = p->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else { NRED_ONCE("sampler border colour %g %g %g %g is none of Vulkan's three: the nearest (nvn_mtl.m's choice)", b[0], b[1], b[2], b[3]);
               ci.borderColor = b[3] > 0.5f ? (b[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK) : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; }
        if (integer) ci.borderColor = (VkBorderColor)(ci.borderColor + 1);   // FLOAT_x → INT_x (Vulkan's order: each float colour, then its integer one)
    }
    VkSampler s; VkResult r = vk.vkCreateSampler(vd.dev, &ci, NULL, &s);
    if (r) { NRED_ONCE("vkCreateSampler failed (VkResult %d): textures sampled with these parameters get no sampler", r); return VK_NULL_HANDLE; }
    uint64_t n = TSA(TS_SMADE, 1);
    if (n < 16) NLOG("sampler %llu: min filter %u mag %u, wrap %u %u %u, LOD %g..%g bias %g, anisotropy %g, compare %u/%u, reduction %u, border %g %g %g %g%s", (unsigned long long)n + 1, p->minf, p->magf,
                     p->ws, p->wt, p->wr, p->lod_min, p->lod_max, p->lod_bias, p->aniso, p->cmp_mode, p->cmp_func, p->reduction, p->border[0], p->border[1], p->border[2], p->border[3],
                     integer ? " (an integer twin)" : "");
    return s;
}
// Sampler ID → its VkSampler (submit thread), made at the first lookup of its parameters. integer: for a binding the shader reads as integers (a stencil view, an integer format, their dummies):
// the sampler's nearest twin, made beside it. Integer formats never have linear filtering or min / max reduction (VUID-vkCmdDraw-magFilter-04553, -None-06479 and the like), and a usampler
// has no depth compare, so the twin has nearest filters and mips, no anisotropy, reduction or compare, and the integer border colour; wraps, LOD clamp and bias stay. nvn_mtl.m has no twin:
// Metal samples an integer texture unfiltered whatever the sampler says
static VkSampler samp_lookup_as(int sid, int integer) {
    if (!sid) return VK_NULL_HANDLE;
    TSA(TS_SLOOK, 1);
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE;
    if (sid < 0 || sid >= g_nsreg || !g_sreg[sid].ok) { pthread_mutex_unlock(&g_rmu); if (!TSA(TS_UNREG, 1)) NLOG("sampler ID %d bound but not registered (first case): no sampler", sid); return s; }
    sreg_t *r = &g_sreg[sid];
    if (!integer && !(s = r->s)) {
        if (!(s = (VkSampler)(uintptr_t)nvn_hm_get(&g_samps, &r->p)) && (s = samp_make(&r->p, 0))) nvn_hm_put(&g_samps, &r->p, (void *)(uintptr_t)s);
        r->s = s;
    } else if (integer && !(s = r->si)) {
        samp_par q = r->p; q.magf = NVN_MAG_FILTER_NEAREST; q.minf = q.minf <= 1 ? NVN_MIN_FILTER_NEAREST : NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST; q.aniso = 1; q.reduction = 0; q.cmp_mode = 0; q.cmp_func = 0;
        if (!(s = (VkSampler)(uintptr_t)nvn_hm_get(&g_isamps, &q)) && (s = samp_make(&q, 1))) nvn_hm_put(&g_isamps, &q, (void *)(uintptr_t)s);
        r->si = s;
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
static VkSampler samp_lookup(int sid) { return samp_lookup_as(sid, 0); }
// A draw (compute 0: stages 0-4) or dispatch (compute 1: stage 5): the slots bound or changed since the last resolve, every bound one where something said all
static void tex_bound(int compute) {
    g_tl_from = "tex_bound";
    uint64_t g = atomic_load_explicit(&g_reg_gen, memory_order_acquire);
    if (g != T.gen) { T.gen = g; T.all = 0x3f; }
    for (int st = compute ? 5 : 0, se = compute ? 6 : 5; st < se; st++) {
        int all = T.all >> st & 1; T.all &= ~(1u << st);
        for (int w = 0; w < 4; w++) {
            uint64_t m = T.tb[st][w] & (all ? ~0ull : T.td[st][w]); T.td[st][w] = 0;
            if (w < 2) T.tsd[st][w] &= ~m;
            while (m) { int s = w * 64 + __builtin_ctzll(m); m &= m - 1; uint64_t h = T.tex[st][s];
                T.tv[st][s] = tex_lookup((int)(h & 0xfffff), -1, 0, NULL, st); T.tvt[st][s] = 0;
                if (s < NVN_SEPTEX_SLOT0) T.ts[st][s] = samp_lookup((int)(h >> 20)); }
        }
        for (int w = 0; w < 2; w++) {
            uint64_t m = T.sb[st][w] & (all ? ~0ull : T.sd[st][w]); T.sd[st][w] = 0;
            while (m) { int s = w * 64 + __builtin_ctzll(m); m &= m - 1; T.ss[st][s] = samp_lookup((int)(T.samp[st][s] >> 20)); }
        }
    }
}

// ---- The commit's part (b_commit, under g_pmu): pools, protection, the staging bytes; then the upload command buffer's barriers, copies and fills ----
static VkBufferImageCopy *g_tcr; static int g_ntcr, g_tcrcap;   // the copy regions (submit thread)
// The bytes at each end of e's range in pages it shares with other data (nvn_mtl.m pkg_edge_ranges): r[0] before the first whole page, r[1] after the last; ia..ie the whole pages between (none:
// the range is one edge)
static void edge_ranges(const vkn_img *e, uintptr_t r[2][2], uintptr_t *ia, uintptr_t *ie) {
    uintptr_t a = (uintptr_t)e->gmem, z = a + e->size;
    *ia = (a + g_pgsz - 1) & ~(g_pgsz - 1); *ie = z & ~(g_pgsz - 1);
    if (*ie <= *ia) { r[0][0] = a; r[0][1] = z; r[1][0] = r[1][1] = 0; *ia = *ie = 0; return; }
    r[0][0] = a; r[0][1] = *ia; r[1][0] = *ie; r[1][1] = z;
}
static void edges_take(vkn_img *e, uint64_t stamp) {   // right after an upload with this stamp
    for (int i = 0; i < 2; i++) { free(e->edge[i]); e->edge[i] = NULL; e->edge_n[i] = 0; }
    if (e->alw || !e->span) return;
    uintptr_t r[2][2], ia, ie; edge_ranges(e, r, &ia, &ie);
    for (int i = 0; i < 2; i++) if (r[i][1] > r[i][0]) {
        e->edge_n[i] = (uint32_t)(r[i][1] - r[i][0]); if (!(e->edge[i] = malloc(e->edge_n[i]))) die("vkn backend: texture edge copy allocation failed");
        memcpy(e->edge[i], (const void *)r[i][0], e->edge_n[i]); }
    // A write after the stamp may have reached the copy but not the upload: no copy to trust then, and the next write uploads again
    if (nvn_dirty_since_span(e->span, e->gmem, (size_t)e->size, stamp)) for (int i = 0; i < 2; i++) { free(e->edge[i]); e->edge[i] = NULL; e->edge_n[i] = 0; }
}
static int edges_clean(const vkn_img *e) {   // after the protect: no whole page written since the upload, and its bytes in the shared pages as uploaded
    uintptr_t r[2][2], ia, ie; edge_ranges(e, r, &ia, &ie);
    if (ie > ia && nvn_dirty_since_span(e->span, (const void *)ia, ie - ia, e->stamp)) return 0;
    for (int i = 0; i < 2; i++) if (e->edge_n[i] && memcmp(e->edge[i], (const void *)r[i][0], e->edge_n[i])) return 0;
    return 1;
}
// Each upload's pool, its protection and its share of the staging buffer (from off, after the buffers'); edge checks decided here, after the protect. Returns the end of the textures' bytes
static uint64_t tex_prep_all(uint64_t off, uint64_t stamp) {
    for (int j = 0; j < g_ntj; j++) {
        tjob *q = &g_tj[j]; vkn_img *e = q->e;
        if ((q->kind < TJ_UP1 || q->kind > TJ_EDGE) && q->kind != TJ_INIT) continue;
        uintptr_t a = (uintptr_t)e->gmem; int k = -1;
        for (int i = 0; k < 0 && i < g_npool; i++) if (in_pool(i, a, e->size)) k = i;
        if (q->kind == TJ_INIT) {   // read once, not tracked: only its pool must still be there (under g_pmu: it cannot go before tex_fill_all has read it)
            if (k < 0) { NRED_ONCE("texture ser=%llu: its storage %#llx + %llu is in no memory pool: its initial content is not filled", (unsigned long long)e->serial, (unsigned long long)a, (unsigned long long)e->size); tj_fallback(q); continue; }
            q->bytes = tex_bytes(e); q->off = off; off += q->bytes;
            if (!q->bytes) { NRED_ONCE("texture ser=%llu format %#x: no block size known: not uploaded", (unsigned long long)e->serial, e->nvn_fmt); tj_fallback(q); }
            continue;
        }
        // A packaged texture (its storage holds the GPU layout from the game's files) in a CPU-no-access pool: its content is what the storage held at the pool's creation, which the
        // CPU cannot write again: read once, as an initial-content fill, and never checked again (nvn_mtl.m's packaged first uploads). Paper Mario: The Thousand-Year Door's materials
        // (BC7) are such textures: refused here they were zero, every alpha-tested G-buffer pixel was discarded and the frame stayed black. DM_VK_NOACC_UP=0: refused as before
        static int noacc_up = -1; if (noacc_up < 0) { const char *v = getenv("DM_VK_NOACC_UP"); noacc_up = !(v && *v == '0'); }
        if (k >= 0 && g_pools[k].noacc && q->kind == TJ_UP1 && noacc_up) {
            q->kind = TJ_INIT; e->once = 1; q->bytes = tex_bytes(e); q->off = off; off += q->bytes;
            if (!q->bytes) { NRED_ONCE("texture ser=%llu format %#x: no block size known: not uploaded", (unsigned long long)e->serial, e->nvn_fmt); tj_fallback(q); }
            continue;
        }
        if (k < 0 || !g_pools[k].pf || g_pools[k].noacc) {
            NRED_ONCE("texture ser=%llu: its storage %#llx + %llu is in %s: not uploaded", (unsigned long long)e->serial, (unsigned long long)a, (unsigned long long)e->size, k < 0 ? "no memory pool" : "a CPU-no-access pool");
            tj_fallback(q); continue; }
        pool_ent *p = &g_pools[k];
        pool_span(p);   // before any protect (mtl_pool_del keys on it); found here first: the pool's buffer pages NEW as in b_commit
        uintptr_t i0 = (a - p->lo) >> g_sh, i1 = (a + e->size - 1 - p->lo) >> g_sh;
        // Never protected: an edge page of two pools (mtl_pool_add), a page of commands, a pool the tracker does not cover or a range it finds elsewhere (b_commit's rules); uploaded at every use
        uint8_t ef = p->pf[i0] | p->pf[i1];
        const char *why = !p->span ? "its pool is not tracked" : ef & PF_ALW ? "it is in an edge page of two pools" : ef & PF_CMD ? "it shares a page with commands"
                        : nvn_dirty_span(e->gmem, (size_t)e->size) != p->span ? "the tracker finds it in another range" : NULL;
        if (!why && !nvn_dirty_protect_wait(e->gmem, (size_t)e->size)) why = "a protect did not hold";   // said there too: uploaded now and at every use until a protect holds
        if (why && !e->alw) { static int n; if (n++ < 8) NLOG("texture ser=%llu %ux%u at %#llx + %llu B is never protected (%s): uploaded at every use (first cases)", (unsigned long long)e->serial, e->w, e->h,
                                                              (unsigned long long)a, (unsigned long long)e->size, why); }
        int alw = why != NULL;
        q->alw = alw; q->span = p->span;
        if (q->kind == TJ_EDGE) {
            if (!alw && edges_clean(e)) { e->stamp = stamp; TSA(TS_EDGE, 1); q->kind = q->old == SR_LAYOUT ? TJ_NONE : TJ_LAYOUT; continue; }
            q->kind = TJ_RE;
        }
        q->bytes = tex_bytes(e); q->off = off; off += q->bytes;
        if (!q->bytes) { NRED_ONCE("texture ser=%llu format %#x: no block size known: not uploaded", (unsigned long long)e->serial, e->nvn_fmt); tj_fallback(q); }
    }
    return off;
}
// The deswizzle into the staging buffer, one copy region per level and layer (a LINEAR texture's rows are copied as they are, by its stride)
static uint8_t *g_bcs; static size_t g_bcs_cap;   // the deswizzled blocks of a BC level decoded on the CPU
static void tex_fill_all(uint8_t *stg) {
    g_ntcr = 0; int dbpp;
    for (int j = 0; j < g_ntj; j++) {
        tjob *q = &g_tj[j]; vkn_img *e = q->e;
        if (!tj_copies(q->kind)) continue;
        int bw, bh, bpb = nvn_fmt_bpb(e->d.format, &bw, &bh), lin = (e->d.flags & NVN_TEXTURE_FLAGS_LINEAR) != 0, bad = 0; uint64_t o = q->off;
        q->r0 = g_ntcr;
        for (uint32_t l = 0; l < e->levels && !bad; l++) for (uint32_t y = 0; y < e->layers && !bad; y++) {
            uint32_t w = e->w >> l ? e->w >> l : 1, h = e->h >> l ? e->h >> l : 1; size_t pitch = (size_t)((w + bw - 1) / bw) * (size_t)bpb, rows = (h + bh - 1) / bh;
            if (lin) {   // one level, one layer, rows stride bytes apart; stride 0 (the builder's default) is nvn_tex_layout's pitch: the row's bytes rounded up to 32 (128 for a linear render target)
                uint64_t al = (e->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) ? 128 : 32, sd = e->d.stride ? e->d.stride : ((uint64_t)pitch + al - 1) & ~(al - 1);
                if (l || y || sd < pitch || sd * (rows - 1) + pitch > e->size) bad = 1;
                else for (size_t i = 0; i < rows; i++) memcpy(stg + o + i * pitch, e->gmem + i * sd, pitch);
            } else if (e->depth > 1) {   // 3D: the level's slices at once (nvn_deswizzle), slice k at k x pitch x rows
                uint32_t dz = e->depth >> l ? e->depth >> l : 1;
                if (vkn_bc_decoded((int)e->nvn_fmt, &dbpp)) bad = 1;   // BC 3D textures decoded on the CPU: a later slice
                else bad = nvn_deswizzle(&e->d, l, 0, e->gmem, stg + o, pitch) != 0;
                if (!bad) { if (g_ntcr == g_tcrcap) { g_tcrcap = g_tcrcap ? g_tcrcap * 2 : 64; g_tcr = realloc(g_tcr, (size_t)g_tcrcap * sizeof *g_tcr); if (!g_tcr) die("vkn backend: copy region list allocation failed"); }
                    g_tcr[g_ntcr++] = (VkBufferImageCopy){o, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1}, {0, 0, 0}, {w, h, dz}};
                    o += ((uint64_t)pitch * rows * dz + 15) & ~15ull; }
                continue;
            } else if (vkn_bc_decoded((int)e->nvn_fmt, &dbpp)) {   // BC without device support (vk_bc.c): the blocks deswizzled into a scratch buffer, decoded into the staging
                size_t bp = pitch; pitch = (size_t)w * (size_t)dbpp; rows = h;
                if (g_bcs_cap < bp * ((h + 3) / 4)) { g_bcs_cap = bp * ((h + 3) / 4); free(g_bcs); if (!(g_bcs = malloc(g_bcs_cap))) die("vkn backend: BC scratch allocation failed"); }
                bad = nvn_deswizzle(&e->d, l, y, e->gmem, g_bcs, bp) != 0 || vkn_bc_decode((int)e->nvn_fmt, g_bcs, bp, w, h, stg + o, pitch) != 0;
            } else bad = nvn_deswizzle(&e->d, l, y, e->gmem, stg + o, pitch) != 0;
            if (bad) break;
            if (g_ntcr == g_tcrcap) { g_tcrcap = g_tcrcap ? g_tcrcap * 2 : 64; g_tcr = realloc(g_tcr, (size_t)g_tcrcap * sizeof *g_tcr); if (!g_tcr) die("vkn backend: copy region list allocation failed"); }
            g_tcr[g_ntcr++] = (VkBufferImageCopy){o, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, l, y, 1}, {0, 0, 0}, {w, h, 1}};
            o += ((uint64_t)pitch * rows + 15) & ~15ull;
        }
        if (bad) { NRED_ONCE("texture ser=%llu format %#x flags %#x: its storage layout could not be undone: not uploaded", (unsigned long long)e->serial, e->nvn_fmt, e->d.flags); g_ntcr = q->r0; tj_fallback(q); }
        else q->nr = g_ntcr - q->r0;
    }
}
static void tex_done_all(uint64_t stamp) {   // the uploads landed in the staging buffer: their content is as of this stamp
    for (int j = 0; j < g_ntj; j++) {
        tjob *q = &g_tj[j]; vkn_img *e = q->e;
        if (q->kind == TJ_ZERO) { TSA(TS_ZERO, 1); continue; }
        if (q->kind == TJ_LAYOUT) { TSA(TS_LAY_UP, 1); continue; }
        if (q->kind == TJ_INIT) { TSA(TS_INIT, 1); TSA(TS_INIT_B, q->bytes); continue; }   // not tracked: e->up stays 0 (a CPU write later makes it a first upload)
        if (q->kind < TJ_UP1 || q->kind > TJ_REALW) continue;
        e->up = 1; e->stamp = stamp; e->alw = q->alw; e->span = q->span;
        edges_take(e, stamp);
        TSA(TS_UP1 + q->kind - TJ_UP1, 1); TSA(TS_UP1_B + q->kind - TJ_UP1, q->bytes);
    }
}
static void tex_unprep_all(void) { for (int j = 0; j < g_ntj; j++) if (tj_copies(g_tj[j].kind)) tj_fallback(&g_tj[j]); }   // the staging buffer failed
// The upload command buffer's part (after the buffers' copies): each image from its layout at the submission's start into TRANSFER_DST (uploads, fills) or straight to shader-read, the copies and
// zero fills, then to shader-read
static void tex_record(VkCommandBuffer cb, VkBuffer stg) {
    static VkImageMemoryBarrier2 *b; static int cap;
    if (cap < g_ntj) { cap = g_ntj * 2; b = realloc(b, (size_t)cap * sizeof *b); if (!b) die("vkn backend: texture barrier list allocation failed"); }
    uint32_t nb = 0;
    for (int j = 0; j < g_ntj; j++) {
        tjob *q = &g_tj[j]; vkn_img *e = q->e; int dst = q->kind != TJ_LAYOUT;
        if (q->kind == TJ_NONE) continue;
        b[nb++] = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
            dst ? VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT : SR_STAGES, dst ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            q->old, dst ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : SR_LAYOUT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    }
    if (!nb) return;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, nb, b};
    vk.vkCmdPipelineBarrier2(cb, &di);
    nb = 0;
    for (int j = 0; j < g_ntj; j++) {
        tjob *q = &g_tj[j]; vkn_img *e = q->e;
        if (q->kind == TJ_NONE || q->kind == TJ_LAYOUT) continue;
        VkImageSubresourceRange all = {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        if (q->kind == TJ_ZERO && e->aspect == VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue z = {{0}}; vk.vkCmdClearColorImage(cb, e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &all); }
        else if (q->kind == TJ_ZERO) { VkClearDepthStencilValue z = {0, 0}; vk.vkCmdClearDepthStencilImage(cb, e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &all); }
        else vk.vkCmdCopyBufferToImage(cb, stg, e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)q->nr, g_tcr + q->r0);
        b[nb++] = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            SR_STAGES, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, SR_LAYOUT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, all};
    }
    if (!nb) return;
    di.imageMemoryBarrierCount = nb;
    vk.vkCmdPipelineBarrier2(cb, &di);
}
// The texture line (periodic as the surface lines, and at mtl_drain): per frame since the last line, then totals. Compare nvn_mtl.m's "[nvn] packaged textures" (hle_nvn.c prints it for this
// backend too, from mtl_pkg_stats / mtl_pkg_edge_skips)
static void t_report(const char *why) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static uint64_t p_n, p[TS_N];
    pthread_mutex_lock(&mu);
    uint64_t n = atomic_load(&g_present_n), c[TS_N]; for (int i = 0; i < TS_N; i++) c[i] = atomic_load(&g_ts[i]);
    double f = n > p_n ? (double)(n - p_n) : 1.0;
#define TD(i) ((double)(c[i] - p[i]) / f)
    NLOG("textures up to frame %llu (%s), per frame over the last %llu: uploads %.2f (first %.2f, after a guest write %.2f, never protected %.2f), %.1f KiB (%.1f / %.1f / %.1f), writes beside a texture only %.2f, "
         "zero fills %.2f, layout changes %.2f ahead of the submission and %.2f in it, texture lookups %.1f, sampler lookups %.1f, dirty checks %.1f (%.1f us); totals: uploads %llu first, %llu after a guest write, "
         "%llu never protected, %.1f MiB uploaded in all, initial-content fills %llu (%.1f MiB), zero fills %llu, host images made for sampling %llu, CPU-written textures %llu, failed uploads %llu, "
         "bound while a target %llu, unregistered IDs bound %llu, finalized textures bound %llu, samplers made %llu, feedback copies colour %llu depth %llu reused %llu",
         (unsigned long long)n, why, (unsigned long long)(n - p_n), TD(TS_UP1) + TD(TS_RE) + TD(TS_REALW), TD(TS_UP1), TD(TS_RE), TD(TS_REALW),
         (TD(TS_UP1_B) + TD(TS_RE_B) + TD(TS_REALW_B)) / 1024.0, TD(TS_UP1_B) / 1024.0, TD(TS_RE_B) / 1024.0, TD(TS_REALW_B) / 1024.0, TD(TS_EDGE), TD(TS_ZERO), TD(TS_LAY_UP), TD(TS_LAY_CB),
         TD(TS_TLOOK), TD(TS_SLOOK), TD(TS_CHK), TD(TS_CHK_NS) / 1000.0, (unsigned long long)c[TS_UP1], (unsigned long long)c[TS_RE], (unsigned long long)c[TS_REALW],
         (double)(c[TS_UP1_B] + c[TS_RE_B] + c[TS_REALW_B]) / 1048576.0, (unsigned long long)c[TS_INIT], (double)c[TS_INIT_B] / 1048576.0, (unsigned long long)c[TS_ZERO], (unsigned long long)c[TS_MADE], (unsigned long long)c[TS_CPUW], (unsigned long long)c[TS_FAIL], (unsigned long long)c[TS_FEED], (unsigned long long)c[TS_UNREG], (unsigned long long)c[TS_STALE],
         (unsigned long long)c[TS_SMADE], (unsigned long long)g_stat_fb[0], (unsigned long long)g_stat_fb[1], (unsigned long long)g_stat_fb[2]);
#undef TD
    p_n = n; memcpy(p, c, sizeof p);
    pthread_mutex_unlock(&mu);
}

// ---------- Submission ----------
// Counter timestamps (19.2 MHz) from the guest clock, as the Metal layer's gpu_ticks: games compare them with svc GetSystemTick. Monotonic fallback where guest_ns is 0 (unit tests)
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; } return g * 12 / 625; }
static void post(post_op o) {
    sub_slot *s = g_cur ? g_cur : &g_sub[0];
    if (s->npost == s->postcap) { s->postcap = s->postcap ? s->postcap * 2 : 64; s->post = realloc(s->post, (size_t)s->postcap * sizeof *s->post); if (!s->post) die("vkn backend: completion list allocation failed"); }
    s->post[s->npost++] = o;
}
// A submission's completion effects, in record order: event values, then counter reports (all with this moment's time, as nvn_mtl.m's handler writes them)
static void post_run(sub_slot *s) {
    uint64_t t = gpu_ticks();
    for (int i = 0; i < s->npost; i++) {
        post_op *o = &s->post[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; continue; }
        // A timestamp report's counter field holds the time (0 under render_cond, as on the GPU); samples passed: none were drawn; other counters: 0
        uint64_t rec[2] = {(o->type == NVN_COUNTER_TYPE_TIMESTAMP || o->type == NVN_COUNTER_TYPE_TIMESTAMP_TOP) && !DM_GAME_RENDER_COND ? t : 0, t};
        if (cpu_target(o->addr, sizeof rec, "counter report")) memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);
    }
    s->npost = 0;
}
// The completion thread: each committed submission in order, once the GPU has signalled its timeline value; its effects, then its fence counts as done (mtl_wait's waiters wake)
static void *comp_main(void *arg) {
    (void)arg;
    pthread_setname_np("vkn-complete");   // Darwin's form: dm/host/plat/plat.h maps it on Linux
    pthread_mutex_lock(&g_mu);
    for (;;) {
        while (g_ncomp == g_ncommit) pthread_cond_wait(&g_ccv, &g_mu);
        sub_slot *s = &g_sub[g_ncomp % NSUB]; uint64_t v = s->value; int failed = s->failed;
        pthread_mutex_unlock(&g_mu);
        VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &g_tl, &v};
        // A submitted one waits for the GPU's answer, even once vkn_lost is set (a failed vkQueueSubmit2 is not always a lost device: earlier submissions still run); on a lost device the wait
        // returns in finite time (VK_ERROR_DEVICE_LOST or VK_SUCCESS). Only a driver that breaks that (10 s more of timeouts with vkn_lost set) gets its effects written without the GPU, and then
        // g_leak: nothing the GPU may still use is freed from here on. VK_ERROR_DEVICE_LOST is the device's answer (nothing of it runs any more). Any other error (out of host / device memory)
        // says nothing of the GPU's work: retired without it, and g_leak as well. g_leak is set before g_done moves past it (vkn_sweep's order). Not submitted (failed): at once, in order
        for (int k = 0; !failed; k++) {
            VkResult r = vk.vkWaitSemaphores(vd.dev, &wi, 1000000000ull);
            if (r == VK_SUCCESS) break;
            if (r == VK_ERROR_DEVICE_LOST) { vkn_gpu_lost("the completion thread's vkWaitSemaphores", r); break; }
            if (r != VK_TIMEOUT) {
                vkn_gpu_lost("the completion thread's vkWaitSemaphores", r);
                if (!atomic_exchange(&g_leak, 1)) NLOG("[RED] submission %llu: vkWaitSemaphores %d, not an answer from the GPU: retired without it; from here nothing the GPU may still use is freed (host images and command buffers are leaked)", (unsigned long long)v, r);
                break;
            }
            if (k == 5) NRED_ONCE("submission %llu has not completed on the GPU in 5 s", (unsigned long long)v);
            if (k >= 10 && atomic_load(&vkn_lost)) {
                if (!atomic_exchange(&g_leak, 1)) NLOG("[RED] submission %llu: no answer from the GPU in %d s although it is lost to this run: retired without it; from here nothing the GPU may still use is freed (host images and command buffers are leaked)", (unsigned long long)v, k + 1);
                break;
            }
        }
        post_run(s);
        pthread_mutex_lock(&g_mu); atomic_store(&g_done, v); g_ncomp++; pthread_cond_broadcast(&g_cv);
    }
    return NULL;
}

// ---------- Draws (PLAN.md 2.5 "As built, slice 2c"; Y and winding: 2.4): the draw state, programs, the one push-descriptor set, pipelines, dynamic state, vertex / index buffers, the draw ----------
// The replay keeps NVN's draw state as nvn_mtl.m's S does (state_reset, state_op). A draw resolves its program's vertex and pixel shaders to slice 0's modules (vk_shader.c), their bindings to
// descriptors (draw_desc: NVN uniform slots → the pools' shadow buffers, combined texture slots → the views and samplers slice 2b resolves, here for the slots the shaders sample only), its state
// and the open pass's formats to a monolithic pipeline (draw_pso, cached in memory by vkn's own key), and records it: vkCmdDrawIndexed with the base vertex as the vertex offset, or vkCmdDraw.
// Y and winding (PLAN.md 2.4): NVN decides facing by the winding in normalized device coordinates, after the viewport swizzle. Its memory row 0 is window y 0 in both origin modes; upper left
// maps NDC y +1 to row 0 (a flipped Vulkan viewport), lower left NDC y -1 (Vulkan's own); NEGATIVE_Y toggles the flip. Vulkan's facing, in framebuffer rows, then agrees with NVN's for upper left
// and is reversed for lower left, swizzle or not: the front face depends on the origin only. Scissors and viewports are rows as NVN gives them (never flipped).
// Clip z: [-w, w] as NVN's default (depth clip control's negativeOneToOne), [0, w] after mtl_set_depth_zero. What a draw needs that this slice does not do is said in red once and the draw skipped
static _Atomic int g_depth_zero;   // NVN_DEPTH_MODE_NEAR_IS_ZERO (mtl_set_depth_zero, the game thread)
static struct {
    const prog_t *bst[6];          // the program bound per stage (bit st of BindProgram's stages; nvn_mtl.m bind_stages)
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream;
    NVNbufferRange vbuf[16];
    int tex_base, samp_base;                // the bound texture / sampler pools' bases (SET_TEX_POOL / SET_SAMP_POOL: nvn_dpool_base): handles read from uniform buffers take them
    int64_t copy_stride, copy_img_stride;   // SetCopyRowStride / SetCopyImageStride (0: tightly packed)
    p_bind_buf ubo[6][16], ssbo[6][16]; uint64_t img[6][32];   // img: the image handles per stage and slot (BIND_IMG)
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp, depth_clamp; uint8_t vp_sw[16][4];   // vp_sw: XOR the identity (p_vp_swizzle)
    float blend_color[4]; p_prim_restart restart; int rast_discard;
} S;
static void state_reset(void) {   // nvn_mtl.m's, once (vkn_hello): NVN's initial state
    memset(&S, 0, sizeof S);
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1;
}
static void state_op(uint32_t op, const uint8_t *a) {   // nvn_mtl.m's state_op (the scissor is R's, the texture and sampler binds T's); a count past the table: an overwritten record, skipped
    switch (op) {
        case OP_BIND_PROGRAM: { const p_bind_program *b = (const void *)a; for (int st = 0; st < 6; st++) if (b->stages >> st & 1) S.bst[st] = b->prog; break; }   // compute: bit 5
        case OP_BIND_STATE: { const p_bind_state *b = (const void *)a; blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); S.blend[bs.target] = bs; break;
                case 1: memcpy(&S.color, b->bytes, sizeof S.color); break;
                case 2: memcpy(&S.chmask, b->bytes, sizeof S.chmask); break;
                case 3: memcpy(&S.ms, b->bytes, sizeof S.ms); break;
                case 4: memcpy(&S.poly, b->bytes, sizeof S.poly); break;
                case 5: memcpy(&S.ds, b->bytes, sizeof S.ds); break;
                default: break;
            } break; }
        case OP_BIND_VATTR: { const p_bind_vattr *b = (const void *)a; if (b->n < 0 || b->n > 16) break; memcpy(S.attr, b->st, 4 * (size_t)b->n); S.nattr = b->n; break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = (const void *)a; if (b->n < 0 || b->n > 16) break; memcpy(S.stream, b->st, 8 * (size_t)b->n); S.nstream = b->n; break; }
        case OP_BIND_VBUF: { const p_bind_vbuf *b = (const void *)a; if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break; memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * (size_t)b->n); break; }
        case OP_BIND_UBO: { const p_bind_buf *b = (const void *)a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) S.ubo[b->stage][b->index] = *b; break; }
        case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: { int nvn_dpool_base(const void *pool, int tex); int tex = op == OP_SET_TEX_POOL, b = nvn_dpool_base(((const p_set_pool *)a)->pool, tex); if (tex) S.tex_base = b; else S.samp_base = b; break; }   // hle_nvn.c
        case OP_COPY_STRIDE: { const p_copy_stride *cs = (const p_copy_stride *)a; if (cs->image) S.copy_img_stride = cs->stride; else S.copy_stride = cs->stride; break; }
        case OP_BIND_IMG: { const p_bind_handles *b = (const p_bind_handles *)a; if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->n >= 0 && b->first + b->n <= 32) memcpy(&S.img[b->stage][b->first], b->h, 8 * (size_t)b->n); break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = (const void *)a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) S.ssbo[b->stage][b->index] = *b; break; }
        case OP_STENCIL: { const p_stencil *b = (const void *)a;
            for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = (uint32_t)b->value_mask; else if (b->which == 1) S.st_wmask[f] = (uint32_t)b->mask; else S.st_ref[f] = (uint32_t)b->ref; }
            break; }
        case OP_POLY_OFFSET: memcpy(&S.po, a, sizeof S.po); break;
        case OP_VIEWPORTS: { const p_viewports *b = (const void *)a; if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break; for (int i = 0; i < b->n; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = (const void *)a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_depth_clamp *)(const void *)a)->on; break;
        case OP_VP_SWIZZLE: { const p_vp_swizzle *b = (const void *)a; if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break; memcpy(S.vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)(const void *)a)->rgba, 16); break;
        case OP_PRIM_RESTART: memcpy(&S.restart, a, sizeof S.restart); break;
        case OP_RAST_DISCARD: S.rast_discard = ((const p_rast_discard *)(const void *)a)->on; break;
        default: break;
    }
}
// The records being run (exec_run, one per call depth) and the submission's command handles after the current one (mtl_exec_rest): ds_scan reads ahead from them
static const uint8_t *g_xp[32]; static int g_xd;
static const uint64_t *g_rest; static int g_nrest;
void mtl_exec_rest(const uint64_t *h, int n) { g_rest = h; g_nrest = n; }
static unsigned ds_writes(const ds_st *ds, const uint32_t wm[2]) {   // bits 1 depth, 2 stencil: what a draw with this state may write (NVN stencil ops 1..8, 1 = keep; others as keep, as draw_pso)
    unsigned w = ds->dt && ds->dw ? 1 : 0;
    if (ds->st) { const unsigned o[2][3] = {{ds->fail_f, ds->zfail_f, ds->zpass_f}, {ds->fail_b, ds->zfail_b, ds->zpass_b}};
        for (int f = 0; f < 2; f++) if (wm[f] & 0xff) for (int i = 0; i < 3; i++) if (o[f][i] >= 2 && o[f][i] <= 8) w |= 2; }
    return w;
}
// What the pass beginning at the current record may write: of its depth target (bits as ds_writes), and where (the union of its draws' viewport 0 and scissor, as op_draw clips them,
// 2 pixels wider for wide points and lines). From that record (the draw or clear that opens it) on, with the bound state as it changes, until a SET_RT the pass cannot keep (rt_sub:
// it ends there) or the submission's end (mtl_sub_commit ends the pass). Ends that come sooner (a copy, a barrier, a split) only make the scan longer than the pass: what it finds
// still covers the pass. A clear in the pass (vkCmdClearAttachments over the area), a CALL nested too deep or 8192 records: everything (w 3, the full fw x fh)
static pass_look pass_scan(uint32_t fw, uint32_t fh) {
    pass_look all = {3, 0, 0, fw, fh}, L = {0, 1, 1, 0, 0};
    if (g_xd < 1 || g_xd > 32) return all;
    const uint8_t *stk[32]; int d = g_xd - 1, nr = 0; memcpy(stk, g_xp, sizeof(*stk) * (size_t)g_xd);
    const uint8_t *p = stk[d]; ds_st ds = S.ds; uint32_t wm[2] = {S.st_wmask[0], S.st_wmask[1]};
    int hv = S.has_vp, hs = R.has_sc; float vp[4]; memcpy(vp, S.vp[0], sizeof vp); p_scissor sc = R.sc;
    for (int n = 0; n < 8192; n++) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END:
                if (d > 0) { p = stk[--d]; memcpy(&h, p, sizeof h); p += h.size; continue; }   // after the caller's CALL
                if (nr < g_nrest) { p = (const uint8_t *)(uintptr_t)g_rest[nr++]; continue; }
                return L;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (d >= 31) return all; stk[d++] = p; p = (const uint8_t *)(uintptr_t)t; continue; }
            case OP_BIND_STATE: { const p_bind_state *b = (const void *)a; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); break; }
            case OP_STENCIL: { const p_stencil *b = (const void *)a; if (b->which == 1) for (int f = 0; f < 2; f++) if (b->face & (1 << f)) wm[f] = (uint32_t)b->mask; break; }
            case OP_VIEWPORTS: { const p_viewports *b = (const void *)a; if (b->first == 0 && b->n > 0 && b->n <= 16) { memcpy(vp, b->v[0], sizeof vp); hv = 1; } break; }
            case OP_SCISSOR: memcpy(&sc, a, sizeof sc); hs = 1; break;
            case OP_DRAW: case OP_DRAW_IDX: {
                L.w |= ds_writes(&ds, wm);
                double x0 = 0, y0 = 0, x1 = fw, y1 = fh;
                if (hv) { x0 = vp[0]; y0 = vp[1]; x1 = (double)vp[0] + vp[2]; y1 = (double)vp[1] + vp[3]; if (!(x1 > x0) || !(y1 > y0)) break; }   // op_draw skips it
                if (hs) { if (sc.x > x0) x0 = sc.x; if (sc.y > y0) y0 = sc.y; if ((double)sc.x + sc.w < x1) x1 = (double)sc.x + sc.w; if ((double)sc.y + sc.h < y1) y1 = (double)sc.y + sc.h; }
                if (!(x1 > x0) || !(y1 > y0)) break;
                int64_t ix0 = (int64_t)floor(x0) - 2, iy0 = (int64_t)floor(y0) - 2, ix1 = (int64_t)ceil(x1) + 2, iy1 = (int64_t)ceil(y1) + 2;
                if (L.x0 > L.x1) { L.x0 = ix0; L.y0 = iy0; L.x1 = ix1; L.y1 = iy1; }
                else { if (ix0 < L.x0) L.x0 = ix0; if (iy0 < L.y0) L.y0 = iy0; if (ix1 > L.x1) L.x1 = ix1; if (iy1 > L.y1) L.y1 = iy1; }
                break; }
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: return all;
            case OP_SET_RT: if (rt_sub((const p_set_rt *)a) < 0) return L; break;
            default: break;
        }
        p += h.size;
    }
    return all;
}

// Statistics (draw_report): draws recorded (indexed among them), skipped by why; pipelines made, their creation time, cache hits; descriptor pushes, writes, dummies; uniform bindings, those at
// an address the device cannot bind (copied), those cut short (to the pool's end or the device's range); record time (mtl_exec, outermost)
enum { DK_NOPROG, DK_SHADER, DK_KIND, DK_STATE, DK_EMPTY, DK_RES, DK_PSO, DK_N };
static const char *const g_dk_name[DK_N] = {"no program", "shader refused or missing", "draw kind", "state", "nothing to draw (culled, empty viewport or scissor)", "resources", "pipeline"};
static _Atomic uint64_t g_dr_n, g_dr_idx, g_dr_skip[DK_N], g_pso_made, g_pso_hit, g_pso_ns, g_pso_max, g_pso_fail, g_push, g_dwr, g_ddum, g_ubo_n, g_ubo_unal, g_ubo_short, g_rec_ns, g_lay_made;
#define DS(i, n) atomic_fetch_add_explicit(&(i), (uint64_t)(n), memory_order_relaxed)

// The dummies (made once, draw_init): an image per component type (1x1, 6 layers, cube-compatible, zeros, shader-read) with a view of each kind, a sampler, and a 64 KiB zero buffer for
// uniform slots with nothing bound and vertex inputs nothing feeds. g_draw_ok: they exist, so draws are recorded
#define ZBUF_B 65536u
// dummies per class: 2D / 2D array / cube views of a 1x1 cube-compatible image, a 3D view of a 1x1x1 3D image (g_dimg[3 + class])
static VkImage g_dimg[6]; static VkDeviceMemory g_dmem[6], g_zmem; static VkImageView g_dview[3][4]; static VkSampler g_dsamp; static VkBuffer g_zbuf; static int g_draw_ok;
// Ryujinx's support_buffer (SupportBuffer.cs; std140 as the translator lays it out: alpha_test 0, is_bgra[8] 16, viewport_inverse 144, viewport_size 160, frag_scale_count 176,
// render_scale float[73] at 192 with stride 16): render scale 1.0, the rest 0, as nvn_mtl.m support_buf. Paper Mario: The Thousand-Year Door: 6,100 of its modules bind it
static VkBuffer g_supbuf; static VkDeviceMemory g_supmem;
static VkPipelineCache g_pcache;
// The pipeline cache on disk: <DM_VK_SHADERS>/pipelines.vkpc, read when the cache is made, written (whole, through a temporary file) every 300 frames when pipelines were made since
// and at mtl_drain. Without it every launch compiles every pipeline again: TTYD's title makes ~150 (4.1 s, up to 264 ms each on the RG 476H). The driver checks the header and
// ignores data from another device or driver. DM_VK_PCACHE=0: in memory only
static uint64_t g_pc_saved; static size_t g_pc_size;   // g_pso_made at the last write; the cache's size then (read or written): unchanged, nothing new to write
static const char *pc_path(void) {
    static char p[1024]; static int done;
    if (!done) { done = 1; const char *x = getenv("DM_VK_PCACHE"), *d = getenv("DM_VK_SHADERS"); if (!(x && *x == '0') && d && *d) snprintf(p, sizeof p, "%s/pipelines.vkpc", d); }
    return p[0] ? p : NULL;
}
static void *pc_read(size_t *n) {
    const char *p = pc_path(); *n = 0; if (!p) return NULL;
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = sz > 0 ? malloc((size_t)sz) : NULL;
    if (b && fread(b, 1, (size_t)sz, f) == (size_t)sz) *n = (size_t)sz; else { free(b); b = NULL; }
    fclose(f); return b;
}
static void pc_save(void) {
    const char *p = pc_path(); if (!p || !g_pcache) return;
    uint64_t made = atomic_load(&g_pso_made); if (made == g_pc_saved) return;
    size_t n = 0; if (vk.vkGetPipelineCacheData(vd.dev, g_pcache, &n, NULL) || !n) return;
    if (n == g_pc_size) { g_pc_saved = made; return; }   // pipelines found in the cache: nothing new
    void *b = malloc(n); if (!b) return;
    if (vk.vkGetPipelineCacheData(vd.dev, g_pcache, &n, b) == VK_SUCCESS) {
        char t[1040]; snprintf(t, sizeof t, "%s.tmp", p);
        FILE *f = fopen(t, "wb"); int ok = f && fwrite(b, 1, n, f) == n; if (f) ok = !fclose(f) && ok;
        if (ok && !rename(t, p)) { g_pc_saved = made; g_pc_size = n; NLOG("pipeline cache written: %zu KiB, %llu pipelines made this run", n >> 10, (unsigned long long)made); }
        else NRED_ONCE("the pipeline cache cannot be written to %s: pipelines are compiled again at the next launch", p);
    }
    free(b);
}
static int buf_host(VkDeviceSize n, VkBufferUsageFlags u, VkBuffer *b, VkDeviceMemory *m, void **map) {   // a host-visible coherent buffer, mapped
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL}; VkMemoryRequirements mr; int mt;
    *b = VK_NULL_HANDLE; *m = VK_NULL_HANDLE;
    if (vk.vkCreateBuffer(vd.dev, &bi, NULL, b)) { *b = VK_NULL_HANDLE; return 0; }
    vk.vkGetBufferMemoryRequirements(vd.dev, *b, &mr);
    mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, m)) { *m = VK_NULL_HANDLE; vk.vkDestroyBuffer(vd.dev, *b, NULL); *b = VK_NULL_HANDLE; return 0; }
    if (vk.vkBindBufferMemory(vd.dev, *b, *m, 0) || vk.vkMapMemory(vd.dev, *m, 0, VK_WHOLE_SIZE, 0, map)) { vk.vkDestroyBuffer(vd.dev, *b, NULL); vk.vkFreeMemory(vd.dev, *m, NULL); *b = VK_NULL_HANDLE; *m = VK_NULL_HANDLE; return 0; }
    return 1;
}
static void draw_init(void) {   // vkn_hello, before any submission: the dummies' clear and layout go through a one-off command buffer, waited for
    if (!vkd_init()) return;   // push descriptors, or descriptor pools where the device has none (vk_desc.c); said there
    void *zm; if (!buf_host(ZBUF_B, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &g_zbuf, &g_zmem, &zm)) { NLOG("[RED] the draws' zero buffer cannot be made: nothing is drawn"); return; }
    memset(zm, 0, ZBUF_B);
    void *sm; if (buf_host(ZBUF_B, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &g_supbuf, &g_supmem, &sm)) { memset(sm, 0, ZBUF_B); for (int i = 0; i < 73; i++) ((float *)((uint8_t *)sm + 192))[4 * i] = 1.0f; }
    else NLOG("[RED] the support buffer cannot be made: shaders that bind it read zeros (render scale 0)");
    static const VkFormat df[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT};
    for (int c = 0; c < 3; c++) {
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, VK_IMAGE_TYPE_2D, df[c], {1, 1, 1}, 1, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkMemoryRequirements mr; int mt;
        if (vk.vkCreateImage(vd.dev, &ii, NULL, &g_dimg[c])) { NLOG("[RED] the draws' dummy images cannot be made: nothing is drawn"); return; }
        vk.vkGetImageMemoryRequirements(vd.dev, g_dimg[c], &mr);
        if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, 0);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
        if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &g_dmem[c]) || vk.vkBindImageMemory(vd.dev, g_dimg[c], g_dmem[c], 0)) { NLOG("[RED] the draws' dummy images have no memory: nothing is drawn"); return; }
        VkImageCreateInfo i3 = ii; i3.flags = 0; i3.imageType = VK_IMAGE_TYPE_3D; i3.arrayLayers = 1;
        if (vk.vkCreateImage(vd.dev, &i3, NULL, &g_dimg[3 + c])) { NLOG("[RED] the draws' dummy images cannot be made: nothing is drawn"); return; }
        vk.vkGetImageMemoryRequirements(vd.dev, g_dimg[3 + c], &mr);
        if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, 0);
        ai = (VkMemoryAllocateInfo){VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
        if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &g_dmem[3 + c]) || vk.vkBindImageMemory(vd.dev, g_dimg[3 + c], g_dmem[3 + c], 0)) { NLOG("[RED] the draws' dummy images have no memory: nothing is drawn"); return; }
        static const VkImageViewType vt[4] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_3D};
        for (int t = 0; t < 4; t++) {
            VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_dimg[t == 3 ? 3 + c : c], vt[t], df[c], {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, t == 0 || t == 3 ? 1 : 6}};
            if (vk.vkCreateImageView(vd.dev, &vi, NULL, &g_dview[c][t])) { NLOG("[RED] the draws' dummy views cannot be made: nothing is drawn"); return; }
        }
    }
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.maxLod = 0.25f;
    if (vk.vkCreateSampler(vd.dev, &si, NULL, &g_dsamp)) { NLOG("[RED] the draws' dummy sampler cannot be made: nothing is drawn"); return; }
    VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    size_t pcn; void *pcd = pc_read(&pcn); pci.initialDataSize = pcn; pci.pInitialData = pcd;
    if (vk.vkCreatePipelineCache(vd.dev, &pci, NULL, &g_pcache)) {   // data the driver refuses: an empty cache
        pci.initialDataSize = 0; pci.pInitialData = NULL; if (vk.vkCreatePipelineCache(vd.dev, &pci, NULL, &g_pcache)) g_pcache = VK_NULL_HANDLE;   // none: pipelines are made without one
    } else if (pcn) { size_t n = 0; if (!vk.vkGetPipelineCacheData(vd.dev, g_pcache, &n, NULL)) g_pc_size = n; NLOG("pipeline cache read: %zu KiB from %s", pcn >> 10, pc_path()); }
    free(pcd);
    VkCommandPool pool; VkCommandBuffer cb;
    VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, vd.qfam};
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (vk.vkCreateCommandPool(vd.dev, &pi, NULL, &pool)) { NLOG("[RED] the draws' dummies cannot be cleared: nothing is drawn"); return; }
    cai.commandPool = pool;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    VkResult r = vk.vkAllocateCommandBuffers(vd.dev, &cai, &cb);
    if (!r) r = vk.vkBeginCommandBuffer(cb, &bi);
    if (!r) {
        VkImageMemoryBarrier2 b[6]; VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6}, one = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 6, b};
        for (int c = 0; c < 6; c++) b[c] = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_dimg[c], c < 3 ? all : one};
        vk.vkCmdPipelineBarrier2(cb, &di);
        VkClearColorValue z = {{0}};
        for (int c = 0; c < 6; c++) vk.vkCmdClearColorImage(cb, g_dimg[c], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, c < 3 ? &all : &one);
        for (int c = 0; c < 6; c++) b[c] = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, SR_STAGES,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, SR_LAYOUT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_dimg[c], c < 3 ? all : one};
        vk.vkCmdPipelineBarrier2(cb, &di);
        r = vk.vkEndCommandBuffer(cb);
    }
    if (!r) {
        VkCommandBufferSubmitInfo c = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, cb, 0};
        VkSubmitInfo2 s2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 1, &c, 0, NULL};
        pthread_mutex_lock(&vkn_qmu); r = vk.vkQueueSubmit2(vd.q, 1, &s2, VK_NULL_HANDLE); if (!r) r = vk.vkQueueWaitIdle(vd.q); pthread_mutex_unlock(&vkn_qmu);
    }
    vk.vkDestroyCommandPool(vd.dev, pool, NULL);
    if (r) { NLOG("[RED] the draws' dummies cannot be cleared (VkResult %d): nothing is drawn", r); return; }
    g_draw_ok = 1;
}

// ---- Programs: a program object's shaders by NVN stage, by its serial (new at each SetShaders), as nvn_mtl.m's prog_get ----
typedef struct { vkn_shader *sh[6]; uint8_t has, bad; } prog_ent;   // has: the stages its shaders map to; bad: those refused (bit 7: a key the map lacks)
static nvn_hmap g_progs = {.klen = 8};
static const prog_ent *prog_ent_get(const prog_t *p) {
    static const prog_t *lp; static uint64_t ls; static prog_ent *le, none;
    if (!p || !p->serial) return &none;
    if (p == lp && p->serial == ls) return le;
    uint64_t ser = p->serial; prog_ent *e = nvn_hm_get(&g_progs, &ser);
    if (!e) {
        if (!(e = calloc(1, sizeof *e))) die("vkn backend: program entry allocation failed");
        for (unsigned i = 0; i < p->nstage && i < 6; i++) {
            uint8_t k[8]; if (!prog_key(p, i, k)) continue;
            int st = vkn_shmap_stage(k);
            if (st < 0) { static int n; e->bad |= 0x80; if (n++ < 16) NLOG("[RED] shader %02x%02x%02x%02x%02x%02x%02x%02x is not in the shader map: the draws of its program are skipped", k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]); continue; }
            e->has |= (uint8_t)(1u << st); if (!(e->sh[st] = vkn_shader_get(k, st))) e->bad |= (uint8_t)(1u << st);
        }
        nvn_hm_put(&g_progs, &ser, e);
    }
    lp = p; ls = p->serial; le = e;
    return e;
}

// ---- Layouts: one push-descriptor set per (vertex, pixel) pair, vertex bindings 0-31 and pixel bindings 32-63 (vk_shader.c), kept for the run. A pair whose interface does not match is refused
// with it (the same red line, its draws skipped): every pixel input variable written by the vertex shader as the same variable, at each of its locations the same first location, component
// type, components and array lengths (Vulkan's interface matching, as dmsh_build_vk.py pair_inputs: a vec4[2] is not two vec4s; more components out than in needs maintenance4, which is not
// enabled, and slice 0's chain pads without widening). Component decorations (two variables sharing a location) cannot reach here: the builder marks them unsupported (structure()), a module
// whose status is not ok is refused, and spv_sha256 ties a reflection to its SPIR-V, so each location is one whole variable's ----
typedef struct { VkDescriptorSetLayout dsl; VkPipelineLayout pl; int n; } lay_ent;
static void io_desc(const vkn_shader *s, uint32_t l, char *o, size_t n) {   // location l's variable: "sint x4", "float x4[2] from 0"
    static const char *const tn[3] = {"float", "uint", "sint"};
    int k = snprintf(o, n, "%s x%u", tn[s->io_ct[l] % 3], s->io_n[l]);
    for (int d = 0; d < s->io_nd[l] && k > 0 && (size_t)k < n; d++) k += snprintf(o + k, n - (size_t)k, "[%u]", s->io_dims[l][d]);
    if (s->io_nd[l] && k > 0 && (size_t)k < n) snprintf(o + k, n - (size_t)k, " from %u", s->io_first[l]);
}
static nvn_hmap g_lays = {.klen = 8};
static lay_ent g_lay_bad;
// A binding's descriptor type and count (the layouts of draws and dispatches)
static VkDescriptorType bind_type(const vkn_sbind *b) {
    switch (b->kind) {
        case VKN_B_TEX: case VKN_B_TEXU: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        case VKN_B_SBUF: case VKN_B_UBOP: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case VKN_B_IMG: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case VKN_B_TEXD: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case VKN_B_SAMPD: return VK_DESCRIPTOR_TYPE_SAMPLER;
        default: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
}
static uint32_t bind_count(const vkn_sbind *b) { return b->kind == VKN_B_TEXD || b->kind == VKN_B_SAMPD || b->kind == VKN_B_TEXU ? b->n : 1; }
static const lay_ent *lay_get(const vkn_shader *vs, const vkn_shader *ps) {
    static const vkn_shader *lv, *lps; static const lay_ent *ll;
    if (vs == lv && ps == lps && ll) return ll;
    uint64_t k = (uint64_t)vs->id << 32 | (ps ? ps->id : 0); lay_ent *e = nvn_hm_get(&g_lays, &k);
    if (!e) {
        VkDescriptorSetLayoutBinding b[2 * VKN_MAXB]; uint32_t n = 0;
        for (int s = 0; s < 2; s++) { const vkn_shader *sh = s ? ps : vs; if (!sh) continue;
            for (int i = 0; i < sh->nb; i++) b[n++] = (VkDescriptorSetLayoutBinding){sh->b[i].binding, bind_type(&sh->b[i]), bind_count(&sh->b[i]),
                                                                                    s ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT, NULL}; }
        e = &g_lay_bad; char why[128] = "";
        for (uint32_t l = 0; ps && l < 32 && !*why; l++) if (ps->io_mask >> l & 1) {
            if (!(vs->io_mask >> l & 1)) snprintf(why, sizeof why, "the pixel shader reads location %u, the vertex shader does not write it", l);
            else if (vs->io_ct[l] != ps->io_ct[l] || vs->io_n[l] != ps->io_n[l] || vs->io_first[l] != ps->io_first[l] || vs->io_nd[l] != ps->io_nd[l] || memcmp(vs->io_dims[l], ps->io_dims[l], 4)) {
                char wv[40], rv[40]; io_desc(vs, l, wv, sizeof wv); io_desc(ps, l, rv, sizeof rv);
                snprintf(why, sizeof why, "location %u: written as %s, read as %s", l, wv, rv); }
        }
        if (*why) NLOG("[RED] shaders %s and %s do not match (%s): their draws are skipped", vs->key, ps->key, why);
        else if (n > vkd_max_bindings()) NLOG("[RED] shaders %s and %s have %u bindings, past the device's %u push descriptors: their draws are skipped (more sets are a later slice)", vs->key, ps ? ps->key : "-", n, vkd_max_bindings());
        else {
            lay_ent *m = calloc(1, sizeof *m); if (!m) die("vkn backend: layout entry allocation failed");
            VkDescriptorSetLayoutCreateInfo di = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, vkd_layout_flags(), n, b};
            VkResult r = vk.vkCreateDescriptorSetLayout(vd.dev, &di, NULL, &m->dsl);
            VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};   // the pixel shader's slot handles (draw_idmap)
            VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &m->dsl, ps && ps->idmap ? 1 : 0, &pr};
            if (!r && (r = vk.vkCreatePipelineLayout(vd.dev, &pi, NULL, &m->pl))) vk.vkDestroyDescriptorSetLayout(vd.dev, m->dsl, NULL);
            if (r) { NLOG("[RED] the layout of shaders %s and %s cannot be made (VkResult %d): their draws are skipped", vs->key, ps ? ps->key : "-", r); free(m); }
            else { m->n = (int)n; e = m; DS(g_lay_made, 1); }
        }
        nvn_hm_put(&g_lays, &k, e);
    }
    lv = vs; lps = ps; ll = e;
    return e;
}

// ---- Formats ----
static int fmt_comp(VkFormat f) {   // a format's numeric class as a shader reads it: 0 float (and normalized), 1 unsigned integer, 2 signed integer
    switch (f) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_B8G8R8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_A8B8G8R8_UINT_PACK32:
        case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_S8_UINT: return 1;
        case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_B8G8R8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_SINT: case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A2R10G10B10_SINT_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT: return 2;
        default: return 0;
    }
}
static VkFormatFeatureFlags fmt_opt(VkFormat f, int buffer) {   // optimal-tiling (attachment blend) or buffer (vertex) features, cached
    static struct { VkFormat f; VkFormatFeatureFlags o, b; } c[32]; static int n;
    for (int i = 0; i < n; i++) if (c[i].f == f) return buffer ? c[i].b : c[i].o;
    VkFormatProperties p; vk.vkGetPhysicalDeviceFormatProperties(vd.pd, f, &p);
    if (n < 32) { c[n].f = f; c[n].o = p.optimalTilingFeatures; c[n++].b = p.bufferFeatures; }
    return buffer ? p.bufferFeatures : p.optimalTilingFeatures;
}
// NVN vertex formats as (components, bits, kind); kinds in VkFormat's order within a size (UNORM SNORM USCALED SSCALED UINT SINT, then SFLOAT), so a VkFormat is computed from them
enum { VK_UN, VK_SN, VK_US, VK_SS, VK_UI, VK_SI, VK_F };
static VkFormat vkf(int n, int bits, int kind) {
    static const int b8[4] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}, b16[4] = {VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16A16_UNORM},
                     b32[4] = {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT};
    if (n < 1 || n > 4) return VK_FORMAT_UNDEFINED;
    if (bits == 8) return kind < VK_F ? (VkFormat)(b8[n - 1] + kind) : VK_FORMAT_UNDEFINED;
    if (bits == 16) return (VkFormat)(b16[n - 1] + kind);
    if (bits == 32) return kind == VK_UI ? (VkFormat)b32[n - 1] : kind == VK_SI ? (VkFormat)(b32[n - 1] + 1) : kind == VK_F ? (VkFormat)(b32[n - 1] + 2) : VK_FORMAT_UNDEFINED;
    return VK_FORMAT_UNDEFINED;
}
// The VkFormat of NVN vertex format f for an input of component type ct (as nvn_mtl.m's vfmt + vfmt_for_ct): integers keep their bits (a sign change, 32-bit floats as integers and the reverse);
// an integer input fed a normalized, half or packed format reads its raw bytes as the integer format of the same layout (red: Maxwell gives the converted float's bits, nvn_mtl.m converts
// them on the CPU, a later slice here); a float input fed 8- or 16-bit integers reads them scaled (red: Maxwell gives the raw bits). UNDEFINED = none
static VkFormat vtx_fmt(unsigned f, int ct) {
    int n = 0, bits = 0, kind = -1;
    static const struct { uint8_t bits, kind; } t12[12] = {{8, VK_UN}, {8, VK_SN}, {8, VK_UI}, {8, VK_SI}, {16, VK_F}, {16, VK_UN}, {16, VK_SN}, {16, VK_UI}, {16, VK_SI}, {32, VK_F}, {32, VK_UI}, {32, VK_SI}};
    if (f >= 0x01 && f <= 0x30) { n = (int)(f - 1) / 12 + 1; bits = t12[(f - 1) % 12].bits; kind = t12[(f - 1) % 12].kind; }
    else if (f >= 0x51 && f <= 0x68) { int i = (int)(f - 0x51) >> 1; n = i / 3 + 1; bits = 8 << (i % 3); kind = (f - 0x51) & 1 ? VK_SS : VK_US; if (bits == 32) return VK_FORMAT_UNDEFINED; }   // integers read as floats: scaled
    else {
        VkFormat pf = f == 0x3D ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : f == 0x69 ? VK_FORMAT_A2B10G10R10_SNORM_PACK32 : f == 0x3F ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : f == 0x9A ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_UNDEFINED;
        if (!pf || !ct) return pf;
        NRED_ONCE("vertex format %#x feeds an integer input: its raw bytes are read as integers (Maxwell gives the converted float's bits; a CPU conversion is a later slice)", f);
        return f == 0x3D || f == 0x69 ? (ct == 1 ? VK_FORMAT_A2B10G10R10_UINT_PACK32 : VK_FORMAT_A2B10G10R10_SINT_PACK32) : f == 0x9A ? vkf(4, 8, ct == 1 ? VK_UI : VK_SI) : vkf(1, 32, ct == 1 ? VK_UI : VK_SI);
    }
    if (ct == 1 || ct == 2) {
        int want = ct == 1 ? VK_UI : VK_SI;
        if (kind != VK_UI && kind != VK_SI && !(kind == VK_F && bits == 32))
            NRED_ONCE("vertex format %#x feeds an integer input: its raw bytes are read as integers (Maxwell gives the converted float's bits; a CPU conversion is a later slice)", f);
        return vkf(n, bits, want);
    }
    if (kind == VK_UI || kind == VK_SI) {
        if (bits == 32) return vkf(n, 32, VK_F);   // the same bits as floats (nvn_mtl.m)
        NRED_ONCE("vertex format %#x (integers) feeds a float input: read scaled to float (Maxwell gives the raw bits)", f);
        return vkf(n, bits, kind == VK_UI ? VK_US : VK_SS);
    }
    return vkf(n, bits, kind);
}
// The bytes a vertex fetch of NVN vertex format f reads (vtx_fmt's decoding; the integer reinterpretations it makes keep the size). 0 = a format vtx_fmt has none for (the draw is skipped there)
static uint32_t vtx_bytes(unsigned f) {
    static const uint8_t b12[12] = {8, 8, 8, 8, 16, 16, 16, 16, 16, 32, 32, 32};
    if (f >= 0x01 && f <= 0x30) return ((f - 1) / 12 + 1) * b12[(f - 1) % 12] / 8;
    if (f >= 0x51 && f <= 0x68) { unsigned i = (f - 0x51) >> 1; return (i / 3 + 1) * (8u << (i % 3)) / 8; }
    return f == 0x3D || f == 0x69 || f == 0x3F || f == 0x9A ? 4 : 0;
}
static VkBlendFactor bfac(unsigned f) {   // nvn_mtl.m's bfac; the dual-source ones are refused before (draw_key)
    switch (f) { case 1: return VK_BLEND_FACTOR_ZERO; case 2: return VK_BLEND_FACTOR_ONE; case 3: return VK_BLEND_FACTOR_SRC_COLOR; case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA; case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; case 7: return VK_BLEND_FACTOR_DST_ALPHA; case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR; case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR; case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR; case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA; case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR; case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA; }
    return VK_BLEND_FACTOR_ONE;
}
static VkBlendOp bop(unsigned e) { switch (e) { case 2: return VK_BLEND_OP_SUBTRACT; case 3: return VK_BLEND_OP_REVERSE_SUBTRACT; case 4: return VK_BLEND_OP_MIN; case 5: return VK_BLEND_OP_MAX; } return VK_BLEND_OP_ADD; }
static blend_st adv_blend(blend_st b) {   // nvn_mtl.m's: the Porter-Duff advanced modes as fixed-function factors; others keep the state's functions, said
    if (!b.adv) return b;
    enum { Z = NVN_BLEND_FUNC_ZERO, O = NVN_BLEND_FUNC_ONE, SA = NVN_BLEND_FUNC_SRC_ALPHA, ISA = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, DA = NVN_BLEND_FUNC_DST_ALPHA, IDA = NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA };
    static const uint8_t pd[][4] = {
        [NVN_BLEND_ADVANCED_MODE_ZERO] = {Z, Z, Z, Z},        [NVN_BLEND_ADVANCED_MODE_SRC] = {O, Z, O, Z},           [NVN_BLEND_ADVANCED_MODE_DST] = {Z, O, Z, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_OVER] = {O, ISA, O, ISA}, [NVN_BLEND_ADVANCED_MODE_DST_OVER] = {IDA, O, IDA, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_IN] = {DA, Z, DA, Z},     [NVN_BLEND_ADVANCED_MODE_DST_IN] = {Z, SA, Z, SA},
        [NVN_BLEND_ADVANCED_MODE_SRC_OUT] = {IDA, Z, IDA, Z},  [NVN_BLEND_ADVANCED_MODE_DST_OUT] = {Z, ISA, Z, ISA},
        [NVN_BLEND_ADVANCED_MODE_SRC_ATOP] = {DA, ISA, Z, O},  [NVN_BLEND_ADVANCED_MODE_DST_ATOP] = {IDA, SA, O, Z},
        [NVN_BLEND_ADVANCED_MODE_XOR] = {IDA, ISA, IDA, ISA},  [NVN_BLEND_ADVANCED_MODE_PLUS] = {O, O, O, O},
        [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED] = {O, O, O, O}, [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED_ALPHA] = {O, O, O, O},
    };
    const uint8_t *f = b.adv < sizeof pd / sizeof *pd && pd[b.adv][0] ? pd[b.adv] : NULL;
    if (!f || b.adv_ovl || (b.adv_npm && f[0] != O && f[0] != Z)) {
        static uint64_t seen; if (!(seen >> (b.adv & 63) & 1)) { seen |= 1ull << (b.adv & 63);
            NLOG("[RED] advanced blend mode %#x (overlap %d, %spremultiplied source) has no fixed-function equivalent: the state's blend functions are used", (unsigned)b.adv, (int)b.adv_ovl, b.adv_npm ? "not " : ""); }
        return b;
    }
    b.src_rgb = b.adv_npm && f[0] == O ? SA : f[0]; b.dst_rgb = f[1]; b.src_a = f[2]; b.dst_a = f[3];
    b.eq_rgb = b.eq_a = NVN_BLEND_EQUATION_ADD;
    return b;
}

// ---- Pipelines: vkn's key (every field the pipeline is made from; memset first, so padding is zero), cached for the run (never destroyed: an open item) ----
typedef struct {
    uint32_t vs, ps, nc, cf[8], df, sf;
    uint8_t topo, restart, cull, front, pmode, bias, dclamp, clip11, a2c, nin, nbd, pad0;
    uint8_t ben[8], wm[8], bsc[8], bdc[8], bec[8], bsa[8], bda[8], bea[8];
    uint8_t dt, dw, dcmp, st, scmp[2], sfail[2], szfail[2], szpass[2], pad1[2];
    struct { uint32_t fmt, off; uint8_t loc, bind, pad[2]; } in[16];
    struct { uint32_t stride; uint8_t idx, rate, pad[2]; } bd[16];
} pso_key;
static nvn_hmap g_psos = {.klen = sizeof(pso_key)};
static const char g_pso_bad;
#define PSO_BAD ((void *)&g_pso_bad)
// DM_VK_DIAG=solidps: every draw's pixel stage writes magenta to all eight outputs (diagnostics: does a pass's geometry rasterize at all)
static const uint32_t g_solid_ps[] = {0x7230203, 0x10600, 0x70000, 0x13, 0x0, 0x20011, 0x1, 0x3000e, 0x0, 0x1, 0xd000f, 0x4, 0x1, 0x6e69616d, 0x0, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0x30010, 0x1, 0x7, 0x40047, 0x2, 0x1e, 0x0, 0x40047, 0x3, 0x1e, 0x1, 0x40047, 0x4, 0x1e, 0x2, 0x40047, 0x5, 0x1e, 0x3, 0x40047, 0x6, 0x1e, 0x4, 0x40047, 0x7, 0x1e, 0x5, 0x40047, 0x8, 0x1e, 0x6, 0x40047, 0x9, 0x1e, 0x7, 0x20013, 0xa, 0x30021, 0xb, 0xa, 0x30016, 0xc, 0x20, 0x40017, 0xd, 0xc, 0x4, 0x40020, 0xe, 0x3, 0xd, 0x4002b, 0xc, 0xf, 0x0, 0x4002b, 0xc, 0x10, 0x3f800000, 0x7002c, 0xd, 0x11, 0x10, 0xf, 0x10, 0x10, 0x4003b, 0xe, 0x2, 0x3, 0x4003b, 0xe, 0x3, 0x3, 0x4003b, 0xe, 0x4, 0x3, 0x4003b, 0xe, 0x5, 0x3, 0x4003b, 0xe, 0x6, 0x3, 0x4003b, 0xe, 0x7, 0x3, 0x4003b, 0xe, 0x8, 0x3, 0x4003b, 0xe, 0x9, 0x3, 0x50036, 0xa, 0x1, 0x0, 0xb, 0x200f8, 0x12, 0x3003e, 0x2, 0x11, 0x3003e, 0x3, 0x11, 0x3003e, 0x4, 0x11, 0x3003e, 0x5, 0x11, 0x3003e, 0x6, 0x11, 0x3003e, 0x7, 0x11, 0x3003e, 0x8, 0x11, 0x3003e, 0x9, 0x11, 0x100fd, 0x10038};
static VkShaderModule solid_ps(void) {
    static VkShaderModule m; static int tried;
    if (!tried) { tried = 1; VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof g_solid_ps, g_solid_ps}; if (vk.vkCreateShaderModule(vd.dev, &ci, NULL, &m)) m = VK_NULL_HANDLE; }
    return m;
}
static VkPipeline pso_make(const pso_key *k, const vkn_shader *vs, const vkn_shader *ps, VkPipelineLayout pl) {
    VkPipelineShaderStageCreateInfo stg[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vs->mod, "main", NULL},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ps ? ps->mod : VK_NULL_HANDLE, "main", NULL}};
    { const char *dg = getenv("DM_VK_DIAG"), *sp = getenv("DM_VK_SOLID_PS");   // DM_VK_SOLID_PS=<pixel key>[,…]: only those shaders' draws (which draw writes a pixel)
      if (((dg && strstr(dg, "solidps")) || (sp && ps && strstr(sp, ps->key))) && ps && solid_ps()) stg[1].module = solid_ps(); }
    VkVertexInputAttributeDescription ad[16]; VkVertexInputBindingDescription bd[16];
    for (int i = 0; i < k->nin; i++) ad[i] = (VkVertexInputAttributeDescription){k->in[i].loc, k->in[i].bind, (VkFormat)k->in[i].fmt, k->in[i].off};
    for (int i = 0; i < k->nbd; i++) bd[i] = (VkVertexInputBindingDescription){k->bd[i].idx, k->bd[i].stride, k->bd[i].rate ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL, 0, k->nbd, bd, k->nin, ad};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo, k->restart};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, VK_TRUE};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->clip11 ? &dcc : NULL, 0, 1, NULL, 1, NULL};
    static int rdis = -1; if (rdis < 0) { const char *x = getenv("DM_VK_DIAG"); rdis = x && strstr(x, "rdiscard"); }   // DM_VK_DIAG=rdiscard (wrong on purpose, a bound): no draw rasterizes (vertex shading and per-pass cost only)
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->dclamp, rdis ? VK_TRUE : VK_FALSE, (VkPolygonMode)k->pmode, k->cull, (VkFrontFace)k->front, k->bias, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0, NULL, k->a2c, VK_FALSE};
    VkStencilOpState sf = {(VkStencilOp)k->sfail[0], (VkStencilOp)k->szpass[0], (VkStencilOp)k->szfail[0], (VkCompareOp)k->scmp[0], 0, 0, 0},
                     sb = {(VkStencilOp)k->sfail[1], (VkStencilOp)k->szpass[1], (VkStencilOp)k->szfail[1], (VkCompareOp)k->scmp[1], 0, 0, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, k->dt, k->dw, (VkCompareOp)k->dcmp, VK_FALSE, k->st, sf, sb, 0, 1};
    VkPipelineColorBlendAttachmentState ba[8];
    for (uint32_t i = 0; i < k->nc; i++) ba[i] = (VkPipelineColorBlendAttachmentState){k->ben[i], (VkBlendFactor)k->bsc[i], (VkBlendFactor)k->bdc[i], (VkBlendOp)k->bec[i], (VkBlendFactor)k->bsa[i], (VkBlendFactor)k->bda[i], (VkBlendOp)k->bea[i], k->wm[i]};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_LOGIC_OP_COPY, k->nc, ba, {0, 0, 0, 0}};
    static const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                         VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof dyn / sizeof *dyn, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k->nc; i++) cf[i] = (VkFormat)k->cf[i];
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->nc, cf, (VkFormat)k->df, (VkFormat)k->sf};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, ps ? 2 : 1, stg, &vi, &ia, NULL, &vps, &rs, &ms, &ds, &cb, &dy, pl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VkPipeline p; uint64_t t0 = now_ns();
    VkResult r = vk.vkCreateGraphicsPipelines(vd.dev, g_pcache, 1, &gi, NULL, &p);
    uint64_t dt = now_ns() - t0;
    if (r) { DS(g_pso_fail, 1); NLOG("[RED] vkCreateGraphicsPipelines failed (VkResult %d) for shaders %s and %s: their draws with this state are skipped", r, vs->key, ps ? ps->key : "-"); return VK_NULL_HANDLE; }
    uint64_t n = DS(g_pso_made, 1); DS(g_pso_ns, dt);
    for (uint64_t m = atomic_load(&g_pso_max); dt > m && !atomic_compare_exchange_weak(&g_pso_max, &m, dt); ) {}
    if (getenv("DM_VK_PASSLOG")) NLOG("passlog pipeline %llu: %s / %s cull %d front %d depth test %d write %d cmp %d stencil %d blend0 %d (%d,%d,%d) write mask0 %#x a2c %d topo %d",
        (unsigned long long)n + 1, vs->key, ps ? ps->key : "-", (int)k->cull, (int)k->front, (int)k->dt, (int)k->dw, (int)k->dcmp, (int)k->st, (int)k->ben[0], (int)k->bsc[0], (int)k->bdc[0], (int)k->bec[0], (unsigned)k->wm[0], (int)k->a2c, (int)k->topo);
    if (getenv("DM_VK_PASSLOG")) { char vb[512]; int o = 0;
        for (int i = 0; i < k->nin && o < (int)sizeof vb - 40; i++) o += snprintf(vb + o, sizeof vb - o, " loc%u:fmt%u+%u@b%u", k->in[i].loc, k->in[i].fmt, k->in[i].off, k->in[i].bind);
        for (int i = 0; i < k->nbd && o < (int)sizeof vb - 40; i++) o += snprintf(vb + o, sizeof vb - o, " b%u:stride%u%s", k->bd[i].idx, k->bd[i].stride, k->bd[i].rate ? "/inst" : "");
        NLOG("passlog pipeline %llu inputs:%s", (unsigned long long)n + 1, o ? vb : " none"); }
    if (n < 8 || dt > 20000000ull) NLOG("pipeline %llu: shaders %s and %s, %u colour targets, depth %d stencil %d, %.2f ms to make (frame %llu)", (unsigned long long)n + 1, vs->key, ps ? ps->key : "-", k->nc, k->df != 0, k->sf != 0,
                                        (double)dt / 1e6, (unsigned long long)atomic_load(&g_present_n));
    return p;
}
// The key of the draw about to be recorded (the pass open: its formats). 0 = a state this slice cannot draw (said once), the draw skipped
static int draw_key(pso_key *k, const vkn_shader *vs, const vkn_shader *ps, int prim, int indexed, int itype, uint32_t *zbind) {
    memset(k, 0, sizeof *k);
    k->vs = vs->id; k->ps = ps ? ps->id : 0;
    static const int8_t topo[16] = {-1, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, -1, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int tp = prim >= 0 && prim < 16 ? topo[prim] : -1;
    if (tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN && vd.port) tp = -1;   // the portability subset's triangleFans is not asked for
    if (tp < 0) { NRED_ONCE("draws of primitive %d (points, loops, quads, polygons, adjacency and patches are a later slice): skipped", prim); return 0; }
    k->topo = (uint8_t)tp;
    if (S.restart.on && indexed && (tp == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)) {
        uint32_t all = itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 0xffffu : 0xffffffffu;
        if (S.restart.index != all) NRED_ONCE("primitive restart index %#x (Vulkan restarts at all ones only): restarted at %#x", S.restart.index, all);
        k->restart = 1;
    }   // lists: no restart (as nvn_mtl.m)
    k->cull = (uint8_t)S.poly.cull;
    k->front = (uint8_t)(vkn_origin_ll() ? S.poly.front == NVN_FRONT_FACE_CCW : S.poly.front != NVN_FRONT_FACE_CCW);   // VK_FRONT_FACE_CLOCKWISE = 1 (see the section's head)
    int pm = S.poly.mode == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : S.poly.mode == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    unsigned ob = pm == VK_POLYGON_MODE_POINT ? NVN_POLYGON_OFFSET_ENABLE_POINT : pm == VK_POLYGON_MODE_LINE ? NVN_POLYGON_OFFSET_ENABLE_LINE : NVN_POLYGON_OFFSET_ENABLE_FILL;   // NVN's mode picks the enable, also where it is filled below
    if (pm != VK_POLYGON_MODE_FILL && !vd.fill_ns) { NRED_ONCE("polygon mode %u, and the device has no fillModeNonSolid: filled", (unsigned)S.poly.mode); pm = VK_POLYGON_MODE_FILL; }
    k->pmode = (uint8_t)pm;
    k->bias = (S.poly.offset_en & ob) && tp >= VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST && R.hd;
    if (S.depth_clamp && !vd.depth_clamp) NRED_ONCE("depth clamp on, and the device has no depthClamp: depth is clipped");
    k->dclamp = S.depth_clamp && vd.depth_clamp;
    if (!atomic_load_explicit(&g_depth_zero, memory_order_relaxed)) { if (vd.clip_ctl) k->clip11 = 1; else NRED_ONCE("NVN clips z at -w, and the device has no depth clip control: clipped at 0"); }
    if (S.color.logic_op != NVN_LOGIC_OP_COPY) NRED_ONCE("logic op %u (Vulkan's needs logicOp, a later slice): not applied", (unsigned)S.color.logic_op);
    if (S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS) NRED_ONCE("alpha test function %u (a shader's to do in Vulkan, a later slice): not applied", (unsigned)S.color.alpha_func);
    // Alpha to coverage takes colour 0's alpha: Vulkan wants an output at location 0, component 3, Index 0 (VUID-VkGraphicsPipelineCreateInfo-alphaToCoverageEnable-08891). On only where the
    // pixel shader writes all four components at location 0 and has no output of Index other than 0 (sh->dual): the reflection's outputs do not tell an Index, so location 0's four components
    // may be an Index 1 output's with no Index 0 one. Such a shader has it off rather than being refused: refusing it would skip its draws, which draw otherwise (dual-source blend factors are
    // said in red below), and none of Stardew Valley's 65 modules has an Index decoration. A scalar alpha alone (location 0, component 3) needs a Component decoration, which the builder refuses
    k->a2c = S.ms.a2c && ps && (ps->out_mask & 1) && ps->out_n[0] == 4 && !ps->dual;
    if (S.ms.a2c && ps && ps->dual) NRED_ONCE("alpha to coverage on, and the pixel shader %s has an output of Index 1 (dual-source): off", ps->key);
    else if (S.ms.a2c && !k->a2c) NRED_ONCE("alpha to coverage on, and the pixel shader %s writes no alpha at location 0: off", ps ? ps->key : "(none)");
    // Colour: the pass's formats; blending where enabled, the format blends and the shader writes the target; the channel mask where the shader writes the target in its format's class
    k->nc = R.nc;
    for (uint32_t i = 0; i < R.nc; i++) {
        VkFormat f = R.cf[i]; k->cf[i] = (uint32_t)f;
        if (!f) continue;
        int wr = ps && (ps->out_mask >> i & 1) && (R.act >> i & 1);
        if (wr && ps->out_ct[i] != fmt_comp(f)) { NRED_ONCE("shader %s writes target %u as %s, its format %d is %s: not written", ps->key, i, ps->out_ct[i] ? "integers" : "floats", f, fmt_comp(f) ? "integer" : "float"); wr = 0; }
        if (!wr) continue;
        k->wm[i] = (uint8_t)(S.chmask.mask >> (4 * i) & 0xf);
        if (!(S.color.blend_en >> i & 1) || fmt_comp(f) || !(fmt_opt(f, 0) & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)) continue;
        blend_st b = adv_blend(S.blend[i]);
        if ((b.src_rgb >= 0x10 && b.src_rgb <= 0x13) || (b.dst_rgb >= 0x10 && b.dst_rgb <= 0x13) || (b.src_a >= 0x10 && b.src_a <= 0x13) || (b.dst_a >= 0x10 && b.dst_a <= 0x13)) {
            NRED_ONCE("dual-source blending (factors %u %u %u %u: the shaders write no second colour, a later slice): not blended", b.src_rgb, b.dst_rgb, b.src_a, b.dst_a); continue; }
        k->ben[i] = 1; k->bsc[i] = (uint8_t)bfac(b.src_rgb); k->bdc[i] = (uint8_t)bfac(b.dst_rgb); k->bec[i] = (uint8_t)bop(b.eq_rgb);
        k->bsa[i] = (uint8_t)bfac(b.src_a); k->bda[i] = (uint8_t)bfac(b.dst_a); k->bea[i] = (uint8_t)bop(b.eq_a);
    }
    if (!vd.indep_blend) for (uint32_t i = 1; i < k->nc; i++) if (k->ben[i] != k->ben[0] || k->wm[i] != k->wm[0] || k->bsc[i] != k->bsc[0] || k->bdc[i] != k->bdc[0] || k->bec[i] != k->bec[0] ||
                                                                 k->bsa[i] != k->bsa[0] || k->bda[i] != k->bda[0] || k->bea[i] != k->bea[0]) {
        NRED_ONCE("targets blend or mask differently, and the device has no independentBlend: all take target 0's");
        k->ben[i] = k->ben[0]; k->wm[i] = k->wm[0]; k->bsc[i] = k->bsc[0]; k->bdc[i] = k->bdc[0]; k->bec[i] = k->bec[0]; k->bsa[i] = k->bsa[0]; k->bda[i] = k->bda[0]; k->bea[i] = k->bea[0]; }
    // Depth and stencil (nvn_mtl.m dss_get: a depth write only with the test on); NVN's compare functions and stencil operations are Vulkan's plus one
    if (R.hd) { k->df = (uint32_t)R.dsf; if (S.ds.dt) { k->dt = 1; k->dw = S.ds.dw; k->dcmp = (uint8_t)(S.ds.df >= 1 && S.ds.df <= 8 ? S.ds.df - 1 : VK_COMPARE_OP_ALWAYS); } }
    if (R.hs) { k->sf = (uint32_t)R.dsf;
        if (S.ds.st) { k->st = 1; const unsigned c[2] = {S.ds.sf_f, S.ds.sf_b}, fl[2] = {S.ds.fail_f, S.ds.fail_b}, zf[2] = {S.ds.zfail_f, S.ds.zfail_b}, zp[2] = {S.ds.zpass_f, S.ds.zpass_b};
            for (int f = 0; f < 2; f++) { k->scmp[f] = (uint8_t)(c[f] >= 1 && c[f] <= 8 ? c[f] - 1 : VK_COMPARE_OP_ALWAYS); k->sfail[f] = (uint8_t)(fl[f] >= 1 && fl[f] <= 8 ? fl[f] - 1 : 0);
                                          k->szfail[f] = (uint8_t)(zf[f] >= 1 && zf[f] <= 8 ? zf[f] - 1 : 0); k->szpass[f] = (uint8_t)(zp[f] >= 1 && zp[f] <= 8 ? zp[f] - 1 : 0); } } }
    // DM_VK_DIAG=nodepth,nostencil,nocull: those tests off in every pipeline (diagnostics only: which test rejects a pass's geometry)
    { static const char *dg = (const char *)1; if (dg == (const char *)1) { dg = getenv("DM_VK_DIAG"); if (dg) NLOG("[RED] DM_VK_DIAG=%s: diagnostic overrides of the draw state", dg); }
      if (dg && strstr(dg, "nodepth")) { k->dt = 0; k->dw = 0; }
      if (dg && strstr(dg, "nostencil")) k->st = 0;
      if (dg && strstr(dg, "nocull")) k->cull = 0; }
    // Vertex inputs: each the shader reads, from its NVN attribute's stream; one nothing feeds (no attribute, or no buffer bound) reads zeros from the zero buffer at a stream index no fed input uses
    uint32_t used = 0; *zbind = UINT32_MAX;
    for (int i = 0; i < vs->nin; i++) { int l = vs->in[i].loc; va_st a = l < S.nattr ? S.attr[l] : (va_st){0}; if (a.fmt && a.stream < 16 && S.vbuf[a.stream].address) used |= 1u << a.stream; }
    for (int i = 0; i < vs->nin; i++) {
        int l = vs->in[i].loc, ct = vs->in[i].ct; va_st a = l < S.nattr ? S.attr[l] : (va_st){0};
        k->in[i].loc = (uint8_t)l;
        if (a.fmt && a.stream < 16 && S.vbuf[a.stream].address) {
            VkFormat f = vtx_fmt(a.fmt, ct);
            if (!f || !(fmt_opt(f, 1) & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) { NRED_ONCE("vertex format %#x (as %d) for a %s input: the device cannot fetch it, the draws are skipped", a.fmt, f, ct ? "integer" : "float"); return 0; }
            if (a.offset > vd.props.limits.maxVertexInputAttributeOffset) { NRED_ONCE("vertex attribute offset %u past the device's %u: skipped", a.offset, vd.props.limits.maxVertexInputAttributeOffset); return 0; }
            k->in[i].fmt = (uint32_t)f; k->in[i].off = a.offset; k->in[i].bind = (uint8_t)a.stream;
        } else {
            if (*zbind == UINT32_MAX) { for (int s = 15; s >= 0 && *zbind == UINT32_MAX; s--) if (!(used >> s & 1)) *zbind = (uint32_t)s; }
            k->in[i].fmt = (uint32_t)(ct == 1 ? VK_FORMAT_R32G32B32A32_UINT : ct == 2 ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT); k->in[i].bind = (uint8_t)*zbind;
        }
    }
    k->nin = (uint8_t)vs->nin;
    for (int s = 0; s < 16; s++) if (used >> s & 1) {
        vs_st t = S.stream[s];
        if (t.divisor > 1) { NRED_ONCE("vertex stream divisor %u (VK_EXT_vertex_attribute_divisor is a later slice): skipped", t.divisor); return 0; }
        if (t.stride > vd.props.limits.maxVertexInputBindingStride) { NRED_ONCE("vertex stride %u past the device's %u: skipped", t.stride, vd.props.limits.maxVertexInputBindingStride); return 0; }
        k->bd[k->nbd].idx = (uint8_t)s; k->bd[k->nbd].stride = t.stride; k->bd[k->nbd++].rate = t.divisor != 0;
    }
    if (*zbind != UINT32_MAX) { k->bd[k->nbd].idx = (uint8_t)*zbind; k->bd[k->nbd++].stride = 0; }
    return 1;
}

// ---- Descriptors: every binding of the pair gets one (a dummy where nothing usable is bound), pushed at each draw ----
// [addr, ...) in a pool with a shadow (nvn_vk.c's own lookup, under g_pmu): the shadow, the offset in it, the pool's own bytes from addr on; the shadow's last use is this submission
static int b_use(uint64_t addr, VkBuffer *buf, uint64_t *off, uint64_t *avail) {
    int ok = 0;
    pthread_mutex_lock(&g_pmu);
    int i = g_plast >= 0 && g_plast < g_npool && g_pools[g_plast].sh && in_pool(g_plast, addr, 1) ? g_plast : -1;
    if (i < 0 && (i = pool_find(addr, 1, 1)) >= 0) g_plast = i;
    if (i >= 0) { b_shadow *s = g_pools[i].sh; *buf = s->buf; *off = addr - g_pools[i].lo; *avail = g_pools[i].end - addr; s->use_nvn = atomic_load(&g_fence) + 1; ok = 1; }
    pthread_mutex_unlock(&g_pmu);
    return ok;
}
#define RING_B (4u << 20)   // 4 MiB: unaligned uniform buffers, unaligned storage buffers and the c0 banks (draw_drv0) of a submission
static int ring_take(uint64_t n, VkBuffer *buf, uint64_t *off, uint8_t **p) {   // n bytes of this slot's ring at the device's uniform alignment (the slot's last submission has completed: mtl_sub_begin)
    sub_slot *s = g_cur; uint64_t al = vd.props.limits.minUniformBufferOffsetAlignment ? vd.props.limits.minUniformBufferOffsetAlignment : 256;
    if (vd.props.limits.minStorageBufferOffsetAlignment > al) al = vd.props.limits.minStorageBufferOffsetAlignment;   // storage buffers are copied here too (draw_sbuf)
    if (!s->ring && !s->ring_no) { void *m; if (buf_host(RING_B, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &s->ring, &s->ring_mem, &m)) s->ring_map = m; else { s->ring_no = 1; NRED_ONCE("the uniform ring cannot be made: draws whose uniform buffer is at an unaligned address are skipped"); } }
    if (!s->ring) return 0;
    uint64_t o = (s->ring_used + al - 1) & ~(al - 1);
    if (o + n > RING_B) { NRED_ONCE("more than %u KiB of unaligned uniform buffers in a submission: the draws past it are skipped", RING_B >> 10); return 0; }
    s->ring_used = o + n; *buf = s->ring; *off = o; *p = s->ring_map + o;
    return 1;
}
// A uniform binding: NVN slot reg of the stage, bound with its size, cut to the pool's own end and the device's range. At an offset the device cannot bind (minUniformBufferOffsetAlignment),
// its bytes are copied now into the ring (as they are at the replay; the shadow has them as of the commit: the same moment of the submission). 0 = the draw is skipped
// A storage binding: NVN storage slot reg of the stage, from the pool's shadow (the GPU's writes stay in the shadow: strategy B). At an offset the device cannot bind
// (minStorageBufferOffsetAlignment, 64 on the RG 476H), its bytes are copied into the ring: read correctly, written to the copy only (said once). Unbound: the zero buffer
static int draw_sbuf(int st, const vkn_shader *sh, const vkn_sbind *b, VkDescriptorBufferInfo *o) {
    const p_bind_buf *u = b->kind == VKN_B_UBOP ? &S.ubo[st][b->reg] : &S.ssbo[st][b->reg]; VkBuffer buf; uint64_t off, avail;   // UBOP: a uniform slot as storage
    if (!u->addr || !u->size) { NRED_ONCE("shader %s reads storage slot %u of stage %d, which has no buffer bound: it reads zeros", sh->key, b->reg, st); *o = (VkDescriptorBufferInfo){g_zbuf, 0, ZBUF_B}; DS(g_ddum, 1); return 1; }
    if (!b_use(u->addr, &buf, &off, &avail)) { NRED_ONCE("storage buffer %#llx (stage %d slot %u) is in no pool with a shadow: it reads zeros", (unsigned long long)u->addr, st, b->reg); *o = (VkDescriptorBufferInfo){g_zbuf, 0, ZBUF_B}; DS(g_ddum, 1); return 1; }
    uint64_t r = u->size < avail ? u->size : avail, al = vd.props.limits.minStorageBufferOffsetAlignment;
    if (al && off % al) {
        if (b->rw) NRED_ONCE("shader %s writes storage buffer %#llx (stage %d slot %u) at an offset the device cannot bind: its writes go to a copy", sh->key, (unsigned long long)u->addr, st, b->reg);
        uint8_t *p; if (r > RING_B / 4 || !ring_take(r, &buf, &off, &p)) return 0;
        memcpy(p, (const void *)(uintptr_t)u->addr, (size_t)r);
    }
    *o = (VkDescriptorBufferInfo){buf, off, r};
    return 1;
}
// Driver bank c0 of stage st, as nvn_mtl.m fills it: the storage buffer table at the stage's base (0x110 vertex, 0x510 pixel, 0x310 compute; address 64 bits, size 32 bits per slot),
// for compute the dispatch size at 0x200 and the uniform buffer table at 0x210, for the graphics stages the largest point size at 0x610 (base instance 0x644 stays 0: such draws are
// skipped). Translated shaders take global memory addresses relative to the table: left at zero, Paper Mario: The Thousand-Year Door's lighting compute shader lost the GPU
static uint32_t g_disp_dim[3];
#define DRV0_B 0x800u
static int draw_drv0(int st, VkDescriptorBufferInfo *o) {
    VkBuffer buf; uint64_t off; uint8_t *d;
    if (!ring_take(DRV0_B, &buf, &off, &d)) return 0;
    memset(d, 0, DRV0_B);
    int base = st == 0 ? 0x110 : st == 1 ? 0x510 : st == 5 ? 0x310 : -1;
    if (base < 0) NRED_ONCE("stage %d reads driver bank c0: its storage buffer table base was never measured (zeros)", st);
    else for (int r = 0; r < 16; r++) { uint64_t ad = S.ssbo[st][r].addr; uint32_t n = (uint32_t)S.ssbo[st][r].size; memcpy(d + base + 16 * r, &ad, 8); memcpy(d + base + 16 * r + 8, &n, 4); }
    if (st == 5) { memcpy(d + 0x200, g_disp_dim, 12); for (int r = 0; r < 16; r++) { uint64_t ad = S.ubo[5][r].addr; uint32_t n = (uint32_t)S.ubo[5][r].size; memcpy(d + 0x210 + 16 * r, &ad, 8); memcpy(d + 0x210 + 16 * r + 8, &n, 4); } }
    else { const float max_point = 2047.0f; memcpy(d + 0x610, &max_point, 4); }
    *o = (VkDescriptorBufferInfo){buf, off, DRV0_B};
    return 1;
}
// Driver bank c2 of stage st, as nvn_mtl.m fills it: the image handles (0x120), the combined texture handles (0x20, slots 0-40; a slot past 31 only where its image slot is empty),
// the separate texture (0x168) and sampler (0x568) handles. For a shader with dynamic handle arrays (sh->dyn, profile key tex_dyn_slots) the tables hold slot indices instead:
// texture slot r → r, and r << 20 when a combined handle carries a sampler; sampler slot r → r << 20; the shader's own arithmetic then picks array element r (draw_dyn)
enum { C2_TEX = 0x20, C2_TEX_SLOTS = 41, C2_IMG = 0x120, C2_IMG_SLOTS = 8, C2_SEPTEX = 0x168, C2_SAMP = 0x568, DRV2_B = 0xA00 };
static int draw_drv2(int st, const vkn_shader *sh, VkDescriptorBufferInfo *o) {
    VkBuffer buf; uint64_t off; uint8_t *d;
    if (!ring_take(DRV2_B, &buf, &off, &d)) return 0;
    memset(d, 0, DRV2_B);
    const uint64_t *tx = T.tex[st], *im = S.img[st]; int dyn = sh->dyn;
    for (int r = 0; r < C2_IMG_SLOTS; r++) { uint32_t v = (uint32_t)im[r]; memcpy(d + C2_IMG + 8 * r, &v, 4); }
    for (int r = 0; r < C2_TEX_SLOTS; r++) {
        if (r >= 32 && (!tx[r] || im[r - 32])) continue;
        uint32_t v = !dyn ? (uint32_t)tx[r] : !tx[r] ? 0 : (uint32_t)r | (tx[r] >> 20 ? (uint32_t)r << 20 : 0); memcpy(d + C2_TEX + 8 * r, &v, 4); }
    for (int r = 0; r < 128; r++) { uint64_t h = tx[NVN_SEPTEX_SLOT0 + r]; uint32_t v = !dyn ? (uint32_t)h : h ? (uint32_t)r : 0; memcpy(d + C2_SEPTEX + 8 * r, &v, 4); }
    for (int r = 0; r < 128; r++) { uint64_t h = T.samp[st][r]; uint32_t v = !dyn ? (uint32_t)h : h ? (uint32_t)r << 20 : 0; memcpy(d + C2_SAMP + 8 * r, &v, 4); }
    *o = (VkDescriptorBufferInfo){buf, off, DRV2_B};
    return 1;
}
static uint32_t g_xt[6][32], g_xs[6][32];   // by stage: element k of the draw's or dispatch's TEXD / SAMPD arrays holds this texture / sampler ID read from memory, 0 = what slot k holds (idmap_extra)
// A dynamic handle array's elements (TEXD: element k = the texture of combined slot k, else separate slot k, as the array's view type; one of another kind or none: the dummy;
// SAMPD: the separate sampler of slot k, else the sampler half of combined slot k; either: an ID idmap_extra put there). out: b->n entries
static void draw_dyn(int st, const vkn_sbind *b, VkDescriptorImageInfo *out) {
    g_tl_from = "draw_dyn";
    for (int k = 0; k < b->n; k++) {
        if (b->kind == VKN_B_TEXD) {
            int why = 0; VkImageView v = k < 32 && g_xt[st][k] ? tex_lookup((int)g_xt[st][k], b->vt, b->comp, &why, st) : VK_NULL_HANDLE;   // a texture read from memory put here (idmap_extra), when it is of this array's kind
            if (v) { out[k] = (VkDescriptorImageInfo){VK_NULL_HANDLE, v, SR_LAYOUT}; continue; }
            uint64_t h = k < 128 ? (T.tex[st][k] ? T.tex[st][k] : T.tex[st][NVN_SEPTEX_SLOT0 + k]) : 0; int tid = (int)(h & 0xfffff); why = 0;
            v = tid ? tex_lookup(tid, b->vt, b->comp, &why, st) : VK_NULL_HANDLE;
            if (!v && why) { static int n; if (n++ < 12) NLOG("dynamic array element %d (stage %d, view type %d, class %d): texture ID %d reads the dummy, reason %d (%s) (first cases)", k, st, b->vt, b->comp, tid, why, tl_what()); }
            if (!v) v = g_dview[b->comp][b->vt];
            out[k] = (VkDescriptorImageInfo){VK_NULL_HANDLE, v, SR_LAYOUT};
        } else {
            int sid = k < 32 && g_xs[st][k] ? (int)g_xs[st][k] : k < 128 ? (int)(T.samp[st][k] >> 20) : 0;
            if (!sid && k < 256) sid = (int)(T.tex[st][k] >> 20);
            VkSampler sm = sid ? samp_lookup(sid) : VK_NULL_HANDLE;
            out[k] = (VkDescriptorImageInfo){sm ? sm : g_dsamp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        }
    }
}
// The 32-word push constant table of a shader whose array indices look up handles read from memory (vk_shader.c spv_idmap): word k is what element k of its TEXD / SAMPD arrays
// holds (draw_dyn), slot k's texture ID | its sampler ID << 20
static uint32_t idm_word(int st, int k) {   // slot k's word of that table: its texture ID | its sampler ID << 20
    uint64_t h = T.tex[st][k] ? T.tex[st][k] : T.tex[st][NVN_SEPTEX_SLOT0 + k];
    uint32_t sid = (uint32_t)(T.samp[st][k] >> 20); if (!sid) sid = (uint32_t)(T.tex[st][k] >> 20);
    return (uint32_t)(h & 0xfffff) | sid << 20;
}
static void draw_idmap(VkCommandBuffer cb, VkPipelineLayout pl, VkShaderStageFlags sf, int st) {
    uint32_t t[32];
    for (int k = 0; k < 32; k++) { uint32_t w = idm_word(st, k); t[k] = (g_xt[st][k] ? g_xt[st][k] : w & 0xfffff) | (g_xs[st][k] ? g_xs[st][k] : w >> 20) << 20; }
    vk.vkCmdPushConstants(cb, pl, sf, 0, sizeof t, t);
}
// The handles a shader reads from memory at an address the host can work out (vkn_shader.hs, vk_shader.c spv_hsrc), read from guest memory now (inline uniform updates replayed so
// far are there). A texture (or sampler) ID no slot holds is put in the highest element whose own is free or also held by a lower slot (the search takes the lowest match, so that
// element was never found by it), and draw_idmap's table says so; an array whose view type the texture is not keeps the slot's own there (draw_dyn), and only a slot named by c2 reads
// it directly. Paper Mario: The Thousand-Year Door: the S block's material names Rogueport's reflection cubes (texture IDs 392 / 393) in every room, and only Rogueport binds them:
// elsewhere the face read element 31 (no cube: black, no colours). Before the descriptors are written. DM_VK_IDMAP_MEM=0: none (vk_shader.c)
static uint32_t ubo_handle(int st, int reg, uint32_t off);
static void idmap_extra(int st, const vkn_shader *sh) {
    memset(g_xt[st], 0, sizeof g_xt[st]); memset(g_xs[st], 0, sizeof g_xs[st]);
    for (int i = 0; sh && i < sh->nhs; i++) {
        uint32_t v[16]; int n = 0, ok = 1;
        for (int j = 0; j + 1 < sh->hs[i].n && ok; j += 2) { uint32_t op = sh->hs[i].c[j], a = sh->hs[i].c[j + 1];
            if (op == VKN_HS_CONST || op == VKN_HS_UBO) { if (n >= 16) ok = 0; else v[n++] = op == VKN_HS_CONST ? a : ubo_handle(st, (int)(a >> 16), a & 0xffff); continue; }
            if (n < 2) { ok = 0; break; }
            uint32_t y = v[--n], x = v[n - 1];
            v[n - 1] = op == VKN_HS_ADD ? x + y : op == VKN_HS_SUB ? x - y : op == VKN_HS_MUL ? x * y : op == VKN_HS_SHL ? x << (y & 31) : op == VKN_HS_SHR ? x >> (y & 31)
                     : op == VKN_HS_SAR ? (uint32_t)((int32_t)x >> (y & 31)) : op == VKN_HS_AND ? x & y : x | y; }
        const p_bind_buf *sb = &S.ssbo[st][sh->hs[i].reg]; uint32_t h = 0;
        if (!ok || n != 1 || !sb->addr || (uint64_t)v[0] * 4 + 4 > sb->size) continue;
        memcpy(&h, (const void *)(uintptr_t)(sb->addr + (uint64_t)v[0] * 4), 4);
        uint32_t id[2] = {h & 0xfffff, h >> 20};   // texture, sampler
        for (int part = 0; part < 2; part++) {
            uint32_t x = id[part], (*g)[32] = part ? g_xs : g_xt; if (!x) continue;
            int held = 0, fr = -1;
            for (int k = 0; k < 32; k++) { uint32_t w = idm_word(st, k); held |= (g[st][k] ? g[st][k] : part ? w >> 20 : w & 0xfffff) == x; }
            if (held) continue;
            for (int k = 31; k >= 0 && fr < 0; k--) { if (g[st][k]) continue;
                uint32_t own = part ? idm_word(st, k) >> 20 : idm_word(st, k) & 0xfffff; int dup = !own;
                for (int q = 0; q < k && !dup; q++) dup = (part ? idm_word(st, q) >> 20 : idm_word(st, q) & 0xfffff) == own;
                if (dup) fr = k; }
            if (fr < 0) { NRED_ONCE("shader %s reads %s ID %u from storage slot %u, which no slot holds, and every element of its arrays is a slot's own: it reads element 31", sh->key, part ? "sampler" : "texture", x, sh->hs[i].reg); continue; }
            g[st][fr] = x;
            { static int nl; if (nl < 8) { nl++; NLOG("shader %s reads %s ID %u from storage slot %u word %u, which no slot holds: element %d (first cases)", sh->key, part ? "sampler" : "texture", x, sh->hs[i].reg, v[0], fr); } }
        }
    }
}
// A handle read from the stage's uniform slot reg at byte off (TEXU, nvn_mtl.m ubo_handle / ubo_tex_id / ubo_samp_id): the texture ID in its low 20 bits, the sampler ID above,
// each in the pools bound now. Read from guest memory (an inline update of the buffer within this submission is not seen). Paper Mario: The Thousand-Year Door's deferred
// lighting reads its lights' shadow and pattern textures this way (uniform slot 11): refused, Rogueport's lit pass was skipped and the town stayed dark
static uint32_t ubo_handle(int st, int reg, uint32_t off) {
    const p_bind_buf *u = &S.ubo[st][reg]; uint32_t h = 0;
    if (u->addr && (uint64_t)off + 4 <= u->size) memcpy(&h, (const void *)(uintptr_t)(u->addr + off), 4);
    return h;
}
static void draw_texu(int st, const vkn_sbind *b, VkDescriptorImageInfo *out) {   // b->n entries: element k from the handle at off + 8k
    g_tl_from = "draw_texu";
    for (int k = 0; k < b->n; k++) {
        uint32_t h = ubo_handle(st, b->reg, b->off + 8u * (uint32_t)k); int tid = h & 0xfffff ? (int)(h & 0xfffff) + S.tex_base : 0, sid = h >> 20 ? (int)(h >> 20) + S.samp_base : 0, why = 0;
        VkImageView v = tid ? tex_lookup(tid, b->vt, b->comp, &why, st) : VK_NULL_HANDLE; VkSampler sm = sid ? samp_lookup(sid) : VK_NULL_HANDLE;
        out[k] = (VkDescriptorImageInfo){sm ? sm : g_dsamp, v ? v : g_dview[b->comp][b->vt], SR_LAYOUT};
    }
}
static VkDescriptorImageInfo g_dyn_ii[8 * 64]; static int g_dyn_n;   // the array elements of the draw or dispatch being written (8 arrays of up to 64)
static VkDescriptorImageInfo *dyn_take(int n) { if (g_dyn_n + n > (int)(sizeof g_dyn_ii / sizeof *g_dyn_ii)) return NULL; VkDescriptorImageInfo *p = g_dyn_ii + g_dyn_n; g_dyn_n += n; return p; }
static int draw_ubo(int st, const vkn_shader *sh, const vkn_sbind *b, VkDescriptorBufferInfo *o) {
    uint64_t lim = vd.props.limits.maxUniformBufferRange < 65536 ? vd.props.limits.maxUniformBufferRange : 65536;
    if (b->kind == VKN_B_C1) { *o = (VkDescriptorBufferInfo){sh->c1_buf, 0, sh->c1_n}; return 1; }
    *o = (VkDescriptorBufferInfo){g_zbuf, 0, ZBUF_B < lim ? ZBUF_B : lim};
    if (b->kind == VKN_B_ZERO) return 1;
    if (b->kind == VKN_B_DRV0) return draw_drv0(st, o);
    if (b->kind == VKN_B_DRV2) return draw_drv2(st, sh, o);
    if (b->kind == VKN_B_SUPPORT) { if (g_supbuf) o->buffer = g_supbuf; return 1; }
    const p_bind_buf *u = &S.ubo[st][b->reg]; VkBuffer buf; uint64_t off, avail;
    if (!u->addr || !u->size) { NRED_ONCE("shader %s reads uniform slot %u of stage %d, which has no buffer bound: it reads zeros", sh->key, b->reg, st); DS(g_ddum, 1); return 1; }
    if (!b_use(u->addr, &buf, &off, &avail)) { NRED_ONCE("uniform buffer %#llx (stage %d slot %u) is in no pool with a shadow: it reads zeros", (unsigned long long)u->addr, st, b->reg); DS(g_ddum, 1); return 1; }
    uint64_t r = u->size;
    if (r > lim || r > avail) { r = r > lim ? lim : r; r = r > avail ? avail : r; DS(g_ubo_short, 1); }
    DS(g_ubo_n, 1);
    uint64_t al = vd.props.limits.minUniformBufferOffsetAlignment;
    if (al && off % al) {
        uint8_t *p; DS(g_ubo_unal, 1);
        if (!ring_take(r, &buf, &off, &p)) return 0;
        memcpy(p, (const void *)(uintptr_t)u->addr, (size_t)r);
    }
    *o = (VkDescriptorBufferInfo){buf, off, r};
    return 1;
}
static void draw_tex_prep(void) {   // a draw with a program: everything said to be resolved again becomes per-slot dirty bits (the slots its shaders sample are resolved, draw_view / draw_samp)
    uint64_t g = atomic_load_explicit(&g_reg_gen, memory_order_acquire);
    if (g != T.gen) { T.gen = g; T.all = 0x3f; }
    for (int st = 0; st < 6; st++) if ((st < 2 || st == 5) && (T.all >> st & 1)) {   // the vertex, pixel and compute stages
        T.all &= ~(1u << st);
        for (int w = 0; w < 4; w++) T.td[st][w] |= T.tb[st][w];
        for (int w = 0; w < 2; w++) { T.tsd[st][w] |= T.tb[st][w]; T.sd[st][w] |= T.sb[st][w]; }
    }
}
static VkImageView draw_view(int st, const vkn_shader *sh, const vkn_sbind *b) {
    g_tl_from = "draw_view";
    int s = b->reg; uint64_t bit = 1ull << (s & 63); uint8_t want = (uint8_t)(1 + b->vt + 3 * b->comp);
    // A combined slot past 31 shares its c2 word with image slot s - 32, and a bound image is what lies there (nvn_mtl.m tex_slot_h; draw_drv2 fills c2 so): a size query of the image
    // a compute shader writes. Paper Mario: The Thousand-Year Door's reflection mip chain (reflection.bfsha's compute) asks slots 32-34 for its three levels' sizes; the empty texture
    // slots gave the dummy's 1x1. Image binds set no dirty bit, so such a slot is resolved at every use. DM_VK_IMGSLOT=0: the texture slot, as before
    static int imgslot = -1; if (imgslot < 0) { const char *e = getenv("DM_VK_IMGSLOT"); imgslot = !(e && *e == '0'); }
    uint64_t ih = imgslot && s >= 32 && s - 32 < C2_IMG_SLOTS ? S.img[st][s - 32] : 0;
    if (!ih && !(T.td[st][s >> 6] & bit) && T.tvt[st][s] == want) return T.tv[st][s];
    T.td[st][s >> 6] &= ~bit; T.tvt[st][s] = want;
    int tid = (int)((ih ? ih : T.tex[st][s]) & 0xfffff), why = 0;
    if (ih) { static int n; if (n++ < 4) NLOG("shader %s: texture slot %d reads image slot %d's texture ID %d (first cases)", sh->key, s, s - 32, tid); }
    VkImageView v = tid ? tex_lookup(tid, b->vt, b->comp, &why, st) : VK_NULL_HANDLE;
    if (why == TL_FEED) NRED_ONCE("shader %s samples texture ID %d (stage %d slot %d, %s), a target of the pass it draws in: the slot reads zeros (a copy is a later slice)", sh->key, tid, st, s, tl_what());
    else if (why == TL_COMP) NRED_ONCE("shader %s samples texture ID %d (stage %d slot %d, %s) as %s, its format is not: the slot reads zeros", sh->key, tid, st, s, tl_what(), b->comp == 1 ? "unsigned integers" : b->comp == 2 ? "signed integers" : "floats");
    else if (why == TL_CUBE) { static int n; if (n++ < 8) NLOG("shader %s samples stage %d slot %d as a cube, and texture ID %d is not one this slice has (cube textures are a later slice): the slot reads zeros", sh->key, st, s, tid); }
    else if (why == TL_3D) { static int n; if (n++ < 8) NLOG("shader %s samples stage %d slot %d as a 3D texture (texture ID %d): 3D textures are a later slice, the slot reads zeros", sh->key, st, s, tid); }
    T.tv[st][s] = v;
    return v;
}
static VkSampler draw_samp(int st, const vkn_sbind *b) {   // a binding read as integers gets the sampler's nearest twin (samp_lookup_as), looked up at each draw (Stardew Valley has none)
    int s = b->sreg; uint64_t bit = 1ull << (s & 63);
    if (b->comp) return samp_lookup_as(b->skind == VKN_S_TEX ? (int)(T.tex[st][s] >> 20) : b->skind == VKN_S_SAMP ? (int)(T.samp[st][s] >> 20) : s, 1);
    if (b->skind == VKN_S_TEX) { if (T.tsd[st][s >> 6] & bit) { T.tsd[st][s >> 6] &= ~bit; T.ts[st][s] = samp_lookup((int)(T.tex[st][s] >> 20)); } return T.ts[st][s]; }
    if (b->skind == VKN_S_SAMP) { if (T.sd[st][s >> 6] & bit) { T.sd[st][s >> 6] &= ~bit; T.ss[st][s] = samp_lookup((int)(T.samp[st][s] >> 20)); } return T.ss[st][s]; }
    return samp_lookup(s);
}

// ---- The draw ----
static struct { int valid; VkViewport vp; VkRect2D sc; float bc[4]; uint32_t cm[2], wm[2], ref[2]; float bias[3]; VkPipeline pso; } D;   // what the command buffer has (reset at each submission)
static int draw_skip(int why) { DS(g_dr_skip[why], 1); return 0; }
static float clamp01(float v) { return v >= 0.0f ? (v <= 1.0f ? v : 1.0f) : 0.0f; }   // NaN → 0
// ---------- Compute (DISPATCH) ----------
// Paper Mario: The Thousand-Year Door lights its 1600x900 scene in a compute shader (100x57 groups of 16x16) that writes two images the rest of the frame samples: without
// dispatches every later pass read zeros and the window stayed black. A compute shader gets its own descriptor set layout and pipeline (made at its first dispatch, kept on the
// shader); its textures, uniform, storage and support buffers bind as a draw's do (stage 5), its storage images are 2D views of the host images in the general layout.
// Indirect dispatches are a later slice. After a dispatch everything later waits for its writes. DM_VK_COMPUTE=0: dispatches are not run (and compute modules refused)
static _Atomic uint64_t g_disp_n, g_disp_skip;
static int cs_make(vkn_shader *sh) {
    if (sh->cpso) return 1;
    if (sh->cbad) return 0;
    VkDescriptorSetLayoutBinding b[VKN_MAXB];
    for (int i = 0; i < sh->nb; i++) b[i] = (VkDescriptorSetLayoutBinding){sh->b[i].binding, bind_type(&sh->b[i]), bind_count(&sh->b[i]), VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo di = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, vkd_layout_flags(), (uint32_t)sh->nb, b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &sh->cdsl, sh->idmap ? 1 : 0, &pr};
    VkResult r = vk.vkCreateDescriptorSetLayout(vd.dev, &di, NULL, &sh->cdsl);
    if (!r) r = vk.vkCreatePipelineLayout(vd.dev, &pi, NULL, &sh->cpl);
    if (!r) {
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sh->mod, "main", NULL}, sh->cpl, VK_NULL_HANDLE, -1};
        uint64_t t0 = now_ns(); r = vk.vkCreateComputePipelines(vd.dev, g_pcache, 1, &ci, NULL, &sh->cpso);
        static int n; if (!r && n++ < 8) NLOG("compute pipeline: shader %s, %d bindings, %.2f ms to make", sh->key, sh->nb, (now_ns() - t0) / 1e6);
    }
    if (r) { sh->cbad = 1; NLOG("[RED] compute shader %s: its layout or pipeline cannot be made (VkResult %d): its dispatches are skipped", sh->key, r); return 0; }
    return 1;
}
// Whether the device writes format fmt through a storage image of Unknown format (VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT: the device feature is not enough on its own); said once per format
static int nofmt_write_ok(VkFormat fmt) {
    static struct { VkFormat f; int ok; } c[16]; static int n;
    for (int i = 0; i < n; i++) if (c[i].f == fmt) return c[i].ok;
    VkFormatProperties3 p3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3}; VkFormatProperties2 p2 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &p3};
    static PFN_vkGetPhysicalDeviceFormatProperties2 gp; if (!gp) gp = (PFN_vkGetPhysicalDeviceFormatProperties2)vk.vkGetInstanceProcAddr(vd.inst, "vkGetPhysicalDeviceFormatProperties2");   // an instance function
    if (gp) gp(vd.pd, fmt, &p2);
    int ok = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT) != 0;
    NLOG("%sformat %d: %s as a storage image of Unknown format", ok ? "" : "[RED] ", fmt, ok ? "written" : "cannot be written (its dispatches are skipped)");
    if (n < 16) { c[n].f = fmt; c[n++].ok = ok; }
    return ok;
}
// A dispatch's descriptor writes (as a draw's, stage 5) and its storage images (imgs, ni). 0 = the dispatch is skipped (said). frag (csf_run): the storage images are not resolved here
// and their writes are left with no descriptor (descriptorCount 0)
static int cs_binds(const vkn_shader *sh, VkDescriptorImageInfo *ii, VkDescriptorBufferInfo *bi, VkWriteDescriptorSet *wr, vkn_img **imgs, int *ni, int frag) {
    g_dyn_n = 0;
    idmap_extra(5, sh->idmap ? sh : NULL);
    for (int i = 0; i < sh->nb; i++) { const vkn_sbind *b = &sh->b[i];
        wr[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, b->binding, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NULL, NULL, NULL};
        if (b->kind == VKN_B_TEX) {
            VkImageView v = draw_view(5, sh, b); VkSampler sm = draw_samp(5, b);
            if (!v) { v = g_dview[b->comp][b->vt]; DS(g_ddum, 1); static int n; if (n++ < 12) NLOG("dispatch of compute shader %s: texture slot %u (texture ID %d) reads the dummy (first cases)", sh->key, b->reg, (int)(T.tex[5][b->reg] & 0xfffff)); }
            ii[i] = (VkDescriptorImageInfo){sm ? sm : g_dsamp, v, SR_LAYOUT}; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr[i].pImageInfo = &ii[i];
        } else if (b->kind == VKN_B_IMG && frag) wr[i].descriptorCount = 0;
        else if (b->kind == VKN_B_IMG) {
            int tid = (int)(S.img[5][b->reg] & 0xfffff); vkn_img *e = NULL; VkImageView v = VK_NULL_HANDLE;
            // The handle's registration may be a view of one level or layer (a mip chain written level by level: Paper Mario: The Thousand-Year Door's 8x8-group pass binds one
            // texture three times, as levels 0-2); writing level 0 through all three lost the GPU on the RG 476H
            pthread_mutex_lock(&g_rmu); uint64_t ser = 0; tex_t tt; uint32_t lv = 0, ly = 0;
            if (tid && tid < g_ntreg && g_treg[tid].ok) { const treg_t *r = &g_treg[tid]; ser = r->t.serial; tt = r->t;
                if (r->has_v && r->v.set_levels) lv = r->v.base_level; if (r->has_v && r->v.set_layers) ly = r->v.min_layer; }
            pthread_mutex_unlock(&g_rmu);
            pthread_mutex_lock(&vkn_tmu);
            if (ser && !(e = vkn_img_get(ser))) { g_mk_storage = 1; e = img_make(&tt, "a compute shader's storage image", NVK_RT); g_mk_storage = 0; }
            if (e && (e->usage & VK_IMAGE_USAGE_STORAGE_BIT) && nofmt_write_ok(e->fmt) && lv < e->levels && ly < e->layers) { use_mark(e); v = vkn_img_view(e, lv, ly, 0); }
            pthread_mutex_unlock(&vkn_tmu);
            if (!v) { NRED_ONCE("compute shader %s: image slot %u (texture ID %d, ser=%llu) has no host image this device can write as storage: the dispatch is skipped", sh->key, b->reg, tid, (unsigned long long)ser); return 0; }
            { static uint64_t seen[32]; static int ns; int k = 0; while (k < ns && seen[k] != ser) k++;   // which images the dispatches write (DM_VK_STORAGE_SER takes this list)
              if (k == ns && ns < 32) { seen[ns++] = ser; NLOG("compute shader %s writes ser=%llu (%ux%u format %d) as a storage image", sh->key, (unsigned long long)ser, e->w, e->h, e->fmt); } }
            imgs[(*ni)++] = e;
            ii[i] = (VkDescriptorImageInfo){VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_GENERAL}; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr[i].pImageInfo = &ii[i];
        } else if (b->kind == VKN_B_TEXD || b->kind == VKN_B_SAMPD || b->kind == VKN_B_TEXU) {
            VkDescriptorImageInfo *ai = dyn_take(b->n); if (!ai) return 0;
            if (b->kind == VKN_B_TEXU) draw_texu(5, b, ai); else draw_dyn(5, b, ai); wr[i].descriptorType = bind_type(b); wr[i].descriptorCount = b->n; wr[i].pImageInfo = ai;
        } else if (b->kind == VKN_B_SBUF || b->kind == VKN_B_UBOP) { if (!draw_sbuf(5, sh, b, &bi[i])) return 0; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i]; }
        else { if (!draw_ubo(5, sh, b, &bi[i])) return 0; wr[i].pBufferInfo = &bi[i]; }
    }
    return 1;
}
static uint64_t g_stat_cs_gen;   // sampled descriptors of a dispatch's own storage images, given the general layout (op_dispatch)
// A host image a dispatch writes, made without STORAGE (storage_mode 2): remade with it, in place. The new image gets the old one's content (every level and layer, copied in this
// submission, outside a pass) and its layout; the old image and its views go to the pending-destruction list as a husk, freed once this submission has completed. Views made of it are
// made again (g_reg_gen). The present reads the image under vkn_tmu and the published layout under vkn_qmu: that layout is set to undefined until this submission's commit, so a present
// in between shows the image black rather than sampling it in a wrong layout. Under vkn_qmu, then vkn_tmu (the present's order)
static uint64_t g_stor_up;
static void cs_storage_up(vkn_img *e) {
    if (!e || storage_mode() != 2 || e->usage & VK_IMAGE_USAGE_STORAGE_BIT || e->aspect != VK_IMAGE_ASPECT_COLOR_BIT || e->sfmt != e->fmt || !(fmt_feat(e->fmt) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) return;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, e->cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, e->depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, e->fmt, {e->w, e->h, e->depth},
        e->levels, e->layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, e->usage | VK_IMAGE_USAGE_STORAGE_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage im; VkDeviceMemory m = VK_NULL_HANDLE; VkMemoryRequirements mr; int mt;
    if (vk.vkCreateImage(vd.dev, &ii, NULL, &im)) { NRED_ONCE("host image ser=%llu could not be remade with storage usage: its dispatches are skipped", (unsigned long long)e->serial); return; }
    vk.vkGetImageMemoryRequirements(vd.dev, im, &mr);
    if ((mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) < 0) mt = vkn_mem_type(mr.memoryTypeBits, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &m) || vk.vkBindImageMemory(vd.dev, im, m, 0)) {
        NRED_ONCE("device memory for host image ser=%llu with storage usage failed: its dispatches are skipped", (unsigned long long)e->serial);
        if (m) vk.vkFreeMemory(vd.dev, m, NULL); vk.vkDestroyImage(vd.dev, im, NULL); return; }
    VkImageLayout L = rec_layout(e); VkCommandBuffer cb = cb_now(); VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (L != VK_IMAGE_LAYOUT_UNDEFINED && e->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        VkImageMemoryBarrier2 b[2] = {
            {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
             L, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img, all},
            {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im, all}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 2, b}; vk.vkCmdPipelineBarrier2(cb, &di);
        VkImageCopy rg[16]; uint32_t n = e->levels < 16 ? e->levels : 16;
        for (uint32_t l = 0; l < n; l++) {
            uint32_t w = e->w >> l, h = e->h >> l, dz = e->depth >> l;
            rg[l] = (VkImageCopy){{VK_IMAGE_ASPECT_COLOR_BIT, l, 0, e->layers}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, e->layers}, {0, 0, 0}, {w ? w : 1, h ? h : 1, dz ? dz : 1}};
        }
        vk.vkCmdCopyImage(cb, e->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, n, rg);
        VkImageMemoryBarrier2 a = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, L, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im, all};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &a; vk.vkCmdPipelineBarrier2(cb, &di);
    } else e->rec = VK_IMAGE_LAYOUT_UNDEFINED;   // nothing worth keeping (or it cannot be copied): the new image starts undefined
    vkn_img *h = calloc(1, sizeof *h); if (!h) die("vkn backend: image entry allocation failed");
    h->serial = e->serial; h->img = e->img; h->mem = e->mem; h->aspect = e->aspect; h->use_pres = e->use_pres; use_mark(h);
    memcpy(h->views, e->views, sizeof e->views); h->nviews = e->nviews; memcpy(h->sviews, e->sviews, sizeof e->sviews); h->nsviews = e->nsviews;
    img_dead(h);
    e->img = im; e->mem = m; e->nviews = e->nsviews = 0; e->usage |= VK_IMAGE_USAGE_STORAGE_BIT; e->layout = VK_IMAGE_LAYOUT_UNDEFINED; g_img_b += mr.size;
    atomic_fetch_add_explicit(&g_reg_gen, 1, memory_order_release);
    if (g_stor_up++ < 24) NLOG("host image ser=%llu %ux%u format %d remade with storage usage for a dispatch (its content %s)", (unsigned long long)e->serial, e->w, e->h, e->fmt,
                               L != VK_IMAGE_LAYOUT_UNDEFINED && e->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT ? "copied" : "undefined");
}
// The storage images a dispatch binds (as cs_binds finds them): each made without STORAGE is remade with it first (cs_storage_up)
static void cs_storage_prep(const vkn_shader *sh) {
    if (storage_mode() != 2) return;
    for (int i = 0; i < sh->nb; i++) { const vkn_sbind *b = &sh->b[i]; if (b->kind != VKN_B_IMG) continue;
        int tid = (int)(S.img[5][b->reg] & 0xfffff); uint64_t ser = 0;
        pthread_mutex_lock(&g_rmu); if (tid && tid < g_ntreg && g_treg[tid].ok) ser = g_treg[tid].t.serial; pthread_mutex_unlock(&g_rmu);
        if (!ser) continue;
        pthread_mutex_lock(&vkn_qmu); pthread_mutex_lock(&vkn_tmu);
        vkn_img *e = vkn_img_get(ser); if (e && !(e->usage & VK_IMAGE_USAGE_STORAGE_BIT)) cs_storage_up(e);
        pthread_mutex_unlock(&vkn_tmu); pthread_mutex_unlock(&vkn_qmu);
    }
}
// ---- A dispatch as a fragment pass (DM_VK_CS_FRAG=0: never) ----
// A compute shader that writes each of its storage images once, at its own invocation's pixel, has a fragment form (vk_shader.c spv_cs2fs checks that and makes it): its dispatch runs
// as one triangle over those images as colour attachments instead. A dispatch's images need storage usage, and on Mali that keeps them uncompressed (no AFBC) for every later pass
// that reads or draws into them: Paper Mario: The Thousand-Year Door lights its 960x540 scene this way into the two images the rest of the frame draws on (RESEARCH_mali_shaders.md:
// 34.1 → 32.4 ms a frame on the RG 476H). The images must have one extent, which the pass covers; the groups must cover it with none wholly outside it (that group's other effects,
// a buffer write, would be lost). Where they do not, where an image has an integer format, where the dispatch samples one of its images, or where something cannot be made, it is
// dispatched (said)
static uint64_t g_csf_n, g_csf_no;
static int csf_no(vkn_shader *sh, int perm, const char *why) {
    if (perm) sh->fbad = 1;
    if (g_csf_no++ < 8) NLOG("%scompute shader %s: dispatched, not run as a pass (%s)", perm ? "[RED] " : "", sh->key, why);
    return 0;
}
static VkPipeline csf_pso(vkn_shader *sh, const VkFormat *cf, uint32_t nc) {
    for (int i = 0; i < sh->nfp; i++) if (sh->fp[i].n == nc && !memcmp(sh->fp[i].f, cf, nc * sizeof *cf)) return sh->fp[i].p;
    if (sh->nfp == 4 || !vkn_tri_vs()) return VK_NULL_HANDLE;
    if (!sh->fpl) {   // the compute layout's bindings but the storage images', for the fragment stage
        VkDescriptorSetLayoutBinding b[VKN_MAXB]; uint32_t n = 0;
        for (int i = 0; i < sh->nb; i++) if (sh->b[i].kind != VKN_B_IMG) b[n++] = (VkDescriptorSetLayoutBinding){sh->b[i].binding, bind_type(&sh->b[i]), bind_count(&sh->b[i]), VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo di = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, vkd_layout_flags(), n, b};
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};
        VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &sh->fdsl, sh->idmap ? 1 : 0, &pr};
        if (vk.vkCreateDescriptorSetLayout(vd.dev, &di, NULL, &sh->fdsl)) return VK_NULL_HANDLE;
        if (vk.vkCreatePipelineLayout(vd.dev, &pi, NULL, &sh->fpl)) { sh->fpl = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
    }
    VkPipelineShaderStageCreateInfo stg[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vkn_tri_vs(), "main", NULL},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sh->fmod, "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba[8];
    for (uint32_t i = 0; i < nc; i++) ba[i] = (VkPipelineColorBlendAttachmentState){VK_FALSE, 0, 0, 0, 0, 0, 0, VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_LOGIC_OP_COPY, nc, ba, {0, 0, 0, 0}};
    static const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, nc, cf, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, 2, stg, &vi, &ia, NULL, &vps, &rs, &ms, NULL, &cbs, &dy, sh->fpl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VkPipeline p; uint64_t t0 = now_ns();
    if (vk.vkCreateGraphicsPipelines(vd.dev, g_pcache, 1, &gi, NULL, &p)) return VK_NULL_HANDLE;
    NLOG("compute shader %s: its pass's pipeline (%u colour attachments) made in %.2f ms", sh->key, nc, (now_ns() - t0) / 1e6);
    memcpy(sh->fp[sh->nfp].f, cf, nc * sizeof *cf); sh->fp[sh->nfp].n = nc; sh->fp[sh->nfp].p = p;
    return sh->fp[sh->nfp++].p;
}
// 1 = the dispatch ran as a pass; 0 = it is to be dispatched. After rp_flush, pend_all and draw_tex_prep
static int csf_run(vkn_shader *sh, const p_dispatch *d) {
    if (sh->fbad || !vkn_shader_frag(sh)) return 0;
    uint32_t lx = sh->lsz[0], ly = sh->lsz[1];
    if (!lx || !ly || sh->lsz[2] != 1) return csf_no(sh, 1, "its local size is not 2D");
    if (d->z != 1) return csf_no(sh, 0, "a dispatch more than one group deep");
    const vkn_sbind *ob[8]; uint32_t no = 0;   // its storage images by binding: output location k is the k-th (tools/cs2fs.py)
    for (int i = 0; i < sh->nb; i++) { if (sh->b[i].kind != VKN_B_IMG) continue;
        if (no == 8) return csf_no(sh, 1, "more than 8 storage images");
        uint32_t k = no++; while (k && ob[k - 1]->binding > sh->b[i].binding) { ob[k] = ob[k - 1]; k--; } ob[k] = &sh->b[i]; }
    vkn_img *oe[8]; uint32_t olv[8], oly[8]; VkExtent2D ex = {0, 0}, oex[8];
    for (uint32_t k = 0; k < no; k++) {
        int tid = (int)(S.img[5][ob[k]->reg] & 0xfffff); uint64_t ser = 0; tex_t tt; uint32_t lv = 0, ly0 = 0;
        pthread_mutex_lock(&g_rmu);
        if (tid && tid < g_ntreg && g_treg[tid].ok) { const treg_t *r = &g_treg[tid]; ser = r->t.serial; tt = r->t;
            if (r->has_v && r->v.set_levels) lv = r->v.base_level; if (r->has_v && r->v.set_layers) ly0 = r->v.min_layer; }
        pthread_mutex_unlock(&g_rmu);
        pthread_mutex_lock(&vkn_tmu);
        vkn_img *e = ser ? vkn_img_get(ser) : NULL; if (ser && !e) e = img_make(&tt, "a compute shader's output, as a target", NVK_RT);
        int ok = e && e->aspect == VK_IMAGE_ASPECT_COLOR_BIT && att_ok(e) && lv < e->levels && ly0 < e->layers && e->depth <= 1; if (ok) use_mark(e);
        pthread_mutex_unlock(&vkn_tmu);
        if (!ok) return csf_no(sh, 0, "an image it writes is no colour target here");
        if (fmt_comp(e->fmt)) return csf_no(sh, 0, "an image it writes has an integer format (the outputs are floats)");
        oe[k] = e; olv[k] = lv; oly[k] = ly0; oex[k] = lv_ext(e, lv);
        if (k && (oex[k].width != ex.width || oex[k].height != ex.height)) return csf_no(sh, 0, "the images it writes differ in extent (the pass covers one)");
        ex = oex[k];
    }
    if ((uint64_t)d->x * lx < ex.width || (uint64_t)(d->x - 1) * lx >= ex.width || (uint64_t)d->y * ly < ex.height || (uint64_t)(d->y - 1) * ly >= ex.height)
        return csf_no(sh, 0, "its groups do not cover its images' extent, or some lie outside it");
    VkDescriptorImageInfo ii[VKN_MAXB]; VkDescriptorBufferInfo bi[VKN_MAXB]; VkWriteDescriptorSet wr[VKN_MAXB]; int ni = 0;
    g_in_dispatch = 1; int ok = cs_binds(sh, ii, bi, wr, NULL, &ni, 1); g_in_dispatch = 0;
    if (!ok) return 0;   // said there; the dispatch says it again and is skipped
    uint32_t nw = 0;
    for (int w = 0; w < sh->nb; w++) {
        if (!wr[w].descriptorCount) continue;
        if (wr[w].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || wr[w].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
            for (uint32_t a = 0; a < wr[w].descriptorCount; a++) for (uint32_t k = 0; k < no; k++) {
                const vkn_img *e = oe[k]; VkImageView q = wr[w].pImageInfo[a].imageView;
                for (int v = 0; v < e->nviews; v++) if (e->views[v].v == q) return csf_no(sh, 0, "it samples an image it writes");
                for (int v = 0; v < e->nsviews; v++) if (e->sviews[v].v == q) return csf_no(sh, 0, "it samples an image it writes");
            }
        wr[nw++] = wr[w];
    }
    VkFormat cf[8]; for (uint32_t k = 0; k < no; k++) cf[k] = oe[k]->fmt;
    VkPipeline p = csf_pso(sh, cf, no);
    if (!p) return csf_no(sh, 1, "its pass's pipeline cannot be made");
    VkRenderingAttachmentInfo ca[8]; VkImageMemoryBarrier2 b[8];
    pthread_mutex_lock(&vkn_tmu);
    for (uint32_t k = 0; k < no; k++) {
        VkImageView v = vkn_img_view(oe[k], olv[k], oly[k], 0);
        if (!v) { pthread_mutex_unlock(&vkn_tmu); return csf_no(sh, 0, "an attachment view cannot be made"); }
        ca[k] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        ca[k].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   // every pixel is written
        ca[k].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    pthread_mutex_unlock(&vkn_tmu);
    VkCommandBuffer cb = cb_now();
    vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); D.valid = 0;   // the draws' cached state is not the command buffer's now
    if (nw && !vkd_write_bp(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, sh->fpl, sh->fdsl, nw, wr)) return 0;
    if (sh->idmap) draw_idmap(cb, sh->fpl, VK_SHADER_STAGE_FRAGMENT_BIT, 5);
    for (uint32_t k = 0; k < no; k++) att_barrier(&b[k], oe[k]);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT};   // what it reads, and its buffer writes after earlier ones
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &mb, 0, NULL, no, b};
    vk.vkCmdPipelineBarrier2(cb, &di);
    VkViewport vp = {0, 0, (float)ex.width, (float)ex.height, 0, 1}; VkRect2D sc = {{0, 0}, ex};
    vk.vkCmdSetViewport(cb, 0, 1, &vp); vk.vkCmdSetScissor(cb, 0, 1, &sc);
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, sc, 1, 0, no, ca, NULL, NULL};
    vk.vkCmdBeginRendering(cb, &ri); vk.vkCmdDraw(cb, 3, 1, 0, 0); vk.vkCmdEndRendering(cb);
    mem_barrier(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);   // its buffer writes; the images go by their tracked layout
    if (g_csf_n++ < 6) NLOG("dispatch %dx%d: compute shader %s run as a pass over %u images (%ux%u)", d->x, d->y, sh->key, no, ex.width, ex.height);
    return 1;
}
static void op_dispatch(const p_dispatch *d) {
    g_fb_ep++;   // what it writes is not tracked: every feedback copy is made again (fb_copy)
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_COMPUTE"); on = !(e && *e == '0'); if (!on) NLOG("DM_VK_COMPUTE=0: dispatches are not run"); }
    if (!on) { tex_bound(1); return; }
    DS(g_disp_n, 1);
    if (d->ind) { NRED_ONCE("an indirect dispatch: not run (a later slice)"); DS(g_disp_skip, 1); return; }
    const prog_ent *pe = S.bst[5] ? prog_ent_get(S.bst[5]) : NULL; vkn_shader *sh = pe ? pe->sh[5] : NULL;
    if (!sh || !cs_make(sh)) { DS(g_disp_skip, 1); return; }   // a refused module or pipeline: said there
    // DM_VK_CS_SKIP=<key>[,<key>...]: those compute shaders' dispatches are not run (bisecting a dispatch that loses the GPU)
    { const char *e = getenv("DM_VK_CS_SKIP"); if (e && strstr(e, sh->key)) { static int n; if (n++ < 4) NLOG("DM_VK_CS_SKIP: compute shader %s not run", sh->key); DS(g_disp_skip, 1); return; } }
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    g_disp_dim[0] = (uint32_t)d->x; g_disp_dim[1] = (uint32_t)d->y; g_disp_dim[2] = (uint32_t)d->z;
    rp_flush(); pend_all();   // its storage images and buffers: whatever it reads or writes sees the clears
    draw_tex_prep();
    if (csf_run(sh, d)) return;
    cs_storage_prep(sh);
    draw_tex_prep();
    VkDescriptorImageInfo ii[VKN_MAXB]; VkDescriptorBufferInfo bi[VKN_MAXB]; VkWriteDescriptorSet wr[VKN_MAXB]; vkn_img *imgs[VKN_MAXB]; int ni = 0;
    g_in_dispatch = 1; int ok = cs_binds(sh, ii, bi, wr, imgs, &ni, 0); g_in_dispatch = 0;
    if (!ok) { DS(g_disp_skip, 1); return; }
    VkCommandBuffer cb = cb_now();
    for (int k = 0; k < ni; k++) {   // the storage images into the general layout, after everything before wrote them
        vkn_img *e = imgs[k];
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, rec_layout(e), VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img,
            {e->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &b};
        vk.vkCmdPipelineBarrier2(cb, &di); e->rec = VK_IMAGE_LAYOUT_GENERAL;
        { const char *dg = getenv("DM_VK_DIAG"); if (dg && strstr(dg, "csclear")) {   // diagnostics: the storage images magenta before the dispatch (which pixels does it write?)
            VkClearColorValue mg = {{1.0f, 0.0f, 1.0f, 1.0f}}; VkImageSubresourceRange rr = {e->aspect, 0, 1, 0, 1};
            vk.vkCmdClearColorImage(cb, e->img, VK_IMAGE_LAYOUT_GENERAL, &mg, 1, &rr);
            mem_barrier(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT); } }
    }
    // A storage image the dispatch also samples (Paper Mario: The Thousand-Year Door's lighting reads the images it writes) is in the general layout now: its sampled descriptors
    // say so (they were made for the sampled layout before; VUID-vkCmdDispatch-imageLayout-00344, a mismatch the driver may act on)
    for (uint32_t w = 0; w < (uint32_t)sh->nb; w++) {
        if (wr[w].descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && wr[w].descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) continue;
        for (uint32_t a = 0; a < wr[w].descriptorCount; a++) {
            VkDescriptorImageInfo *q = (VkDescriptorImageInfo *)&wr[w].pImageInfo[a];
            for (int k = 0; k < ni && q->imageLayout != VK_IMAGE_LAYOUT_GENERAL; k++) {
                const vkn_img *e = imgs[k]; int hit = 0;
                for (int v = 0; v < e->nviews && !hit; v++) hit = e->views[v].v == q->imageView;
                for (int v = 0; v < e->nsviews && !hit; v++) hit = e->sviews[v].v == q->imageView;
                if (hit) { q->imageLayout = VK_IMAGE_LAYOUT_GENERAL; g_stat_cs_gen++; }
            }
        }
    }
    mem_barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT);
    vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, sh->cpso);
    if (sh->nb && !vkd_write_bp(cb, VK_PIPELINE_BIND_POINT_COMPUTE, sh->cpl, sh->cdsl, (uint32_t)sh->nb, wr)) { DS(g_disp_skip, 1); return; }
    if (sh->idmap) draw_idmap(cb, sh->cpl, VK_SHADER_STAGE_COMPUTE_BIT, 5);
    vk.vkCmdDispatch(cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    mem_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    static int nlog; if (nlog++ < 6) NLOG("dispatch %dx%dx%d: compute shader %s, %d storage images: run", d->x, d->y, d->z, sh->key, ni);
}
// COPY_B2T (CopyBufferToTexture): the region's rows from the buffer (SetCopyRowStride / SetCopyImageStride, else tightly packed) into the texture's level and layers (as nvn_mtl.m copy_b2t).
// It was counted and not done: the destination kept the zero fill (Paper Mario: The Thousand-Year Door fills a 1x1 texture this way).
// A copy of a format the device takes as it is copies from the pool's shadow (B marks the source range at the record: b_rec); a BC format decoded on the CPU (vk_bc.c) is read from
// guest memory now, decoded into a host buffer of this submission and copied from there. 3D textures are a later slice. DM_VK_COPY=0: not done
static void op_copy_b2t(const p_copy_b2t *c) {
    g_fb_ep++;   // what it writes is not tracked: every feedback copy is made again (fb_copy)
    if (!copy_on() || !c->tex || !c->tex->serial) return;
    const tex_t *t = c->tex;
    rp_flush();
    pthread_mutex_lock(&vkn_tmu); vkn_img *e = xfer_img(t, 1); pthread_mutex_unlock(&vkn_tmu);
    pend_resolve(e);
    if (!e) { NRED_ONCE("COPY_B2T: texture ser=%llu format %#x has no host image: the copy is skipped", (unsigned long long)t->serial, t->d.format); return; }
    if (!(e->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || e->aspect != VK_IMAGE_ASPECT_COLOR_BIT) { NRED_ONCE("COPY_B2T: texture ser=%llu (format %d) cannot be copied into on this device: skipped", (unsigned long long)e->serial, e->fmt); return; }
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), dbpp = 0, dec = vkn_bc_decoded((int)e->nvn_fmt, &dbpp);
    if (!bpb || c->r.width <= 0 || c->r.height <= 0) return;
    uint32_t lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = (c->has_view && c->view.set_layers ? c->view.min_layer : 0) + (uint32_t)(c->r.zoffset > 0 ? c->r.zoffset : 0);
    uint32_t nz = (uint32_t)(c->r.depth > 1 ? c->r.depth : 1); int is3d = e->depth > 1;   // a 3D texture: the region's slices at zoffset of its one layer
    if (is3d) { l0 = 0; if (dec) { NRED_ONCE("COPY_B2T: a BC 3D texture decoded on the CPU (a later slice): skipped"); return; } }
    if (lv >= e->levels || (!is3d && l0 + nz > e->layers) || (is3d && (uint32_t)c->r.zoffset + nz > (e->depth >> lv ? e->depth >> lv : 1))) { NRED_ONCE("COPY_B2T: a level, layer or slice beyond a host image: skipped"); return; }
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * (size_t)bpb, rows = (size_t)((c->r.height + bh - 1) / bh);
    size_t img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * rows;
    VkBuffer src; uint64_t off;
    if (dec) {   // decoded into a host buffer of this submission
        size_t dp = (size_t)c->r.width * (size_t)dbpp, n = dp * (size_t)c->r.height * nz; void *m;
        if (g_cur->ntmp == g_cur->tmpcap) { g_cur->tmpcap = g_cur->tmpcap ? g_cur->tmpcap * 2 : 16; g_cur->tmp = realloc(g_cur->tmp, (size_t)g_cur->tmpcap * sizeof *g_cur->tmp); if (!g_cur->tmp) die("vkn backend: copy temporaries allocation failed"); }
        if (!buf_host(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &g_cur->tmp[g_cur->ntmp].b, &g_cur->tmp[g_cur->ntmp].m, &m)) { NRED_ONCE("COPY_B2T: no host buffer of %zu B for a decoded copy: skipped", n); return; }
        src = g_cur->tmp[g_cur->ntmp++].b; off = 0;
        for (uint32_t z = 0; z < nz; z++)
            if (vkn_bc_decode((int)e->nvn_fmt, (const uint8_t *)(uintptr_t)(c->src + z * img), row, (uint32_t)c->r.width, (uint32_t)c->r.height, (uint8_t *)m + z * dp * (size_t)c->r.height, dp)) {
                NRED_ONCE("COPY_B2T: format %#x could not be decoded: skipped", e->nvn_fmt); return; }
        row = dp; bw = bh = 1; img = dp * (size_t)c->r.height;
    } else {
        uint64_t avail;
        if (!b_use(c->src, &src, &off, &avail) || avail < img * (nz - 1) + row * rows) { NRED_ONCE("COPY_B2T: the source %#llx is in no pool with a shadow (or too short): skipped", (unsigned long long)c->src); return; }
    }
    if (row % (size_t)(dec ? dbpp : bpb) || img % row) { NRED_ONCE("COPY_B2T: row stride %zu / image stride %zu not whole texels or rows: skipped", row, img); return; }
    xfer_barrier(e, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT, lv, l0, nz);
    if (is3d) {
        VkBufferImageCopy r = {off, (uint32_t)(row / (size_t)bpb * (size_t)bw), (uint32_t)(img / row * (size_t)bh), {VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, 1},
                               {c->r.xoffset, c->r.yoffset, c->r.zoffset}, {(uint32_t)c->r.width, (uint32_t)c->r.height, nz}};
        vk.vkCmdCopyBufferToImage(cb_now(), src, e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    } else for (uint32_t z = 0; z < nz; z++) {
        VkBufferImageCopy r = {off + z * img, (uint32_t)(row / (size_t)(dec ? dbpp : bpb) * (size_t)bw), (uint32_t)(img / row * (size_t)bh), {VK_IMAGE_ASPECT_COLOR_BIT, lv, l0 + z, 1},
                               {c->r.xoffset, c->r.yoffset, 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, 1}};
        vk.vkCmdCopyBufferToImage(cb_now(), src, e->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    }
    e->up = 1;   // its content is the copy's from here on (not the zero fill)
    { static int nlog; if (nlog++ < 4) NLOG("COPY_B2T ser=%llu %ux%u format %#x level %u layers %u-%u region %dx%d%s: done", (unsigned long long)e->serial, e->w, e->h, e->nvn_fmt, lv, l0, l0 + nz - 1,
                                             c->r.width, c->r.height, dec ? " (decoded on the CPU)" : ""); }
}
// DM_VK_PASSLOG=<present>: every target change, draw (with its shaders and whether it was skipped), dispatch, copy and clear of that frame's replay, one line each: which pass
// writes the window texture, and what of it is not done (Paper Mario: The Thousand-Year Door's window stayed black with 9% of its draws skipped)
static uint64_t g_plf = ~0ull;
static void passlog(const op_hdr *h, const uint8_t *a, uint64_t skipped_before) {
    char b[256]; int n = snprintf(b, sizeof b, "passlog %s", h->op < OP_COUNT_ && g_opname[h->op] ? g_opname[h->op] : "?");
    if (h->op == OP_SET_RT) { const p_set_rt *r = (const p_set_rt *)a;
        for (int i = 0; i < r->n && i < 8 && n < (int)sizeof b; i++) if (r->color[i]) n += snprintf(b + n, sizeof b - n, " c%d=ser%llu %ux%u f%#x", i, (unsigned long long)r->color[i]->serial, r->color[i]->d.width, r->color[i]->d.height, r->color[i]->d.format);
        if (r->depth && n < (int)sizeof b) n += snprintf(b + n, sizeof b - n, " d=ser%llu", (unsigned long long)r->depth->serial); }
    else if (h->op == OP_DRAW || h->op == OP_DRAW_IDX) { uint64_t sk = 0; for (int i = 0; i < DK_N; i++) sk += atomic_load(&g_dr_skip[i]);
        const prog_ent *pv = S.bst[0] ? prog_ent_get(S.bst[0]) : NULL, *pp = S.bst[1] ? prog_ent_get(S.bst[1]) : NULL;
        n += snprintf(b + n, sizeof b - n, " vs %s ps %s%s", pv && pv->sh[0] ? pv->sh[0]->key : "-", pp && pp->sh[1] ? pp->sh[1]->key : "-", sk != skipped_before ? " SKIPPED" : "");
        for (int k = 0; k < 4 && n < (int)sizeof b; k++) { int tid = (int)(T.tex[1][k] & 0xfffff); if (!tid) continue;   // the pixel stage's first combined slots: which serials it reads
            pthread_mutex_lock(&g_rmu); uint64_t ser = tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; pthread_mutex_unlock(&g_rmu);
            n += snprintf(b + n, sizeof b - n, " t%d=ser%llu", k, (unsigned long long)ser);
            if (k == 0 && ser && n < (int)sizeof b) { pthread_mutex_lock(&vkn_tmu); vkn_img *e = vkn_img_get(ser);   // slot 0's host image: NVN format, size, uploaded, layout
                n += e ? snprintf(b + n, sizeof b - n, "(f%#x %ux%u up%d L%d)", e->nvn_fmt, e->w, e->h, (int)e->up, (int)e->layout) : snprintf(b + n, sizeof b - n, "(no host image)");
                pthread_mutex_unlock(&vkn_tmu); } }
        if (n < (int)sizeof b) n += snprintf(b + n, sizeof b - n, " vp %.0f,%.0f %.0fx%.0f sc %s%d,%d %dx%d dummies %llu", S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3], R.has_sc ? "" : "none ",
                                             R.sc.x, R.sc.y, R.sc.w, R.sc.h, (unsigned long long)atomic_load(&g_ddum));
        // DM_VK_PASSLOG_PS=<pixel key>[,…]: every binding of those draws' pixel shader, as for a dispatch (texture slots with ID, serial, format and target; arrays element by element)
        static const char *pps = (const char *)1; if (pps == (const char *)1) pps = getenv("DM_VK_PASSLOG_PS");
        const vkn_shader *sh = pps && pp ? pp->sh[1] : NULL;
        if (sh && strstr(pps, sh->key)) { NLOG("%s", b); n = 0;
            for (int i = 0; i < sh->nb; i++) { const vkn_sbind *x = &sh->b[i]; char l[200];
                int k0 = x->kind == VKN_B_TEXD ? 0 : x->reg, k1 = x->kind == VKN_B_TEXD ? (x->n < 64 ? x->n : 64) : x->reg + 1;
                if (x->kind == VKN_B_TEX || x->kind == VKN_B_TEXD) for (int k = k0; k < k1; k++) { int tid = (int)(T.tex[1][k] & 0xfffff);
                    pthread_mutex_lock(&g_rmu); uint64_t ser = tid && tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; unsigned fm = ser ? g_treg[tid].t.d.format : 0, tg = ser ? g_treg[tid].t.d.target : 0, w = ser ? g_treg[tid].t.d.width : 0, ht = ser ? g_treg[tid].t.d.height : 0; pthread_mutex_unlock(&g_rmu);
                    if (x->kind == VKN_B_TEXD && !tid) continue;
                    snprintf(l, sizeof l, "passlog   ps %s binding %u slot %d vt %u comp %u: handle %#llx texture ID %d ser%llu %ux%u format %#x target %u", x->kind == VKN_B_TEXD ? "texd" : "tex", x->binding, k, x->vt, x->comp,
                             (unsigned long long)T.tex[1][k], tid, (unsigned long long)ser, w, ht, fm, tg); NLOG("%s", l); }
                else if (x->kind == VKN_B_UBO || x->kind == VKN_B_UBOP) { snprintf(l, sizeof l, "passlog   ps ubo binding %u slot %u: %#llx + %llu", x->binding, x->reg, (unsigned long long)S.ubo[1][x->reg].addr, (unsigned long long)S.ubo[1][x->reg].size); NLOG("%s", l); }
                else if (x->kind == VKN_B_SBUF) { snprintf(l, sizeof l, "passlog   ps sbuf binding %u slot %u: %#llx + %llu", x->binding, x->reg, (unsigned long long)S.ssbo[1][x->reg].addr, (unsigned long long)S.ssbo[1][x->reg].size); NLOG("%s", l); }
                else if (x->kind == VKN_B_TEXU) for (int k = 0; k < x->n && k < 8; k++) { uint32_t hu = ubo_handle(1, x->reg, x->off + 8u * (uint32_t)k); int tid = hu & 0xfffff ? (int)(hu & 0xfffff) + S.tex_base : 0;
                    pthread_mutex_lock(&g_rmu); uint64_t ser = tid && tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; unsigned fm = ser ? g_treg[tid].t.d.format : 0, w = ser ? g_treg[tid].t.d.width : 0, ht = ser ? g_treg[tid].t.d.height : 0; pthread_mutex_unlock(&g_rmu);
                    snprintf(l, sizeof l, "passlog   ps texu binding %u uniform slot %u off %u: handle %#x texture ID %d ser%llu %ux%u format %#x", x->binding, x->reg, x->off + 8u * (uint32_t)k, hu, tid, (unsigned long long)ser, w, ht, fm); NLOG("%s", l); }
                else { snprintf(l, sizeof l, "passlog   ps binding %u kind %u reg %u", x->binding, x->kind, x->reg); NLOG("%s", l); } }
            return; } }
    else if (h->op == OP_DISPATCH) { const p_dispatch *d = (const p_dispatch *)a; const prog_ent *pc = S.bst[5] ? prog_ent_get(S.bst[5]) : NULL; const vkn_shader *sh = pc ? pc->sh[5] : NULL;
        n += snprintf(b + n, sizeof b - n, " %dx%dx%d%s cs %s (has %#x bad %#x)", d->x, d->y, d->z, d->ind ? " indirect" : "", sh ? sh->key : "-", pc ? pc->has : 0, pc ? pc->bad : 0);
        NLOG("%s", b); n = 0;
        for (int i = 0; sh && i < sh->nb; i++) { const vkn_sbind *x = &sh->b[i]; char l[160];   // each binding of the dispatch: what it resolves to
            if (x->kind == VKN_B_TEX) { int tid = (int)(T.tex[5][x->reg] & 0xfffff); pthread_mutex_lock(&g_rmu); uint64_t ser = tid && tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; unsigned fm = ser ? g_treg[tid].t.d.format : 0, tg = ser ? g_treg[tid].t.d.target : 0; pthread_mutex_unlock(&g_rmu);
                snprintf(l, sizeof l, "passlog   cs tex slot %u vt %u: texture ID %d ser%llu format %#x target %u", x->reg, x->vt, tid, (unsigned long long)ser, fm, tg); }
            else if (x->kind == VKN_B_SBUF) snprintf(l, sizeof l, "passlog   cs sbuf slot %u: %#llx + %llu", x->reg, (unsigned long long)S.ssbo[5][x->reg].addr, (unsigned long long)S.ssbo[5][x->reg].size);
            else if (x->kind == VKN_B_UBO) snprintf(l, sizeof l, "passlog   cs ubo slot %u: %#llx + %llu", x->reg, (unsigned long long)S.ubo[5][x->reg].addr, (unsigned long long)S.ubo[5][x->reg].size);
            else if (x->kind == VKN_B_IMG) snprintf(l, sizeof l, "passlog   cs image slot %u: handle %#llx", x->reg, (unsigned long long)S.img[5][x->reg]);
            else continue;
            NLOG("%s", l); }
        return; }
    else if (h->op == OP_BIND_IMG) { const p_bind_handles *bh = (const p_bind_handles *)a; n += snprintf(b + n, sizeof b - n, " stage %d first %d n %d", bh->stage, bh->first, bh->n);
        for (int k = 0; k < bh->n && k < 4 && n < (int)sizeof b; k++) { int tid = (int)(bh->h[k] & 0xfffff);
            pthread_mutex_lock(&g_rmu); uint64_t ser = tid && tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; pthread_mutex_unlock(&g_rmu);
            n += snprintf(b + n, sizeof b - n, " ser%llu", (unsigned long long)ser); } }
    else if (h->op == OP_COPY_T2T) { const p_copy_t2t *c = (const p_copy_t2t *)a; snprintf(b + n, sizeof b - n, " ser%llu -> ser%llu %dx%d", c->src ? (unsigned long long)c->src->serial : 0ull, c->dst ? (unsigned long long)c->dst->serial : 0ull, c->dr.width, c->dr.height); }
    else if (h->op == OP_CLEAR_DS) { const p_clear_ds *c = (const p_clear_ds *)a; snprintf(b + n, sizeof b - n, " depth %g (mask %d) stencil %d (mask %#x)", c->depth, c->depth_mask, c->stencil, (unsigned)c->stencil_mask); }
    else if (h->op == OP_CLEAR_TEX) { const p_clear_tex *c = (const p_clear_tex *)a; snprintf(b + n, sizeof b - n, " ser%llu", c->tex ? (unsigned long long)c->tex->serial : 0ull); }
    else if (h->op != OP_BIND_IMG && h->op != OP_CLEAR_COLOR && h->op != OP_CLEAR_DS) return;
    NLOG("%s", b);
}
// A draw's targets that it may write: their feedback copies are no longer current (fb_copy). Colour targets always; the depth target when depth or stencil is written
static void draw_fb_mark(const pso_key *k) {
    for (int i = 0; i < R.n; i++) if (R.ce[i]) R.ce[i]->fb_ok = 0;
    int sw = 0; for (int f = 0; f < 2; f++) sw |= k->st && (S.st_wmask[f] & 0xff) && (k->sfail[f] || k->szfail[f] || k->szpass[f]);
    if (R.de) R.de->fb_ok &= (uint8_t)~((k->dw ? 1 : 0) | (sw ? 2 : 0));   // bit 0 the depth part, bit 1 the stencil part
}
static int op_draw(const op_hdr *h, const uint8_t *a) {
    int indexed = h->op == OP_DRAW_IDX, prim, itype = 0; int32_t count, first = 0, bvtx = 0, binst, inst; uint64_t ibuf = 0;
    if (indexed) { p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h->size, &t); prim = d->prim; itype = d->itype; count = d->count; bvtx = d->base_vertex; binst = d->base_instance; inst = d->instances; ibuf = d->ibuf;
                   if (d->ind || d->mdi_cnt) { NRED_ONCE("indirect and multi-draw indexed draws are a later slice: skipped"); return draw_skip(DK_KIND); } }
    else { p_draw t; const p_draw *d = draw_rec(a, h->size, &t); prim = d->prim; count = d->count; first = d->first; binst = d->base_instance; inst = d->instances;
           if (d->ind || d->mdi_cnt) { NRED_ONCE("indirect and multi-draw draws are a later slice: skipped"); return draw_skip(DK_KIND); } }
    if (!g_draw_ok) { NRED_ONCE("a draw: the device or the draws' dummies are missing (said at start): skipped"); return draw_skip(DK_KIND); }
    if (binst) { NRED_ONCE("a draw with base instance %d (the shaders' instance index would include it): skipped", binst); return draw_skip(DK_KIND); }
    if (indexed && itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && itype != NVN_INDEX_TYPE_UNSIGNED_INT) { NRED_ONCE("indexed draws with index type %d (8-bit indices are a later slice): skipped", itype); return draw_skip(DK_KIND); }
    if (count <= 0) return draw_skip(DK_EMPTY);
    // The program: its vertex and pixel shaders (no pixel program bound: none, depth only)
    const prog_ent *pv = prog_ent_get(S.bst[0]), *pp = S.bst[1] ? prog_ent_get(S.bst[1]) : NULL;
    for (int st = 2; st < 5; st++) if (S.bst[st] && (prog_ent_get(S.bst[st])->has >> st & 1)) { NRED_ONCE("a draw with a geometry or tessellation shader bound (a later slice): skipped"); return draw_skip(DK_SHADER); }
    if ((pv->bad & 0x81) || (pp && (pp->bad & 0x82))) return draw_skip(DK_SHADER);   // said when the program was resolved
    vkn_shader *vs = pv->sh[0], *ps = pp ? pp->sh[1] : NULL;
    if (!vs) { NRED_ONCE("a draw whose program has no vertex shader: skipped"); return draw_skip(DK_SHADER); }
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK && prim >= NVN_DRAW_PRIMITIVE_TRIANGLES) return draw_skip(DK_EMPTY);   // nothing rasterized (nvn_mtl.m skips it before the pass)
    if (S.rast_discard) { NRED_ONCE("a draw with rasterizer discard on (transform feedback is a later slice): skipped"); return draw_skip(DK_STATE); }
    uint32_t sw; memcpy(&sw, S.vp_sw[0], 4);
    if (sw && sw != 0x100) NRED_ONCE("viewport swizzle %02x %02x %02x %02x (XOR the identity; only NEGATIVE_Y is followed): drawn unswizzled", S.vp_sw[0][0], S.vp_sw[0][1], S.vp_sw[0][2], S.vp_sw[0][3]);
    const lay_ent *L = lay_get(vs, ps);
    if (L == &g_lay_bad) return draw_skip(DK_SHADER);
    // Resources before the pass opens (a texture's layout change ends it)
    draw_tex_prep();
    idmap_extra(1, ps && ps->idmap ? ps : NULL);
    VkDescriptorImageInfo ii[2 * VKN_MAXB]; VkDescriptorBufferInfo bi[2 * VKN_MAXB]; VkWriteDescriptorSet wr[2 * VKN_MAXB]; uint32_t nw = 0; g_dyn_n = 0;
    for (int s = 0; s < 2; s++) { const vkn_shader *sh = s ? ps : vs; if (!sh) continue;
        for (int i = 0; i < sh->nb; i++) { const vkn_sbind *b = &sh->b[i];
            wr[nw] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, b->binding, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NULL, NULL, NULL};
            if (b->kind == VKN_B_TEX) {
                VkImageView v = draw_view(s, sh, b); VkSampler sm = draw_samp(s, b);
                if (!v) { v = g_dview[b->comp][b->vt]; DS(g_ddum, 1); }
                ii[nw] = (VkDescriptorImageInfo){sm ? sm : g_dsamp, v, SR_LAYOUT};
                wr[nw].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr[nw].pImageInfo = &ii[nw];
            } else if (b->kind == VKN_B_TEXD || b->kind == VKN_B_SAMPD || b->kind == VKN_B_TEXU) {
                VkDescriptorImageInfo *ai = dyn_take(b->n); if (!ai) { NRED_ONCE("more than %d dynamic handle array elements in a draw: skipped", 8 * 64); return draw_skip(DK_RES); }
                if (b->kind == VKN_B_TEXU) draw_texu(s, b, ai); else draw_dyn(s, b, ai); wr[nw].descriptorType = bind_type(b); wr[nw].descriptorCount = b->n; wr[nw].pImageInfo = ai;
            } else if (b->kind == VKN_B_SBUF || b->kind == VKN_B_UBOP) { if (!draw_sbuf(s, sh, b, &bi[nw])) return draw_skip(DK_RES); wr[nw].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[nw].pBufferInfo = &bi[nw]; }
            else { if (!draw_ubo(s, sh, b, &bi[nw])) return draw_skip(DK_RES); wr[nw].pBufferInfo = &bi[nw]; }
            nw++;
        } }
    // Vertex and index buffers (their pages were marked as the binds and the draw were replayed: b_rec)
    pso_key k; uint32_t zb;
    if (!rp_begin(0)) return draw_skip(DK_STATE);   // no target (said there)
    if (!draw_key(&k, vs, ps, prim, indexed, itype, &zb)) return draw_skip(DK_STATE);
    draw_fb_mark(&k);
    // Their extent: each stream's fetches end at the last element's start plus the furthest attribute's end. The last element is known for a stream read per instance (instances - 1) and, in a
    // draw without indices, for one read per vertex (first + count - 1): those must end within the pool's own bytes from the bound address (the shadow ends there; Vulkan has no bound past it).
    // A stream read per vertex by an indexed draw is not checked: its bound is the largest index (base vertex added), which only a scan of the guest's indices at each draw would give, and the
    // indices the GPU reads are those the commit uploads after the replay, not necessarily the ones such a scan saw
    uint32_t ni = inst > 1 ? (uint32_t)inst : 1, aend[16] = {0};
    for (int i = 0; i < vs->nin; i++) { uint32_t s = k.in[i].bind, l = k.in[i].loc; if (s >= 16 || s == zb) continue;
        uint32_t e = k.in[i].off + vtx_bytes(S.attr[l].fmt); if (e > aend[s]) aend[s] = e; }
    VkBuffer vb[16]; VkDeviceSize vo[16]; uint32_t nvb = 0, vb0 = 0;
    for (int i = 0; i < k.nbd; i++) {   // the bindings in index order (draw_key lists the fed streams ascending, then the zero one)
        uint32_t s = k.bd[i].idx; VkBuffer buf; uint64_t off = 0, avail;
        if (s == zb) buf = g_zbuf;
        else if (!b_use(S.vbuf[s].address, &buf, &off, &avail)) { NRED_ONCE("vertex buffer %#llx (stream %u) is in no pool with a shadow: skipped", (unsigned long long)S.vbuf[s].address, s); return draw_skip(DK_RES); }
        else if (k.bd[i].rate || !indexed) {
            uint64_t last = k.bd[i].rate ? ni - 1 : (uint64_t)(uint32_t)first + (uint64_t)count - 1, need = last * k.bd[i].stride + aend[s];
            if (need > avail) { NRED_ONCE("vertex buffer %#llx (stream %u): the draw reads %llu B from it, its pool has %llu B there: skipped", (unsigned long long)S.vbuf[s].address, s, (unsigned long long)need,
                                          (unsigned long long)avail); return draw_skip(DK_RES); }
        }
        if (nvb && s != vb0 + nvb) { vk.vkCmdBindVertexBuffers(cb_now(), vb0, nvb, vb, vo); nvb = 0; }
        if (!nvb) vb0 = s;
        vb[nvb] = buf; vo[nvb++] = off;
    }
    VkBuffer ib = VK_NULL_HANDLE; uint64_t io = 0;
    if (indexed) { uint64_t avail, isz = itype == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : 2;
        if (!b_use(ibuf, &ib, &io, &avail)) { NRED_ONCE("index buffer %#llx is in no pool with a shadow: skipped", (unsigned long long)ibuf); return draw_skip(DK_RES); }
        if (io % isz) { NRED_ONCE("index buffer %#llx not aligned to its index size: skipped", (unsigned long long)ibuf); return draw_skip(DK_RES); }
        // The indices end within the pool's own bytes (NVN gives the indices' address, no first index: the draw's are [ibuf, ibuf + count * size)); past them the shadow has none (VUID-vkCmdDrawIndexed-robustBufferAccess2-08798)
        if ((uint64_t)count * isz > avail) { NRED_ONCE("index buffer %#llx: %d indices of %llu B run past its pool's end (%llu B there): skipped", (unsigned long long)ibuf, count, (unsigned long long)isz,
                                                       (unsigned long long)avail); return draw_skip(DK_RES); } }
    // The viewport (rows as NVN gives them, flipped for upper left; NEGATIVE_Y toggles it) and scissor (cut to the pass); an empty one draws nothing
    float vx = 0, vy = 0, vw = (float)R.area.width, vh = (float)R.area.height;
    if (S.has_vp) { vx = S.vp[0][0]; vy = S.vp[0][1]; vw = S.vp[0][2]; vh = S.vp[0][3]; }
    if (!(vw > 0) || !(vh > 0)) return draw_skip(DK_EMPTY);
    const float *vb_ = vd.props.limits.viewportBoundsRange; const uint32_t *vm = vd.props.limits.maxViewportDimensions;
    if (vw > (float)vm[0] || vh > (float)vm[1] || vx < vb_[0] || vy < vb_[0] || vx + vw > vb_[1] || vy + vh > vb_[1]) { NRED_ONCE("viewport %g %g %g %g outside the device's bounds: skipped", vx, vy, vw, vh); return draw_skip(DK_STATE); }
    int flip = !vkn_origin_ll() ^ (sw == 0x100);
    VkViewport vp = {vx, flip ? vy + vh : vy, vw, flip ? -vh : vh, clamp01(S.dr[0]), clamp01(S.dr[1])};
    int64_t x0 = 0, y0 = 0, x1 = R.area.width, y1 = R.area.height;
    if (R.has_sc) { if (R.sc.x > x0) x0 = R.sc.x; if (R.sc.y > y0) y0 = R.sc.y; if ((int64_t)R.sc.x + R.sc.w < x1) x1 = (int64_t)R.sc.x + R.sc.w; if ((int64_t)R.sc.y + R.sc.h < y1) y1 = (int64_t)R.sc.y + R.sc.h; }
    if (x1 <= x0 || y1 <= y0) return draw_skip(DK_EMPTY);
    VkRect2D sc = {{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    // The pipeline
    static pso_key lk; static VkPipeline lp;
    VkPipeline p;
    if (lp && !memcmp(&k, &lk, sizeof k)) { p = lp; DS(g_pso_hit, 1); }
    else {
        void *v = nvn_hm_get(&g_psos, &k);
        if (v == PSO_BAD) return draw_skip(DK_PSO);
        if (v) { p = (VkPipeline)v; DS(g_pso_hit, 1); }
        else { p = pso_make(&k, vs, ps, L->pl); nvn_hm_put(&g_psos, &k, p ? (void *)p : PSO_BAD); if (!p) return draw_skip(DK_PSO); }
        lk = k; lp = p;
    }
    VkCommandBuffer cb = cb_now();
    if (!D.valid || D.pso != p) { vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); D.pso = p; }
    if (nw && vkd_write(cb, L->pl, L->dsl, nw, wr)) { DS(g_push, 1); DS(g_dwr, nw); }   // pushed, or a pool's set bound (vk_desc.c)
    if (ps && ps->idmap) draw_idmap(cb, L->pl, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
    if (!D.valid || memcmp(&D.vp, &vp, sizeof vp)) { vk.vkCmdSetViewport(cb, 0, 1, &vp); D.vp = vp; }
    if (!D.valid || memcmp(&D.sc, &sc, sizeof sc)) { vk.vkCmdSetScissor(cb, 0, 1, &sc); D.sc = sc; }
    if (!D.valid || memcmp(D.bc, S.blend_color, sizeof D.bc)) { vk.vkCmdSetBlendConstants(cb, S.blend_color); memcpy(D.bc, S.blend_color, sizeof D.bc); }
    for (int f = 0; f < 2; f++) {
        VkStencilFaceFlags fl = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
        if (!D.valid || D.cm[f] != (S.st_vmask[f] & 0xff)) { vk.vkCmdSetStencilCompareMask(cb, fl, D.cm[f] = S.st_vmask[f] & 0xff); }
        if (!D.valid || D.wm[f] != (S.st_wmask[f] & 0xff)) { vk.vkCmdSetStencilWriteMask(cb, fl, D.wm[f] = S.st_wmask[f] & 0xff); }
        if (!D.valid || D.ref[f] != (S.st_ref[f] & 0xff)) { vk.vkCmdSetStencilReference(cb, fl, D.ref[f] = S.st_ref[f] & 0xff); }
    }
    float bias[3] = {S.po.units, S.po.clamp, S.po.factor};
    if (bias[1] != 0 && !vd.bias_clamp) { NRED_ONCE("polygon offset clamp %g, and the device has no depthBiasClamp: unclamped", bias[1]); bias[1] = 0; }
    if (!D.valid || memcmp(D.bias, bias, sizeof bias)) { vk.vkCmdSetDepthBias(cb, bias[0], bias[1], bias[2]); memcpy(D.bias, bias, sizeof bias); }
    D.valid = 1;
    if (nvb) vk.vkCmdBindVertexBuffers(cb, vb0, nvb, vb, vo);
    if (indexed) { vk.vkCmdBindIndexBuffer(cb, ib, io, itype == NVN_INDEX_TYPE_UNSIGNED_INT ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16); vk.vkCmdDrawIndexed(cb, (uint32_t)count, ni, 0, bvtx, 0); DS(g_dr_idx, 1); }
    else vk.vkCmdDraw(cb, (uint32_t)count, ni, (uint32_t)first, 0);
    DS(g_dr_n, 1);
    return 1;
}
static void draw_report(const char *why) {   // the draws' line, beside the GPU work one (cumulative)
    uint64_t sl, sr; vkn_shader_stats(&sl, &sr);
    char sk[400]; size_t o = 0; sk[0] = 0;
    for (int i = 0; i < DK_N; i++) { uint64_t v = atomic_load(&g_dr_skip[i]); if (v && o < sizeof sk) o += (size_t)snprintf(sk + o, sizeof sk - o, "%s%s %llu", o ? ", " : "", g_dk_name[i], (unsigned long long)v); }
    { char dsc[256]; vkd_report(dsc, sizeof dsc); NLOG("descriptors up to frame %llu: %s; frame parts sent early %llu", (unsigned long long)atomic_load(&g_present_n), dsc, (unsigned long long)g_split_n); }
    uint64_t n = atomic_load(&g_present_n), pm = atomic_load(&g_pso_made);
    NLOG("draws up to frame %llu (%s): recorded %llu (indexed %llu), skipped: %s; shaders loaded %llu, refused %llu; layouts %llu; pipelines made %llu (%.1f ms in all, the longest %.2f ms), failed %llu, cache hits %llu; "
         "descriptor pushes %llu, writes %llu, dummies %llu; uniform bindings %llu, at an unaligned address %llu (copied), cut short %llu; record time %.2f ms per frame (%.1f ms in all)",
         (unsigned long long)n, why, (unsigned long long)atomic_load(&g_dr_n), (unsigned long long)atomic_load(&g_dr_idx), o ? sk : "none", (unsigned long long)sl, (unsigned long long)sr,
         (unsigned long long)atomic_load(&g_lay_made), (unsigned long long)pm, (double)atomic_load(&g_pso_ns) / 1e6, (double)atomic_load(&g_pso_max) / 1e6, (unsigned long long)atomic_load(&g_pso_fail),
         (unsigned long long)atomic_load(&g_pso_hit), (unsigned long long)atomic_load(&g_push), (unsigned long long)atomic_load(&g_dwr), (unsigned long long)atomic_load(&g_ddum),
         (unsigned long long)atomic_load(&g_ubo_n), (unsigned long long)atomic_load(&g_ubo_unal), (unsigned long long)atomic_load(&g_ubo_short),
         n ? (double)atomic_load(&g_rec_ns) / 1e6 / (double)n : 0.0, (double)atomic_load(&g_rec_ns) / 1e6);
}

void mtl_sub_begin(void) {
    mtl_init();
    if (!g_vk) { g_cur = &g_sub[0]; g_cur->npost = 0; return; }
    uint64_t i = g_nbegun++;
    pthread_mutex_lock(&g_mu); while (g_ncomp + NSUB <= i) pthread_cond_wait(&g_cv, &g_mu); pthread_mutex_unlock(&g_mu);   // the slot's last submission has completed
    g_cur = &g_sub[i % NSUB]; g_cur->npost = 0; g_cur->failed = atomic_load(&vkn_lost) || atomic_load(&g_leak);   // failed from the start: nothing is recorded (its pool may hold a buffer the lost GPU never finished)
    g_sub_id++; g_ntouch = 0; g_ntj = 0; T.all = 0x3f;   // every bound texture is looked at again in a new submission (its upload check, its layout here)
    D.valid = 0; g_cur->ring_used = 0;   // a new command buffer: no pipeline, no dynamic state; the slot's ring is free again
    for (int i = 0; i < g_cur->ntmp; i++) { vk.vkDestroyBuffer(vd.dev, g_cur->tmp[i].b, NULL); vk.vkFreeMemory(vd.dev, g_cur->tmp[i].m, NULL); } g_cur->ntmp = 0;   // and its copies' temporaries
    pthread_mutex_lock(&vkn_tmu); vkn_sweep(); pthread_mutex_unlock(&vkn_tmu);
    if (g_cur->failed) return;
    vk.vkResetCommandPool(vd.dev, g_cur->pool, 0);
    vkd_slot_begin((int)(i % NSUB));   // the slot's descriptor pools are free again (vk_desc.c)
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    vk.vkBeginCommandBuffer(g_cur->cb, &bi);
    rt_reresolve();
}
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
// In submission order on the CPU, at once: later commands of the same submission see the bytes (on the GPU they run in order too).
// The source must be pool memory as well: a sparse virtual pool's range stays PROT_NONE on Linux (hle_nvn.c vpool_map keeps its pages in the page table only), and reading it would kill the process
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    void *b; uint64_t o;
    if (!n) return;
    if (!mtl_addr(src, n, &b, &o)) { NLOG("[RED] buffer copy source outside every memory pool src=%#llx n=%#llx: skipped", (unsigned long long)src, (unsigned long long)n); return; }
    if (cpu_target(dst, n, "buffer copy")) memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)n);
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    if (!n || !cpu_target(dst, n, "buffer fill")) return;
    uint8_t *d = (uint8_t *)(uintptr_t)dst; uint64_t i = 0;
    for (; i + 4 <= n; i += 4) memcpy(d + i, &value, 4);
    if (i < n) memcpy(d + i, &value, (size_t)(n - i));   // a size that is not a multiple of 4 takes the pattern's leading bytes
}
static void sub_update(const void *data, uint64_t dst, uint64_t n) { if (n && cpu_target(dst, n, "uniform update")) memcpy((void *)(uintptr_t)dst, data, (size_t)n); }
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type}); }

// The frame's work goes to the GPU in parts (DM_VK_SPLIT=<draws>, 64 by default, 0: one submission per QueueSubmitCommands): at a SET_RT, once that many draws were recorded
// since the last part, what is recorded is committed (its pass ends there anyway) and a new submission begins, so the GPU starts while the rest of the frame is recorded. Fences
// recorded later take the later part's value (OP_FENCE_SYNC reads g_fence then), the game's commit returns the last part's. As nvn_mtl.m's segmented submission (DM_NVN_CUT).
// Paper Mario: The Thousand-Year Door on the RG 476H: the GPU waited for 8-9 ms of recording each frame; a part every 64 draws made 8 parts a frame that waited 2.5 ms each for one of the 4 submission slots (storybook 62 ms a frame against 47), every 400 draws gains 2-3 ms (storybook 44, Rogueport 34 against 35)
static int split_at(void) { static int n = -1; if (n < 0) { const char *e = getenv("DM_VK_SPLIT"); n = e && *e ? atoi(e) : 400; if (n != 400) NLOG("DM_VK_SPLIT=%d: %s", n, n ? "a part every that many draws" : "one submission per QueueSubmitCommands"); } return n; }
static void exec_run(const uint8_t *p) {
    int xd = g_xd++;
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (xd < 32) g_xp[xd] = p;
        const uint8_t *a = p + 8;
        surf_rec(&h, p, a);
        if (g_vk && !g_cur->failed) b_rec(&h, p, a);   // B: what this record reads as buffers, and its own command pages
        switch (h.op) {
            case OP_END: g_xd--; return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); exec_run((const uint8_t *)(uintptr_t)t); break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); mtl_sub_counter(ad, (uint32_t)t); break; }
            case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = atomic_load(&g_fence) + 1; s->fenced = 1; break; }   // the fence this submission commits with
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s_, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); sub_update(a + 16, d, n); break; }
            case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
            default: break;
        }
        uint64_t plsk = 0; int pl = 0;
        if (g_plf == ~0ull) { const char *e = getenv("DM_VK_PASSLOG"); g_plf = e && *e ? strtoull(e, NULL, 10) : ~1ull; }
        g_pl_on = g_plf < ~1ull && atomic_load(&g_present_n) == g_plf;
        if (g_plf < ~1ull && atomic_load(&g_present_n) == g_plf) { pl = 1; for (int i = 0; i < DK_N; i++) plsk += atomic_load(&g_dr_skip[i]); }
        if (g_vk && !g_cur->failed) switch (h.op) {   // the GPU part: targets, clears, passes, barriers (none once the GPU is lost: mtl_sub_begin)
            case OP_SET_RT: { const p_set_rt *r = (const p_set_rt *)a; int sub = R.on && pass_keep() ? rt_sub(r) : -1; if (sub >= 0) { g_stat_keep_rt++; R.act = (unsigned)sub; T.all = 0x3f; break; }   // the open pass's targets: it stays open
                              rp_flush();
                              if (split_at() > 0 && g_split_draws >= (uint64_t)split_at()) { g_split_draws = 0; g_split_n++; mtl_sub_commit(); mtl_sub_begin(); if (!g_cur || g_cur->failed) break; }   // a part of the frame to the GPU
                              R.n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n; R.has_rt = 1; R.act = 0xff; rt_resolve(r); T.all = 0x3f; break; }   // all: what was drawn is sampled after a barrier
            case OP_CLEAR_COLOR: op_clear_color((const p_clear_color *)a); break;
            case OP_CLEAR_DS: op_clear_ds((const p_clear_ds *)a); break;
            case OP_DRAW: case OP_DRAW_IDX:
                g_split_draws++;
                if (S.bst[0]) { op_draw(&h, a); break; }
                tex_bound(0); rp_begin(0);   // no program (slice 2b's: its textures made ready, the pass opens, its clears land)
                NRED_ONCE("a draw with no program bound: skipped"); DS(g_dr_skip[DK_NOPROG], 1); break;
            case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_IMG: case OP_COPY_STRIDE: case OP_STENCIL: case OP_POLY_OFFSET: case OP_VIEWPORTS:
            case OP_DEPTH_RANGE: case OP_DEPTH_CLAMP: case OP_VP_SWIZZLE: case OP_BLEND_COLOR: case OP_PRIM_RESTART: case OP_RAST_DISCARD: state_op(h.op, a); break;
            case OP_DISPATCH: op_dispatch((const p_dispatch *)a); break;
            case OP_COPY_T2T: op_copy_t2t((const p_copy_t2t *)a); break;
            case OP_CLEAR_TEX: op_clear_tex((const p_clear_tex *)a); break;
            case OP_COPY_B2T: op_copy_b2t((const p_copy_b2t *)a); break;
            case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP: tex_bind(h.op, a); break;
            case OP_BARRIER: { int32_t bits; memcpy(&bits, a, 4); op_barrier(bits); break; }
            case OP_SCISSOR: { memcpy(&R.sc, a, sizeof R.sc); R.has_sc = 1; break; }
            default: break;
        }
        if (pl) passlog(&h, a, plsk);
        p += h.size;
    }
}
void mtl_exec(const uint8_t *p) {   // the record time (outermost: the draws' line)
    uint64_t t0 = now_ns(); exec_run(p); DS(g_rec_ns, now_ns() - t0);
}
uint64_t mtl_sub_commit(void) {
    sub_slot *s = g_cur;
    if (!g_vk || s->failed) g_cur = NULL;   // nothing more is recorded (the GPU path below ends the pass first: rp_flush records through g_cur)
    if (!g_vk) {   // the null backend's: done at once
        uint64_t fence = atomic_load(&g_fence) + 1; post_run(s);
        pthread_mutex_lock(&g_mu); atomic_store(&g_fence, fence); atomic_store(&g_done, fence); pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);
        return fence;
    }
    if (s->failed) {   // begun after the GPU was lost: nothing recorded, it completes behind the earlier ones (the completion thread, in order)
        pthread_mutex_lock(&vkn_qmu); uint64_t v = atomic_load(&g_fence) + 1; atomic_store(&g_fence, v); pthread_mutex_unlock(&vkn_qmu);
        pthread_mutex_lock(&g_mu); s->value = v; g_ncommit++; pthread_cond_signal(&g_ccv); pthread_mutex_unlock(&g_mu);
        return v;
    }
    rp_flush(); pend_all();   // an open pass, recorded and deferred clears land in this submission's command buffer (cb_now: g_cur, still s)
    vk.vkEndCommandBuffer(s->cb);
    g_cur = NULL;
    uint64_t vb = atomic_load(&g_fence) + 1;   // this submission's value (only this thread moves g_fence): the shadows' last use
    int up = b_commit(s, vb);   // B: the pages it reads, into the shadows, by s->up ahead of s->cb
    // Under the queue lock: the layouts this submission leaves become the published ones as it is queued, so the present thread (which samples the window texture and restores its layout
    // under the same lock) always records against the layout the queue will have
    pthread_mutex_lock(&vkn_qmu);
    uint64_t v = atomic_load(&g_fence) + 1;
    for (int i = 0; i < g_ntouch; i++) g_touch[i]->layout = g_touch[i]->rec;
    VkSemaphoreSubmitInfo sig = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, g_tl, v, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
    VkCommandBufferSubmitInfo c[2] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, s->up, 0}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, s->cb, 0}};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, up ? 2 : 1, up ? c : c + 1, 1, &sig};   // the uploads first: their barriers order them after every earlier submission and before this one's reads
    if (v != vb) NRED_ONCE("submission %llu: its value moved to %llu while it was committed (its shadows' last use may be wrong)", (unsigned long long)vb, (unsigned long long)v);
    VkResult r = vk.vkQueueSubmit2(vd.q, 1, &si, VK_NULL_HANDLE);
    atomic_store(&g_fence, v);
    pthread_mutex_unlock(&vkn_qmu);
    if (r) { vkn_gpu_lost("vkQueueSubmit2", r); s->failed = 1; }
    g_ntouch = 0;
    pthread_mutex_lock(&g_mu); s->value = v; g_ncommit++; pthread_cond_signal(&g_ccv); pthread_mutex_unlock(&g_mu);
    return v;
}
uint64_t mtl_last_fence(void) { return atomic_load(&g_fence); }
// A fence beyond the last commit (a sync object fenced by a submission still being replayed on another thread) is waited for, as with Metal
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    if (atomic_load(&g_done) >= fence) return 1;
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600000000000ull ? 3600000000000ull : timeout_ns);
    dl.tv_sec += (time_t)(ns / 1000000000ull); dl.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_mu);
    while (atomic_load(&g_done) < fence) if (pthread_cond_timedwait(&g_cv, &g_mu, &dl)) break;
    int ok = atomic_load(&g_done) >= fence;
    pthread_mutex_unlock(&g_mu);
    return ok;
}
void mtl_finish(void) { uint64_t f = atomic_load(&g_fence); while (!mtl_wait(f, 3600000000000ull)) {} }
static void surf_report(const char *why);
static void vkn_report(const char *why) {   // the backend's own line beside the surface one: what reached the GPU
    pthread_mutex_lock(&vkn_tmu); uint64_t ni = g_nimg, ib = g_img_b, gp = g_ghost_puts; pthread_mutex_unlock(&vkn_tmu);
    NLOG("GPU work up to frame %llu (%s): submissions %llu, passes %llu, clears folded into load ops %llu (deferred %llu: taken by a pass %llu, recorded between passes %llu / after ending one %llu), clears in a pass %llu, oversize clears in a pass of their own %llu, depth / stencil not stored %llu / %llu, smaller render areas %llu (%llu pixels left out), barriers %llu, kept open across %llu SET_RT / %llu barriers, barriers for a further sampling stage %llu (%llu ended a pass); host images %llu (%llu B, ghosts included), ghosts made %llu; validation %s %llu",
         (unsigned long long)atomic_load(&g_present_n), why, (unsigned long long)atomic_load(&g_fence), (unsigned long long)g_stat_pass, (unsigned long long)g_stat_fold, (unsigned long long)g_stat_pend[0], (unsigned long long)g_stat_pend[1], (unsigned long long)g_stat_pend[2], (unsigned long long)g_stat_pend[3], (unsigned long long)g_stat_clr_mid,
         (unsigned long long)g_stat_own, (unsigned long long)g_stat_dsn[0], (unsigned long long)g_stat_dsn[1], (unsigned long long)g_stat_area[0], (unsigned long long)g_stat_area[1], (unsigned long long)g_stat_bar, (unsigned long long)g_stat_keep_rt, (unsigned long long)g_stat_keep_bar, (unsigned long long)g_stat_sr_more, (unsigned long long)g_stat_sr_split, (unsigned long long)ni, (unsigned long long)ib, (unsigned long long)gp, vd.validation ? "on, messages" : "off", (unsigned long long)vkn_validation_count());
}
#define STATS_FROM 60   // no statistics line before this present: the parity cut ends at the host's own line of present 1 or 2 (parity.py PRESENT), and lines before it must be the HLE's alone
// The suspend path's bound (mtl_drain): Android calls it on the UI thread (front_android.c on_stop → dm_gfx_suspend), where a GPU that hangs must not hang the app into an ANR, the same 4 s as
// the window loss (vk_present.c dm_gfx_window_lost). Elsewhere it waits as long as it takes (the graceful exit's lines come after it)
#ifdef __ANDROID__
#define DRAIN_NS 4000000000ull
#else
#define DRAIN_NS 3600000000000ull
#endif
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
void mtl_drain(void) {   // the last lines of a run (the graceful exit ends in _exit, no atexit), or the app going to the background: every present handed over, the GPU idle
    if (g_vk) {   // what has not finished when the bound runs out finishes behind it: nothing here depends on it (resume reopens the gate only; destruction still waits for the GPU's answer)
        uint64_t t0 = mono_ns(); int ok = vkn_present_drain(DRAIN_NS); uint64_t used = mono_ns() - t0;
        ok = ok && used < DRAIN_NS && mtl_wait(atomic_load(&g_fence), DRAIN_NS - used);
        if (!ok) NLOG("[RED] graphics suspend: the GPU work in flight has not finished in %llu s: suspended regardless (it completes behind it)", (unsigned long long)(DRAIN_NS / 1000000000ull));
    }
    if (g_vk) pc_save();
    if (atomic_load(&g_present_n) >= STATS_FROM) { if (g_vk) { vkn_report("graphics suspended: exit, or the app in the background"); draw_report("graphics suspended: exit, or the app in the background"); b_report("graphics suspended: exit, or the app in the background"); t_report("graphics suspended: exit, or the app in the background"); }
        surf_report("graphics suspended: exit, or the app in the background"); }
}

// ---------- The surface lines (see the statistics at the top): one line of counts and the pools made or freed since the last one ----------
static void fmt_list(char *b, size_t cap, size_t *o, const _Atomic uint64_t *bits, int nbits) {
    int any = 0;
    for (int f = 0; f < nbits && *o < cap; f++) if (atomic_load_explicit(&bits[f >> 6], memory_order_relaxed) >> (f & 63) & 1) { *o += (size_t)snprintf(b + *o, cap - *o, "%s%#x", any ? "," : " ", f); any = 1; }
    if (!any && *o < cap) *o += (size_t)snprintf(b + *o, cap - *o, " none");
}
static void surf_report(const char *why) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; pthread_mutex_lock(&mu);   // the present thread and mtl_drain
    static const char *const prim[16] = {"points", "lines", "line_loop", "line_strip", "triangles", "triangle_strip", "triangle_fan", "quads", "quad_strip", "polygon", "lines_adj", "line_strip_adj", "triangles_adj", "triangle_strip_adj", "patches", "?"};
    static const char *const bar[8] = {"order_primitives", "order_fragments", "order_fragments_tiled", "order_indirect", "invalidate_texture", "invalidate_shader", "invalidate_texture_descriptor", "invalidate_zcull"};
    unsigned long long v[S_N], n = atomic_load(&g_present_n); for (int i = 0; i < S_N; i++) v[i] = atomic_load_explicit(&g_st[i], memory_order_relaxed);
    static char b[8192]; size_t o = 0;
#define AP(...) do { if (o < sizeof b) o += (size_t)snprintf(b + o, sizeof b - o, __VA_ARGS__); } while (0)
    AP("surface up to frame %llu (%s): records", n, why);
    for (int i = 0; i < OP_COUNT_; i++) if (v[i]) AP(" %s=%llu", g_opname[i] ? g_opname[i] : "?", v[i]);
    AP("; draws %llu (instanced %llu, base instance %llu, indirect %llu, indirect-count %llu; %llu vertices), indexed %llu (instanced %llu, base vertex %llu, base instance %llu, indirect %llu, indirect-count %llu; %llu indices); primitives",
       v[OP_DRAW], v[S_DRAW_INST], v[S_DRAW_BINST], v[S_DRAW_IND], v[S_DRAW_MDI], v[S_DRAW_VTX], v[OP_DRAW_IDX], v[S_DIDX_INST], v[S_DIDX_BVTX], v[S_DIDX_BINST], v[S_DIDX_IND], v[S_DIDX_MDI], v[S_DIDX_IDX]);
    for (int i = 0; i < 16; i++) if (v[S_PRIM + i]) AP(" %s=%llu", prim[i], v[S_PRIM + i]);
    AP("; index types u8=%llu u16=%llu u32=%llu", v[S_ITYPE], v[S_ITYPE + 1], v[S_ITYPE + 2]);
    AP("; colour clears %llu before the pass's first draw / %llu after it / %llu with a partial mask; depth-stencil clears %llu / %llu / %llu with a partial stencil mask; texture clears %llu; buffer fills %llu (%llu B)",
       v[S_CLR_START], v[S_CLR_MID], v[S_CLR_PART], v[S_DS_START], v[S_DS_MID], v[S_DS_PART], v[OP_CLEAR_TEX], v[OP_CLEAR_BUF], v[S_FILL_B]);
    AP("; copies buffer to buffer %llu (%llu B), buffer to texture %llu (%llu B), texture to buffer %llu (%llu B), texture to texture %llu (%llu resolves); uniform updates %llu (%llu B); bytes GPU ops wrote into pool memory %llu",
       v[OP_COPY_B2B], v[S_B2B_B], v[OP_COPY_B2T], v[S_B2T_B], v[OP_COPY_T2B], v[S_T2B_B], v[OP_COPY_T2T], v[S_T2T_RES], v[OP_UPDATE_UB], v[S_UB_B], v[S_GW_B]);
    AP("; compute dispatches %llu (%llu indirect); counter reports timestamp %llu / samples passed %llu / other %llu, resets %llu; transform feedback binds %llu, begin %llu end %llu pause %llu resume %llu; render-enable records %llu (equal %llu, not equal %llu, on %llu, off %llu)",
       v[OP_DISPATCH], v[S_DISP_IND], v[S_CNT_TS], v[S_CNT_SP], v[S_CNT_OTH], v[OP_RESET_COUNTER], v[OP_TF_BIND], v[S_TF_WHAT], v[S_TF_WHAT + 1], v[S_TF_WHAT + 2], v[S_TF_WHAT + 3],
       v[OP_RENDER_COND], v[S_RC_MODE], v[S_RC_MODE + 1], v[S_RC_MODE + 2], v[S_RC_MODE + 3]);
    { static const char *const sw[8] = {"+x", "-x", "+y", "-y", "+z", "-z", "+w", "-w"}; uint32_t m = atomic_load(&g_vpsw); AP("; viewport swizzle records not the identity %llu", v[S_VPSW]);
      for (int i = 0; i < 32; i++) if (m >> i & 1) AP(" %c=%s", "xyzw"[i / 8], sw[i % 8]); }
    AP("; state binds blend %llu, colour %llu, channel mask %llu, multisample %llu, polygon %llu, depth-stencil %llu", v[S_STATE], v[S_STATE + 1], v[S_STATE + 2], v[S_STATE + 3], v[S_STATE + 4], v[S_STATE + 5]);
    AP("; barriers %llu, bits", v[OP_BARRIER]); for (int i = 0; i < 8; i++) if (v[S_BAR + i]) AP(" %s=%llu", bar[i], v[S_BAR + i]);
    AP("; render-target sets %llu, colour targets", v[OP_SET_RT]); for (int i = 0; i < 9; i++) if (v[S_RT_NC + i]) AP(" %d:%llu", i, v[S_RT_NC + i]);
    AP(", with depth %llu, depth only %llu, multisampled %llu, formats", v[S_RT_DEPTH], v[S_RT_ZONLY], v[S_RT_MSAA]); fmt_list(b, sizeof b, &o, g_fmt[0], 256);
    AP("; registered texture formats"); fmt_list(b, sizeof b, &o, g_fmt[1], 256);
    { _Atomic uint64_t t[1]; atomic_init(&t[0], atomic_load(&g_tgt)); AP(", targets"); fmt_list(b, sizeof b, &o, t, 32); }
    AP("; vertex attribute formats"); fmt_list(b, sizeof b, &o, g_fmt[2], 256);
    AP("; ReadTexels %llu (%llu B), %llu of them of textures the GPU writes", v[S_RTX], v[S_RTX_B], v[S_RTX_GPU]);
    pthread_mutex_lock(&g_pmu);
    AP("; memory pools made %llu (%llu B), freed %llu (%llu B), live %llu B, peak %llu B", (unsigned long long)g_pmade, (unsigned long long)g_pmade_b, (unsigned long long)g_pfreed, (unsigned long long)g_pfreed_b,
       (unsigned long long)g_plive_b, (unsigned long long)g_ppeak_b);
    NLOG("%s", b);
    // Each pool once when made, once more when freed (64 lines at most per report: the rest wait for the next)
    for (int i = 0, k = 0; i < g_nplog && k < 64; i++) {
        pool_log *l = &g_plog[i]; int end = l->died != ~0ull;
        if (l->said == 2 || (l->said == 1 && !end)) continue;
        if (end) NLOG("memory pool %llu: %#llx size %#llx (%llu B) made at frame %llu, freed at frame %llu", (unsigned long long)l->id, (unsigned long long)l->mem, (unsigned long long)l->size, (unsigned long long)l->size, (unsigned long long)l->born, (unsigned long long)l->died);
        else NLOG("memory pool %llu: %#llx size %#llx (%llu B) made at frame %llu", (unsigned long long)l->id, (unsigned long long)l->mem, (unsigned long long)l->size, (unsigned long long)l->size, (unsigned long long)l->born);
        l->said = end ? 2 : 1; k++;
    }
    pthread_mutex_unlock(&g_pmu);
#undef AP
    pthread_mutex_unlock(&mu);
}

// ---------- Present: counted, then handed to the present thread (vk_present.c); pacing stays with hle_nvn.c's own clock (no display link, as the null backend) ----------
// DM_NVN_SHOT=<present>[,<present>...]:<path> (nvn_mtl.m present_now's switch, up to 4096 presents; %d in the path is the present number): those presents' window textures are written as PNG by the
// present thread (vk_present.c shot_one), in nvn_mtl.m shot()'s form. Presents are numbered as Metal numbers them, 1 = the first (the count, not the present thread's ticket). DM_NVN_SHOT_OUT
// (Metal's upscaled copy) is not ported: said once, the shots stay at the texture's own size
static long g_shots[4096]; static int g_nshot; static const char *g_shot_path;
static void shot_parse(void) {   // mtl_init
    const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL;
    if (!e || !*e) return;
    if (!c || !c[1]) { NLOG("[RED] DM_NVN_SHOT=%s: not <present>[,<present>...]:<path>: no screenshots", e); return; }
    for (const char *q = e; q < c && g_nshot < 4096; ) { g_shots[g_nshot++] = atol(q); const char *k = memchr(q, ',', (size_t)(c - q)); q = k ? k + 1 : c; }
    g_shot_path = c + 1;
    const char *o = getenv("DM_NVN_SHOT_OUT"); if (o && *o) NLOG("[RED] DM_NVN_SHOT_OUT=%s: not on the Vulkan backend: screenshots are of the window texture at its own size", o);
}
static const char *shot_for(uint64_t n, char *path, size_t cap) {   // this present's screenshot path, or NULL: %d is the present number, %% a percent sign, the rest as it is
    int hit = 0; for (int i = 0; i < g_nshot && !hit; i++) hit = (uint64_t)g_shots[i] == n;
    if (!hit) return NULL;
    size_t o = 0;
    for (const char *p = g_shot_path; *p && o + 1 < cap; p++) {
        if (p[0] == '%' && p[1] == 'd') { int k = snprintf(path + o, cap - o, "%llu", (unsigned long long)n); o = k < 0 ? o : o + (size_t)k < cap ? o + (size_t)k : cap - 1; p++; }
        else if (p[0] == '%' && p[1] == '%') { path[o++] = '%'; p++; }
        else path[o++] = *p;
    }
    path[o] = 0;
    return path;
}
extern _Atomic uint64_t dm_present_ns;   // hle_thread.c (DM_HITCH_DUMP): when the last present was made
static uint64_t present_count(void) {
    mtl_init();   // DM_NVN_NULL_EVERY
    atomic_store(&dm_present_ns, now_ns());
    uint64_t n = atomic_fetch_add(&g_present_n, 1) + 1;
    // Not worded "first present": lognorm.py --until-present ends the parity cut at that phrase, and this line comes before the host's own (DM_DETCLOCK, [nvn] frame 1) that end it on the Metal backend
    if (n == 1) { NLOG("%s", g_vk ? "presenting: frames go to the swapchain" : "presenting: frames are counted, nothing is shown"); if (g_nshot && !g_vk) NLOG("[RED] DM_NVN_SHOT: no Vulkan device, no screenshots"); }
    if (g_vk && n % 300 == 0) pc_save();
    if (n >= STATS_FROM && (n == 60 || n == 600 || n % 1800 == 0 || (g_every && n % g_every == 0))) { surf_report("periodic"); if (g_vk) { vkn_report("periodic"); draw_report("periodic"); b_report("periodic"); t_report("periodic"); } }   // the count before this present: labelled with the frame just ended
    return n;
}
// DM_DETCLOCK (hle_os.c) stops the guest clock at the first present (hle_nvn.c dump_frame, after mtl_present) and from then on moves it one step per present. A frame loop that sleeps until a
// whole step has gone by since its last tick (Stardew Valley's fixed time step: SleepThread on the guest clock) then sleeps for ever if less than a step of wall time passed between that tick
// and the switch: Metal's first frame builds its pipelines and takes longer; this backend's took a few ms and the slice 2a Mac run stopped at present 1. One step of wall time at the
// first present keeps the switch at least a step after the tick. Test runs only (DM_DETCLOCK unset: nothing)
static void detclock_first_present(void) {
    const char *e = getenv("DM_DETCLOCK"); uint64_t step = e ? strtoull(e, 0, 0) : 0;
    if (!step || step > 1000000000ull) return;
    uint64_t end = mono_ns() + step;
    for (uint64_t t; (t = mono_ns()) < end; ) { struct timespec ts = {0, (long)(end - t)}; nanosleep(&ts, NULL); }
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {   // ticket 0 = nothing pending: the next acquire does not wait
    (void)period_s; uint64_t n = present_count();
    if (n == 1) detclock_first_present();
    char sp[1024]; const char *shot = g_nshot && g_vk ? shot_for(n, sp, sizeof sp) : NULL;
    if (shot && !t) { NRED_ONCE("screenshot %s: present %llu has no window texture: not taken", shot, (unsigned long long)n); shot = NULL; }
    // DM_VK_SHOT_SER=<serial>: the screenshots are of that texture's host image instead (an RGBA8 render target: which pass of the chain to the window draws nothing)
    // a list (DM_VK_SHOT_SER=10,3,33151): the k-th screenshot of DM_NVN_SHOT takes the k-th serial (the last for the rest), several targets of one moment in one run
    static int64_t sser = -1; static uint64_t sl[16]; static int ns, ks;
    if (sser < 0) { const char *e = getenv("DM_VK_SHOT_SER"); sser = 0; for (const char *q = e; q && *q && ns < 16; ) { sl[ns++] = strtoull(q, NULL, 10); q = strchr(q, ','); if (q) q++; } if (ns) sser = (int64_t)sl[0]; }
    if (shot && ns) { sser = (int64_t)sl[ks < ns ? ks : ns - 1]; ks++; }
    if (shot && sser && g_vk) {
        pthread_mutex_lock(&vkn_tmu); vkn_img *e = vkn_img_get((uint64_t)sser); uint32_t w = e ? e->w : 0, h = e ? e->h : 0; pthread_mutex_unlock(&vkn_tmu);
        if (w) return vkn_present_enqueue((uint64_t)sser, w, h, NULL, 0, shot);
        NRED_ONCE("DM_VK_SHOT_SER=%lld: no host image with that serial at present %llu: the window texture is taken", (long long)sser, (unsigned long long)n);
    }
    return g_vk && t ? vkn_present_enqueue(t->serial, t->d.width, t->d.height, crop, n == 1 || n == 60 || n == 600 || n % 1800 == 0, shot) : 0;
}
uint64_t mtl_present_ext(uint64_t serial, double period, void *ev, uint64_t ev_val, const int crop[4]) {   // no external producers (dm/gfx/vk is not built)
    (void)serial; (void)period; (void)ev; (void)ev_val; (void)crop; uint64_t n = present_count();
    char sp[1024]; const char *shot = g_nshot && g_vk ? shot_for(n, sp, sizeof sp) : NULL;
    if (shot) NRED_ONCE("screenshot %s: present %llu is an external producer's: not taken", shot, (unsigned long long)n);
    return 0;
}
uint64_t mtl_present_count(void) { return atomic_load(&g_present_n); }
void mtl_present_wait(uint64_t ticket) { if (g_vk && ticket) vkn_present_wait(ticket); }
void mtl_present_flush(void) { if (g_vk) vkn_present_flush(); }
void mtl_present_discard(int on) { (void)on; }
void mtl_present_test_stall(uint64_t ns) { (void)ns; }
double mtl_presented_time(uint64_t n) { (void)n; return 0; }   // no present feedback yet: "not ready", so no frame counts as dropped
int mtl_display_hz(void) { return 0; }   // as the null backend: hle_nvn.c paces by its own clock (the display-link paths stay off until the pacer is ported, PLAN.md 2.1)
int mtl_has_window(void) { return 0; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
uint32_t mtl_present_qos(void) { return 0; }
int mtl_fps_target(void) {   // as nvn_mtl.m: DM_FPS=30 / 60 pins it, otherwise the game's own rate
    static int env = -2; if (env == -2) { const char *e = getenv("DM_FPS"); env = e ? atoi(e) : -1; if (env != 30 && env != 60) env = -1; else NLOG("DM_FPS=%d: frame rate pinned by the test switch", env); }
    return env > 0 ? env : nvn_game_fps() == 30 ? 30 : 60;
}
void mtl_set_frame(uint64_t f) { (void)f; }
void mtl_ub_present(double iv_ms) { (void)iv_ms; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
int mtl_warm_wait(int timeout_ms) { (void)timeout_ms; return 0; }   // no pipeline warm-up running
void *mtl_device(void) { return NULL; }
void *mtl_device_classic(void) { return NULL; }
uint64_t mtl_ext_tex(void *tex) { (void)tex; return 0; }

// ---------- Textures: host images for render targets and sampled textures, registrations and samplers (above); a program's modules and pipelines are the draws' (prog_ent_get, by its serial) ----------
void mtl_set_depth_zero(int on) { atomic_store(&g_depth_zero, on != 0); }   // NVN's depth mode: z clipped at 0 (on) or -w (draw_key)
void mtl_set_origin_lower_left(int on) { if (atomic_exchange(&g_origin_ll, on) != on) NLOG("window origin: %s", on ? "lower left (the present flips the picture)" : "upper left"); }
void mtl_set_tex_pool_n(int n) { pthread_mutex_lock(&g_rmu); g_tpool_n = n; pthread_mutex_unlock(&g_rmu); atomic_fetch_add(&g_reg_gen, 1); }   // IDs past it are empty slots (tex_lookup)
void mtl_tex_init(const tex_t *t) {   // a texture in a CPU-no-access pool (the render-target pools) gets its host image now; others when first set as a target
    mtl_init();
    if (!t || !t->serial) return;
    uint64_t s = t->serial; pthread_mutex_lock(&g_tsmu); nvn_hm_put(&g_tser, &s, (void *)1); pthread_mutex_unlock(&g_tsmu);   // live (without a device too: the statistics use it)
    if (!g_vk || !t->pool || !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) return;
    pthread_mutex_lock(&g_pmu); for (int k = 0; k < g_npool; k++) if (g_pools[k].key == t->pool->mem) { g_pools[k].noacc = 1; break; } pthread_mutex_unlock(&g_pmu);   // B: no shadow there (b_mark)
    pthread_mutex_lock(&vkn_tmu); if (!vkn_img_get(t->serial)) img_make(t, NULL, NVK_RT); pthread_mutex_unlock(&vkn_tmu);
}
void mtl_tex_release(tex_t *t) {   // out of the table now; destroyed once the submissions and presents that used it have completed
    if (!t || !t->serial) return;
    uint64_t s = t->serial; pthread_mutex_lock(&g_tsmu); nvn_hm_del(&g_tser, &s); pthread_mutex_unlock(&g_tsmu);   // not live: a record replayed later gets a ghost image (img_make)
    if (!g_vk) return;
    pthread_mutex_lock(&vkn_tmu);
    vkn_img **p = it_slot(t->serial), *e = *p;
    if (e) { *p = e->next; img_dead(e); g_nimg--; }
    vkn_sweep();
    pthread_mutex_unlock(&vkn_tmu);
}
// The CPU writes the texture's content (WriteTexels, a packaged layout, a flush over it: hle_nvn.c): uploaded from guest memory from its next use on, and again after guest writes (tex_ready)
void mtl_tex_cpu_written(const tex_t *t) {
    if (!t || !t->serial) return;
    uint64_t s = t->serial; int added = 0;
    pthread_mutex_lock(&g_tsmu); uintptr_t v = (uintptr_t)nvn_hm_get(&g_tser, &s); if (v && !(v & 4)) { nvn_hm_put(&g_tser, &s, (void *)(v | 4)); added = 1; } pthread_mutex_unlock(&g_tsmu);
    if (!added) return;
    if (TSA(TS_CPUW, 1) < 16) NLOG("CPU-written texture ser=%llu %ux%ux%u format %#x target %d levels %u: uploaded from guest memory, again after a guest write", (unsigned long long)s, t->d.width, t->d.height,
                                  t->d.depth, t->d.format, t->d.target, t->d.levels);
    atomic_fetch_add(&g_reg_gen, 1);   // the slots bound to it are resolved again (now with its upload)
}
// The GPU owns the content (nvn_mtl.m's: a host texture made to be rendered into): its host image was made for a target or the GPU drew it, and the CPU has not written it since (hle_nvn.c
// nvn_ft_poll never makes such a texture CPU-written)
int mtl_tex_gpu_owned(uint64_t serial) {
    if (!g_vk) return 0;
    uintptr_t v = tser_get(serial); if (v & 4) return 0;
    pthread_mutex_lock(&vkn_tmu); vkn_img *e = vkn_img_get(serial); int r = e && (e->gpu || (v & 2)); pthread_mutex_unlock(&vkn_tmu);
    return r;
}
// Texture ID → the texture object and view as registered (copied: the game may reuse its objects); registering what the ID holds already changes nothing (nvn_mtl.m). Counted as well: the formats
// and targets a texture table needs (the surface line)
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
    if (t) { fmt_note(1, (unsigned)t->d.format); if (v && v->set_format) fmt_note(1, v->format); atomic_fetch_or_explicit(&g_tgt, 1u << (t->d.target & 31), memory_order_relaxed); }
    if (id < 0) return;
    if (id >= (1 << 20)) { NRED_ONCE("texture ID %d registered, past the 20 bits a handle holds: ignored", id); return; }
    pthread_mutex_lock(&g_rmu);
    if (id >= g_ntreg) { int n = id + 1024; treg_t *g = realloc(g_treg, (size_t)n * sizeof *g); if (!g) die("vkn backend: texture table allocation failed"); memset(g + g_ntreg, 0, (size_t)(n - g_ntreg) * sizeof *g); g_treg = g; g_ntreg = n; }
    treg_t *r = &g_treg[id];
    int same = t && r->ok && r->has_v == (v != NULL) && !memcmp(&r->t, t, sizeof *t) && (!v || !memcmp(&r->v, v, sizeof *v));
    if (!same) { r->ok = t != NULL; if (t) r->t = *t; r->has_v = v != NULL; if (v) r->v = *v; atomic_fetch_add(&g_reg_gen, 1); }
    pthread_mutex_unlock(&g_rmu);
}
void mtl_reg_samp(int id, const samp_par *p) {   // sampler ID → its parameters; the VkSampler comes at the first lookup (samp_lookup)
    if (id < 0 || !p) return;
    if (id >= (1 << 20)) { NRED_ONCE("sampler ID %d registered, past the bits a handle holds: ignored", id); return; }
    pthread_mutex_lock(&g_rmu);
    if (id >= g_nsreg) { int n = id + 256; sreg_t *g = realloc(g_sreg, (size_t)n * sizeof *g); if (!g) die("vkn backend: sampler table allocation failed"); memset(g + g_nsreg, 0, (size_t)(n - g_nsreg) * sizeof *g); g_sreg = g; g_nsreg = n; }
    sreg_t *r = &g_sreg[id];
    if (!r->ok || memcmp(&r->p, p, sizeof *p)) { r->p = *p; r->s = r->si = VK_NULL_HANDLE; r->ok = 1; atomic_fetch_add(&g_reg_gen, 1); }
    pthread_mutex_unlock(&g_rmu);
}
void mtl_prog_release(prog_t *p) { (void)p; }
void bf_want_note(const char *key) { (void)key; }

// ReadTexels (row / image strides as hle_nvn.c read_texels worked them out), as the null backend: the region is read from guest memory, de-laid-out as the Metal layer's CPU path reads textures the
// CPU filled (nvn_mtl.m mtl_read_texels); zeros only for a texture without storage or a layout it cannot undo. Right for a CPU-written texture; one only the GPU wrote would need a readback from its
// host image, which this backend does not do yet (said once: its storage reads as what the pool holds)
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) { NRED_ONCE("ReadTexels: format %#x unknown, nothing read", t->d.format); return; }
    { uintptr_t v = tser_get(t->serial); if ((v & 2) && !(v & 4) && !t->has_layout)
        NRED_ONCE("ReadTexels of texture ser=%llu, which the GPU wrote: no GPU readback yet, read from guest memory (first case)", (unsigned long long)t->serial); }
    size_t wb = (size_t)((r->width + bw - 1) / bw) * (size_t)bpb, hb = (size_t)((r->height + bh - 1) / bh), x0 = (size_t)(r->xoffset / bw) * (size_t)bpb, y0 = (size_t)(r->yoffset / bh);
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    uint32_t lw = t->d.width >> level, lh = t->d.height >> level, lz = is3d ? t->d.depth >> level : 1;
    if (!lw) lw = 1; if (!lh) lh = 1; if (!lz) lz = 1;
    size_t lp = (size_t)((lw + bw - 1) / bw) * (size_t)bpb, lr = (lh + bh - 1) / bh;
    ST(S_RTX, 1); ST(S_RTX_B, wb * hb * (size_t)nz); if (gpu_written(t->serial)) ST(S_RTX_GPU, 1);   // one of a texture the GPU writes would be a readback in a real backend
    uint8_t *lin = t->mem ? malloc(lp * lr * lz) : NULL;
    for (int z = 0; z < nz; z++) {
        const uint8_t *src = NULL;
        if (lin && ((is3d && z == 0) || !is3d) && nvn_deswizzle(&t->d, level, is3d ? 0 : layer0 + (unsigned)r->zoffset + (unsigned)z, t->mem, lin, lp)) { NRED_ONCE("ReadTexels: storage layout not undone, the region reads as zeros"); free(lin); lin = NULL; }
        if (lin) src = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
        for (size_t y = 0; y < hb; y++) { uint8_t *d = dst + (size_t)z * img + y * row; if (src) memcpy(d, src + (y0 + y) * lp + x0, wb); else memset(d, 0, wb); }
    }
    free(lin);
}

// ---------- Statistics and diagnostics: passes, draws, sampled-texture uploads, modules and pipelines made; the rest has nothing to count yet ----------
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = atomic_load(&g_stat_pass); *draws = atomic_load(&g_dr_n); }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
void mtl_vbuf_stats(uint64_t *resolve, uint64_t *reuse, uint64_t *set, uint64_t *skip) { *resolve = *reuse = *set = *skip = 0; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = *store = *full = 0; }
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = atomic_load(&g_ts[TS_UP1]); *reuploads = atomic_load(&g_ts[TS_RE]) + atomic_load(&g_ts[TS_REALW]); }   // sampled textures uploaded from guest memory
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { uint64_t r; vkn_shader_stats(lib, &r); *pso = atomic_load(&g_pso_made); }   // shader modules loaded, pipelines made
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = 0; *pso_ns = atomic_load(&g_pso_ns); }
void mtl_alloc_stats(uint64_t *waits, uint64_t *grown, int *n) { *waits = *grown = 0; *n = 0; }
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = *full = 0; }
void mtl_ubd_stats(uint64_t *deferred, uint64_t *conflicts) { *deferred = *conflicts = 0; }
void mtl_ubp_stats(uint64_t *made, uint64_t *reused, uint64_t *refused) { *made = *reused = *refused = 0; }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
void mtl_lc_stats(uint64_t *taken, uint64_t *landed) { *taken = *landed = 0; }
uint64_t mtl_ds_omit(void) { return 0; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_pkg_edge_skips(void) { return atomic_load(&g_ts[TS_EDGE]); }
uint64_t mtl_pkg_miss(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return atomic_load(&g_ts[TS_UP1_B]) + atomic_load(&g_ts[TS_RE_B]) + atomic_load(&g_ts[TS_REALW_B]) + atomic_load(&g_ts[TS_INIT_B]); }
uint64_t mtl_upload_ns_take(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_blit_splits(void) { return 0; }
double mtl_gpu_ms(uint64_t f) { (void)f; return -1; }   // no GPU time to report
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_res_stats(char *out, size_t n) { if (n) snprintf(out, n, "-"); }
void mtl_res_counts(uint32_t tick, const char *why) { (void)tick; (void)why; }
void mtl_res_snapshot(mtl_res_count *out) { memset(out, 0, sizeof *out); }
void xv_test_counts(int *dc, int *viol) { if (dc) *dc = 0; if (viol) *viol = 0; }
// The memory readout line (hle_nvn.c, every 300 frames): resident memory only, with the Metal layer's throttle (a change of 64 MB or 3000 frames, at most 400 lines)
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) {
    static unsigned long long last_rss, last_frame; static int lines;
    unsigned long long dr = rss_mb > last_rss ? rss_mb - last_rss : last_rss - rss_mb;
    if (lines >= 400 || (lines && dr < 64 && frame - last_frame < 3000)) return;
    lines++; last_rss = rss_mb; last_frame = frame;
    NLOG("memory: frame %llu resident %llu MB (host images not counted here: the GPU work lines have them)", frame, rss_mb);
}
// DM_CTL screenshots and dumps (hle_hid.c): no pictures to take
void dm_gfx_shot_at(long k, const char *path) { (void)k; NRED_ONCE("screenshot %s: no pictures on the Vulkan backend yet", path); }
void dm_gfx_texdump_next(const char *path) { (void)path; NRED_ONCE("texture dump: no textures on the Vulkan backend yet"); }
void dm_gfx_passdump_next(long n) { (void)n; NRED_ONCE("pass dump: no passes on the Vulkan backend yet"); }
void dm_gfx_texlist(int target) { (void)target; NRED_ONCE("texture list: no textures on the Vulkan backend yet"); }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; NRED_ONCE("raw texture dump: no textures on the Vulkan backend yet"); }
