// Shader modules of the draws (docs/android/PLAN.md 2.4, slice 2c): a program's NVN stages → slice 0's SPIR-V modules, as nvn_mtl.m maps them to metallibs (shmap_load, prog_get, shader_get).
// The shader map (DM_NVN_SHMAP, else <DM_VK_SHADERS>/shader_map.tsv) names each key's stage; the module is <DM_VK_SHADERS>/<key>_b.json (dmsh_build_vk.py's reflection) beside <key>_b.spv (the
// .spv.zst decompressed). A game's APK carries that directory (scripts/build_apk_from_nsp.py: assets/nx/vksh), and the Android front end points DM_VK_SHADERS at it once unpacked
// (<files>/pkg/vksh, front_android.c) unless env.txt names another; on the Mac and for development it is a directory given by hand. Each module's resources are moved into the draws' one push-descriptor set (set 0): vertex bindings keep their
// numbers, pixel bindings get 32 more (the DescriptorSet / Binding decorations are patched; nothing else of the module changes). A module that cannot be used as built (its status, chain, stage,
// a binding kind or image type this slice does not bind, a descriptor array, a slot out of its range, decoration groups, a capability the device does not take, a size, SHA-256 or magic that does not
// match, an entry point not of its stage) is refused, said once in red with why, and its draws are skipped. Only the decompressed <key>_b.spv is read.
// Submit thread only (the replay), and the tests' main thread; modules are kept for the run
#include "vkn.h"
#include "../nvn/nvn_hmap.h"
#include "../../host/dm.h"
#include "../../host/plat/sha256.h"   // the vendored portable SHA-256 (header only; plat.h's stand-in for CommonCrypto on Linux): the module's spv_sha256
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------- JSON: a small reader for the reflection files ----------
enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };
typedef struct jv { int t, n; double num; char *s; struct jv *c; char **k; } jv;   // arrays and objects: n children c (objects: keys k)
static const char *j_ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; return p; }
static const char *j_str(const char *p, char **out) {   // p after the opening quote; escapes kept to ASCII (\uXXXX outside it becomes '?')
    size_t cap = 16, n = 0; char *s = malloc(cap);
    if (!s) return NULL;
    for (; *p && *p != '"'; p++) {
        char c = *p;
        if (c == '\\') {
            c = *++p;
            switch (c) { case 'n': c = '\n'; break; case 't': c = '\t'; break; case 'r': c = '\r'; break; case 'b': c = '\b'; break; case 'f': c = '\f'; break;
                case 'u': { unsigned u = 0; for (int i = 0; i < 4; i++) { char h = *++p; if (!h) { free(s); return NULL; } u = u * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10); } c = u < 128 ? (char)u : '?'; break; }
                case 0: free(s); return NULL;
                default: break; }
        }
        if (n + 2 > cap) { char *t = realloc(s, cap *= 2); if (!t) { free(s); return NULL; } s = t; }
        s[n++] = c;
    }
    if (*p != '"') { free(s); return NULL; }
    s[n] = 0; *out = s; return p + 1;
}
static void j_free(jv *v) {
    if (v->t == J_STR) free(v->s);
    if (v->t == J_ARR || v->t == J_OBJ) { for (int i = 0; i < v->n; i++) { j_free(&v->c[i]); if (v->k) free(v->k[i]); } free(v->c); free(v->k); }
    memset(v, 0, sizeof *v);
}
static const char *j_val(const char *p, jv *o, int depth) {
    memset(o, 0, sizeof *o); p = j_ws(p);
    if (depth > 32) return NULL;
    if (*p == '"') { o->t = J_STR; return j_str(p + 1, &o->s); }
    if (*p == '{' || *p == '[') {
        int obj = *p == '{'; char end = obj ? '}' : ']'; int cap = 0;
        o->t = obj ? J_OBJ : J_ARR; p = j_ws(p + 1);
        if (*p == end) return p + 1;
        for (;;) {
            if (o->n == cap) { cap = cap ? cap * 2 : 8; jv *c = realloc(o->c, (size_t)cap * sizeof *c); char **k = obj ? realloc(o->k, (size_t)cap * sizeof *k) : NULL;
                               if (!c || (obj && !k)) { free(c ? c : o->c); o->c = NULL; free(k ? k : o->k); o->k = NULL; o->n = 0; return NULL; } o->c = c; if (obj) o->k = k; }
            char *key = NULL;
            if (obj) { p = j_ws(p); if (*p != '"' || !(p = j_str(p + 1, &key))) return NULL; p = j_ws(p); if (*p != ':') { free(key); return NULL; } p++; }
            if (obj) o->k[o->n] = key;
            p = j_val(p, &o->c[o->n], depth + 1); o->n++;
            if (!p) return NULL;
            p = j_ws(p);
            if (*p == ',') { p++; continue; }
            if (*p == end) return p + 1;
            return NULL;
        }
    }
    if (!strncmp(p, "true", 4)) { o->t = J_BOOL; o->num = 1; return p + 4; }
    if (!strncmp(p, "false", 5)) { o->t = J_BOOL; return p + 5; }
    if (!strncmp(p, "null", 4)) return p + 4;
    char *e; o->num = strtod(p, &e); if (e == p) return NULL;
    o->t = J_NUM; return e;
}
static const jv *j_get(const jv *o, const char *k) { if (o && o->t == J_OBJ) for (int i = 0; i < o->n; i++) if (!strcmp(o->k[i], k)) return &o->c[i]; return NULL; }
static const char *j_s(const jv *o, const char *k) { const jv *v = j_get(o, k); return v && v->t == J_STR ? v->s : ""; }
// A number as an integer: dflt where the key is absent or not a number (null); J_BAD where it is no integer within int's range (NaN, 1e300, 1.5: the reader takes strtod's forms), which every
// caller's range check refuses (a field is never filled with a cut-down value)
#define J_BAD LONG_MIN
static long j_i(const jv *o, const char *k, long dflt) {
    const jv *v = j_get(o, k);
    if (!v || (v->t != J_NUM && v->t != J_BOOL)) return dflt;
    return v->num >= INT_MIN && v->num <= INT_MAX && v->num == (double)(long)v->num ? (long)v->num : J_BAD;
}

// ---------- The shader map and the module directory ----------
static nvn_hmap g_map = {.klen = 8};      // key → NVN stage + 1
static nvn_hmap g_mods = {.klen = 8};     // key → vkn_shader, or REFUSED
#define REFUSED ((void *)1)
static const char *g_dir; static int g_init, g_dir_ok;
static _Atomic uint64_t g_nload, g_nrefuse;   // modules loaded and refused: counted here (the submit thread), read by the statistics line of the present (draw_report) and mtl_create_stats
static int hex8(const char *s, uint8_t k[8]) {
    for (int i = 0; i < 16; i++) { char c = s[i]; int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; if (v < 0) return 0; if (i & 1) k[i >> 1] |= (uint8_t)v; else k[i >> 1] = (uint8_t)(v << 4); }
    return s[16] == 0 || s[16] == '\t' || s[16] == '\n' || s[16] == '\r' || s[16] == '_';
}
static int stage_of(const char *s) {   // the map's stage names (nvn_mtl.m stage_of): NVN stage numbers
    return !strcmp(s, "vertex") ? 0 : !strcmp(s, "pixel") ? 1 : !strcmp(s, "geometry") ? 2 : !strcmp(s, "hull") ? 3 : !strcmp(s, "domain") ? 4 : !strcmp(s, "compute") ? 5 : -1;
}
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = z >= 0 ? malloc((size_t)z + 1) : NULL;
    if (b && fread(b, 1, (size_t)z, f) != (size_t)z) { free(b); b = NULL; }
    fclose(f); if (b) { b[z] = 0; if (n) *n = (size_t)z; }
    return b;
}
static void map_text(const char *txt, const char *path) {
    int n = 0;
    for (const char *l = txt; l && *l; ) {
        const char *e = strchr(l, '\n'); size_t len = e ? (size_t)(e - l) : strlen(l);
        if (len && *l != '#' && strncmp(l, "package", 7)) {
            char ln[512]; if (len >= sizeof ln) len = sizeof ln - 1;
            memcpy(ln, l, len); ln[len] = 0;
            char *c[5] = {0}; int nc = 0; for (char *q = ln, *t; nc < 5 && (t = strsep(&q, "\t")); ) c[nc++] = t;
            uint8_t k[8];
            if (nc > 3 && strlen(c[3]) >= 16 && hex8(c[3], k) && stage_of(c[2]) >= 0) { nvn_hm_put(&g_map, k, (void *)(uintptr_t)(stage_of(c[2]) + 1)); n++; }
        }
        l = e ? e + 1 : NULL;
    }
    VLOG("shader map %s: %d keys", path, n);
}
static void shaders_init(void) {
    if (g_init) return;
    g_init = 1;
    g_dir = getenv("DM_VK_SHADERS");
    const char *mp = getenv("DM_NVN_SHMAP"); char path[1024];
    if (!mp && g_dir) { snprintf(path, sizeof path, "%s/shader_map.tsv", g_dir); mp = path; }
    char *t = mp ? slurp(mp, NULL) : NULL;
    if (!t) VLOG("[RED] no shader map (%s; DM_NVN_SHMAP or <DM_VK_SHADERS>/shader_map.tsv names it): nothing is drawn", mp ? mp : "none given");
    else { map_text(t, mp); free(t); }
    if (!g_dir) { VLOG("[RED] no shader module directory (DM_VK_SHADERS: the modules decompressed; a game's APK sets it to the ones it carries, scripts/build_apk_from_nsp.py): nothing is drawn"); return; }
    // The directory's stamp (inputs.tsv, as nvn_mtl.m's shmap_load reads it): the game it was built for
    snprintf(path, sizeof path, "%s/inputs.tsv", g_dir);
    char *st = slurp(path, NULL), game[64] = "", inputs[64] = "";
    for (char *l = st, *e; l && *l; l = e ? e + 1 : NULL) { e = strchr(l, '\n'); if (e) *e = 0;
        if (!strncmp(l, "game\t", 5)) snprintf(game, sizeof game, "%s", l + 5); else if (!strncmp(l, "inputs\t", 7)) snprintf(inputs, sizeof inputs, "%s", l + 7); }
    free(st);
    if (!*game) VLOG("[RED] shader modules %s: no stamp (inputs.tsv): the game they were built for is not known; used regardless", g_dir);
    else if (strcmp(game, DM_GAME_ID)) { VLOG("[RED] shader modules %s were built for game %s, not " DM_GAME_ID ": nothing is drawn", g_dir, game); return; }
    g_dir_ok = 1;
    VLOG("shader modules: %s, chain %s, game %s, inputs %s (loaded at their first draw)", g_dir, VKN_CHAIN, *game ? game : "?", *inputs ? inputs : "?");
}
int vkn_shmap_stage(const uint8_t key[8]) { shaders_init(); uintptr_t v = (uintptr_t)nvn_hm_get(&g_map, key); return v ? (int)v - 1 : -1; }
void vkn_shader_stats(uint64_t *loaded, uint64_t *refused) { *loaded = atomic_load_explicit(&g_nload, memory_order_relaxed); *refused = atomic_load_explicit(&g_nrefuse, memory_order_relaxed); }

// ---------- A module: its reflection checked and kept, its SPIR-V's resources moved into the one set ----------
// A component type as the reflection names it (dmsh_build_vk.py scalar(): float, uint or sint, with the bit width after it where that is not 32): 0 float, 1 uint, 2 sint. Every other name is
// -1, the 16- and 64-bit ones (float16, uint16, sint64...) included: the formats, dummies and pair matching here are 32-bit, those types a later slice (the builder refuses them on interface
// variables already); ct_no says so in the refusal
static int ct_of(const char *t) { return !strcmp(t, "float") ? 0 : !strcmp(t, "uint") ? 1 : !strcmp(t, "sint") ? 2 : -1; }
static const char *ct_no(int ct) { return ct < 0 ? ": not a 32-bit float, uint or sint (16- and 64-bit types are a later slice)" : ""; }
static int img_kind(const jv *im, uint8_t *vt, uint8_t *comp, char *why, size_t wn) {
    const char *dim = j_s(im, "dim"), *cp = j_s(im, "comp"); long arr = j_i(im, "arrayed", 0), ms = j_i(im, "ms", 0), dep = j_i(im, "depth", 0);
    if (arr < 0 || arr > 1 || ms < 0 || ms > 1 || dep < 0 || dep > 2) { snprintf(why, wn, "image flags arrayed %ld ms %ld depth %ld", arr, ms, dep); return 0; }
    if (ms) { snprintf(why, wn, "a multisampled image"); return 0; }
    if (dep == 1) { snprintf(why, wn, "a depth-compare image"); return 0; }
    if (!strcmp(dim, "2D")) *vt = arr ? VKN_VT_2D_ARRAY : VKN_VT_2D;
    else if (!strcmp(dim, "Cube") && !arr) *vt = VKN_VT_CUBE;
    else if (!strcmp(dim, "3D") && !arr && !(getenv("DM_VK_3D") && *getenv("DM_VK_3D") == '0')) *vt = VKN_VT_3D;   // read as an empty 3D image until 3D textures have host images (nvn_vk.c tex_lookup); DM_VK_3D=0: refused
    else { snprintf(why, wn, "image dim %s%s", dim, arr ? " arrayed" : ""); return 0; }
    int ct = ct_of(cp);
    if (ct < 0) { snprintf(why, wn, "image component type %s%s", cp, ct_no(ct)); return 0; }
    *comp = (uint8_t)ct;
    return 1;
}
// The reflection's bindings and interface into sh; why: the first thing this slice does not handle
static int refl_take(vkn_shader *sh, const jv *j, char *why, size_t wn) {
    const jv *bs = j_get(j, "bindings"), *vi = j_get(j, "vertex_inputs"), *outs = j_get(j, "outputs"), *dr = j_get(j, "drv_reads"), *io = j_get(j, sh->stage == 0 ? "outputs" : "inputs");
    if (bs && bs->t != J_ARR) { snprintf(why, wn, "bindings is not a list"); return 0; }
    for (int i = 0; bs && i < bs->n; i++) {
        const jv *b = &bs->c[i]; const char *kind = j_s(b, "kind"), *tn = j_s(b, "type_name"); long bi = j_i(b, "binding", -1), set = j_i(b, "set", -1), reg = j_i(b, "reg", -1);
        long cnt = j_get(b, "count") ? j_i(b, "count", J_BAD) : 1;   // null (a runtime or specialization-sized array) is no count of 1
        const jv *sm = j_get(b, "sampler"); long sreg = j_i(sm, "reg", -1);
        if (bi == J_BAD || set == J_BAD || reg == J_BAD || sreg == J_BAD) { snprintf(why, wn, "binding %s: its binding, set or a slot is a number that is no integer of int's range", j_s(b, "name")); return 0; }
        if (sh->nb == VKN_MAXB) { snprintf(why, wn, "more than %d bindings", VKN_MAXB); return 0; }
        vkn_sbind *o = &sh->b[sh->nb];
        memset(o, 0, sizeof *o);
        if (bi < 0 || bi >= 128) { snprintf(why, wn, "binding %ld (%s) outside 0-127", bi, j_s(b, "name")); return 0; }
        if (set != (sh->stage == 1 ? 1 : 0)) { snprintf(why, wn, "binding %ld in set %ld", bi, set); return 0; }
        // Ryujinx's fallback for a texture handle the translator cannot trace to one slot (<stage>_<t|s>_cb0_40000000): an array of textures / samplers indexed by the handle's
        // IDs. With the profile's tex_dyn_slots, c2's tables hold slot indices and element k is what slot k holds (nvn_vk.c draw_drv2, draw_dyn; nvn_mtl.m NVN_RB_TEXD / SAMPD)
        { const char *nm = j_s(b, "name"); size_t ln = strlen(nm);
          // A handle the shader reads from a uniform buffer (Ryujinx's bindless texture from a buffer: kind tex_ubo, reg the uniform slot, off its byte), one or an array
          // indexed by the handle's offset / 8 (nvn_mtl.m NVN_RB_TEXU): nvn_vk.c draw_texu. DM_VK_TEXU=0: refused
          if (!strcmp(j_s(b, "kind"), "tex_ubo") && !strcmp(tn, "COMBINED_IMAGE_SAMPLER") && cnt >= 1 && cnt <= 64 && !(getenv("DM_VK_TEXU") && *getenv("DM_VK_TEXU") == '0')) {
            if (sh->nb == VKN_MAXB) { snprintf(why, wn, "more than %d bindings", VKN_MAXB); return 0; }
            long off = j_i(b, "off", 0);
            if (reg < 0 || reg >= 16 || off < 0 || off > 65535) { snprintf(why, wn, "binding %ld (%s): uniform slot %ld offset %ld", bi, nm, reg, off); return 0; }
            vkn_sbind *o = &sh->b[sh->nb]; memset(o, 0, sizeof *o);
            o->set = (uint8_t)set; o->orig = (uint16_t)bi; o->binding = (uint16_t)(bi + (sh->stage == 1 ? 128 : 0)); o->n = (uint8_t)cnt; o->kind = VKN_B_TEXU; o->reg = (uint16_t)reg; o->off = (uint16_t)off;
            if (!img_kind(j_get(b, "image"), &o->vt, &o->comp, why, wn)) return 0;
            for (int k = 0; k < sh->nb; k++) if (sh->b[k].binding == o->binding) { snprintf(why, wn, "binding %ld twice", bi); return 0; }
            sh->nb++; continue; }
          if (DM_GAME_TEX_DYN_SLOTS && cnt >= 1 && cnt <= 64 && ln > 13 && !strcmp(nm + ln - 13, "_cb0_40000000") && !(getenv("DM_VK_TEXD") && *getenv("DM_VK_TEXD") == '0')) {
            if (sh->nb == VKN_MAXB) { snprintf(why, wn, "more than %d bindings", VKN_MAXB); return 0; }
            vkn_sbind *o = &sh->b[sh->nb]; memset(o, 0, sizeof *o);
            o->set = (uint8_t)set; o->orig = (uint16_t)bi; o->binding = (uint16_t)(bi + (sh->stage == 1 ? 128 : 0)); o->n = (uint8_t)cnt;
            if (!strcmp(tn, "SAMPLED_IMAGE")) { o->kind = VKN_B_TEXD; if (!img_kind(j_get(b, "image"), &o->vt, &o->comp, why, wn)) return 0; }
            else if (!strcmp(tn, "SAMPLER")) o->kind = VKN_B_SAMPD;
            else { snprintf(why, wn, "binding %ld (%s): a %s array", bi, nm, tn); return 0; }
            for (int k = 0; k < sh->nb; k++) if (sh->b[k].binding == o->binding) { snprintf(why, wn, "binding %ld twice", bi); return 0; }
            sh->nb++; sh->dyn = 1; continue; } }
        if (cnt != 1) { if (cnt == J_BAD) snprintf(why, wn, "binding %ld (%s) is an array (count null or not an integer)", bi, j_s(b, "name")); else snprintf(why, wn, "binding %ld (%s) is an array of %ld", bi, j_s(b, "name"), cnt); return 0; }
        o->set = (uint8_t)set; o->orig = (uint16_t)bi; o->binding = (uint16_t)(bi + (sh->stage == 1 ? 128 : 0));   // pixel bindings 128 up: vertex ones keep 0-127
        // Slots (reg) are range-checked as read, never cut to their field: a uniform slot 0-15, a texture or sampler slot 0-127, a sampler ID 0-65535; c1 and driver banks have none
        if (!strcmp(kind, "cbuf") && !strcmp(tn, "UNIFORM_BUFFER")) { o->kind = VKN_B_UBO; if (reg < 0 || reg >= 16) { snprintf(why, wn, "uniform slot %ld", reg); return 0; } o->reg = (uint16_t)reg; }
        else if (!strcmp(kind, "c1") && !strcmp(tn, "UNIFORM_BUFFER")) o->kind = VKN_B_C1;
        else if (!strcmp(kind, "ubo_ptr") && !strcmp(tn, "STORAGE_BUFFER") && !(getenv("DM_VK_UBOP") && *getenv("DM_VK_UBOP") == '0')) {   // DM_VK_UBOP=0: refused as before
            o->kind = VKN_B_UBOP; if (reg < 0 || reg >= 16) { snprintf(why, wn, "uniform slot %ld", reg); return 0; } o->reg = (uint16_t)reg; }
        else if (!strcmp(kind, "sbuf") && !strcmp(tn, "STORAGE_BUFFER") && !(getenv("DM_VK_SBUF") && *getenv("DM_VK_SBUF") == '0')) {   // DM_VK_SBUF=0: refused as before
            o->kind = VKN_B_SBUF; o->rw = j_get(b, "rw") && j_get(b, "rw")->t == J_BOOL && j_get(b, "rw")->num != 0;
            if (reg < 0 || reg >= 16) { snprintf(why, wn, "storage buffer slot %ld", reg); return 0; } o->reg = (uint16_t)reg; }
        else if (!strcmp(kind, "support") && !strcmp(tn, "UNIFORM_BUFFER") && !(getenv("DM_VK_SUPPORT") && *getenv("DM_VK_SUPPORT") == '0')) o->kind = VKN_B_SUPPORT;   // DM_VK_SUPPORT=0: refused as before
        else if (!strcmp(kind, "drvcb") && !strcmp(tn, "UNIFORM_BUFFER")) {   // a driver bank the shader reads nothing of (drv_reads empty): the zero buffer
            // Driver bank c0 read (the storage buffer table, a compute shader's dispatch size and uniform buffer table): filled per draw (nvn_vk.c draw_drv0); c2 reads are a later slice
            int rd = 0; for (int k = 0; dr && dr->t == J_OBJ && k < dr->n; k++) rd |= !strcmp(dr->k[k], "0") ? 1 : 2;
            long bank = j_i(b, "bank", -1);
            if (bank == 0 && (rd & 1) && !(getenv("DM_VK_DRV0") && *getenv("DM_VK_DRV0") == '0')) o->kind = VKN_B_DRV0;   // DM_VK_DRV0=0: refused as before
            else if (bank == 2 && (rd & 2) && !(getenv("DM_VK_DRV2") && *getenv("DM_VK_DRV2") == '0')) o->kind = VKN_B_DRV2;   // the handle tables (draw_drv2); DM_VK_DRV2=0: refused
            else if ((bank == 0 && (rd & 1)) || (bank != 0 && (rd & 2))) { snprintf(why, wn, "driver bank reads (drv_reads) of c%ld are a later slice", bank); return 0; }
            else o->kind = VKN_B_ZERO;
        } else if (!strcmp(kind, "tex") && !strcmp(tn, "COMBINED_IMAGE_SAMPLER")) {
            const char *sk = j_s(sm, "kind");
            o->kind = VKN_B_TEX;
            if (reg < 0 || reg >= 128) { snprintf(why, wn, "texture slot %ld (separate textures and images are a later slice)", reg); return 0; }
            if (!strcmp(sk, "samp_tex")) { o->skind = VKN_S_TEX; if (sreg < 0 || sreg >= 128) { snprintf(why, wn, "sampler of texture slot %ld", sreg); return 0; } }
            else if (!strcmp(sk, "samp")) { o->skind = VKN_S_SAMP; if (sreg < 0 || sreg >= 128) { snprintf(why, wn, "sampler slot %ld", sreg); return 0; } }
            else if (!strcmp(sk, "samp_id")) { o->skind = VKN_S_ID; if (sreg < 0 || sreg > 0xffff) { snprintf(why, wn, "sampler ID %ld", sreg); return 0; } }
            else { snprintf(why, wn, "sampler kind %s", *sk ? sk : "(none)"); return 0; }
            o->reg = (uint16_t)reg; o->sreg = (uint16_t)sreg;
            if (!img_kind(j_get(b, "image"), &o->vt, &o->comp, why, wn)) return 0;
        } else if (!strcmp(kind, "img") && !strcmp(tn, "STORAGE_IMAGE") && sh->stage == 5) {   // a storage image of a compute shader: NVN image slot reg, a 2D view of its host image
            o->kind = VKN_B_IMG;
            if (reg < 0 || reg >= 32) { snprintf(why, wn, "image slot %ld", reg); return 0; }
            o->reg = (uint16_t)reg;
            if (!img_kind(j_get(b, "image"), &o->vt, &o->comp, why, wn)) return 0;
            if (o->vt != VKN_VT_2D) { snprintf(why, wn, "a storage image of view type %d (only 2D: a later slice)", o->vt); return 0; }
        } else { snprintf(why, wn, "binding kind %s (%s)", *kind ? kind : "?", tn); return 0; }
        for (int k = 0; k < sh->nb; k++) if (sh->b[k].binding == o->binding) { snprintf(why, wn, "binding %ld twice", bi); return 0; }
        sh->nb++;
    }
    if (sh->stage == 0) for (int i = 0; vi && vi->t == J_ARR && i < vi->n; i++) {
        long loc = j_i(&vi->c[i], "location", -1), nc = j_i(&vi->c[i], "components", 4); int ct = ct_of(j_s(&vi->c[i], "type"));
        if (loc < 0 || loc >= 16 || ct < 0 || nc < 1 || nc > 4 || sh->nin == 16) { snprintf(why, wn, "vertex input %ld (%s x%ld)%s", loc, j_s(&vi->c[i], "type"), nc, ct_no(ct)); return 0; }
        sh->in[sh->nin].loc = (uint8_t)loc; sh->in[sh->nin].ct = (uint8_t)ct; sh->in[sh->nin++].n = (uint8_t)nc;
    }
    if (sh->stage == 1) for (int i = 0; outs && outs->t == J_ARR && i < outs->n; i++) {
        long loc = j_i(&outs->c[i], "location", -1), nc = j_i(&outs->c[i], "components", 4); int ct = ct_of(j_s(&outs->c[i], "type"));
        if (loc < 0 || loc >= 8 || ct < 0 || nc < 1 || nc > 4) { snprintf(why, wn, "pixel output %ld (%s x%ld)%s", loc, j_s(&outs->c[i], "type"), nc, ct_no(ct)); return 0; }
        sh->out_mask |= 1u << loc; sh->out_ct[loc] = (uint8_t)ct; sh->out_n[loc] = (uint8_t)nc;
    }
    // The interface between the stages (nvn_vk.c lay_get matches a pair on it): a vertex shader's outputs, a pixel shader's inputs, each listed (an empty list is a list), one entry per location.
    // An array variable's entries carry its first location and its lengths (array: {location, dims}, dmsh_build_vk.py io()): Vulkan matches variables, so a vec4[2] is not two vec4s
    if (sh->stage == 5) return 1;   // a compute shader has no interface between stages
    if (!io || io->t != J_ARR) { snprintf(why, wn, "its reflection lists no %s", sh->stage == 0 ? "vertex outputs" : "pixel inputs"); return 0; }
    for (int i = 0; i < io->n; i++) {
        const jv *e = &io->c[i], *ar = j_get(e, "array"), *dm = j_get(ar, "dims"); const char *side = sh->stage == 0 ? "vertex output" : "pixel input";
        long loc = j_i(e, "location", -1), nc = j_i(e, "components", -1), first = ar ? j_i(ar, "location", -1) : loc, span = 1; int ct = ct_of(j_s(e, "type"));
        if (loc < 0 || loc >= 32 || ct < 0 || nc < 1 || nc > 4 || (sh->io_mask >> loc & 1)) { snprintf(why, wn, "%s %ld (%s x%ld)%s", side, loc, j_s(e, "type"), nc, ct_no(ct)); return 0; }
        if (ar) {   // an object; its first location 0 up to this one; 1-4 lengths, each an integer 1-32; this location within them
            int ok = ar->t == J_OBJ && dm && dm->t == J_ARR && dm->n >= 1 && dm->n <= 4 && first >= 0 && first <= loc;
            for (int d = 0; ok && d < dm->n; d++) { const jv *x = &dm->c[d]; ok = x->t == J_NUM && x->num >= 1 && x->num <= 32 && x->num == (double)(long)x->num; if (ok) { sh->io_dims[loc][d] = (uint8_t)x->num; span *= (long)x->num; } }
            if (!ok || loc - first >= span) { snprintf(why, wn, "%s %ld: an array the reflection does not describe (its first location, 1-4 lengths of 1-32)", side, loc); return 0; }
            sh->io_nd[loc] = (uint8_t)dm->n;
        }
        sh->io_mask |= 1u << loc; sh->io_ct[loc] = (uint8_t)ct; sh->io_n[loc] = (uint8_t)nc; sh->io_first[loc] = (uint8_t)first;
    }
    return 1;
}
// A SPIR-V capability the device takes: core Vulkan with no feature (the spec's SPIR-V capability table), or one vk_dev.c enabled (slice 0's modules declare ImageGatherExtended, and the vertex
// ones DrawParameters, beyond core); others are a later slice
static int cap_ok(uint32_t c) {
    switch (c) {
        case 0: case 1: case 43: case 44: case 46: case 47: case 50: case 51: return 1;   // Matrix, Shader, Sampled1D, Image1D, SampledBuffer, ImageBuffer, ImageQuery, DerivativeControl
        case 52: return vd.sample_rate;     // InterpolationFunction: sampleRateShading
        case 25: return vd.gather_ext;      // ImageGatherExtended: shaderImageGatherExtended
        case 4427: return vd.draw_params;   // DrawParameters: shaderDrawParameters
        // GroupNonUniform (61) and its Vote, Arithmetic, Ballot, Shuffle, ShuffleRelative, Clustered, Quad (62-68): the operation bits the device reports, in fragment and compute
        // shaders (the builder refuses them elsewhere)
        case 61: case 62: case 63: case 64: case 65: case 66: case 67: case 68:
            return (vd.sg_stages & (VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT)) == (VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT) && (vd.sg_ops >> (c - 61) & 1);
        case 55: return vd.nofmt_r;         // StorageImageReadWithoutFormat: shaderStorageImageReadWithoutFormat
        case 56: return vd.nofmt_w;         // StorageImageWriteWithoutFormat: shaderStorageImageWriteWithoutFormat
        default: return 0;
    }
}
// A decorated resource (ids[0..n)) declared as an array (a pointer to OpTypeArray / OpTypeRuntimeArray): its id, 0 = none. The layout binds one descriptor a binding, whatever the
// reflection's count says. The words up to OpFunction are well formed (spv_patch has walked them)
static uint32_t spv_res_array(const uint32_t *w, size_t nw, const uint32_t *ids, int n) {
    uint32_t bound = w[3], hit = 0; uint8_t *t = calloc(bound ? bound : 1, 1);   // 1: an array type, 2: a pointer to one
    if (!t) die("vkn backend: SPIR-V scan allocation failed");
    for (size_t i = 5; i < nw && !hit; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (op == 54) break;
        if ((op == 28 || op == 29) && wc >= 3 && w[i + 1] < bound) t[w[i + 1]] = 1;   // OpTypeArray / OpTypeRuntimeArray
        if (op == 32 && wc >= 4 && w[i + 1] < bound && w[i + 3] < bound && t[w[i + 3]] == 1) t[w[i + 1]] = 2;   // OpTypePointer
        if (op == 59 && wc >= 4 && w[i + 1] < bound && t[w[i + 1]] == 2) for (int k = 0; k < n; k++) if (ids[k] == w[i + 2]) hit = w[i + 2];   // OpVariable
    }
    free(t);
    return hit;
}
// The resources' DescriptorSet decorations → 0 and pixel Binding decorations + 32. Every decorated resource must be one the reflection lists (else the layout would lack it), and no array; every
// capability one the device takes; one entry point, main, of the reflection's stage (its execution model). The module's size and SHA-256, and the reflection's key, chain and stage, are mod_make's.
// An Output variable decorated Index other than 0 (dual-source blending's second colour; the reflection does not tell its outputs' Index) sets sh->dual (nvn_vk.c: alpha to coverage off),
// and a location only such outputs cover is not written (out_mask: its target keeps its content, where Vulkan would leave it undefined)
static int spv_patch(uint32_t *w, size_t nw, vkn_shader *sh, char *why, size_t wn) {
    enum { MAXID = 64 }; struct { uint32_t id; size_t set, bind; } d[MAXID]; uint32_t ids[MAXID], xi[8], lc[MAXID][2], pri = 0, sec = 0; int nd = 0, ne = 0, main_ = 0, nx = 0, nl = 0; uint32_t em = ~0u;
    if (nw < 5 || w[0] != 0x07230203u) { snprintf(why, wn, "not SPIR-V (magic %#x)", nw ? w[0] : 0); return 0; }
    if (w[3] > (1u << 22)) { snprintf(why, wn, "an id bound of %u", w[3]); return 0; }
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (!wc || i + wc > nw) { snprintf(why, wn, "a malformed instruction at word %zu", i); return 0; }
        if (op == 73 || op == 74 || op == 75) { snprintf(why, wn, "decoration groups"); return 0; }   // OpDecorationGroup / OpGroupDecorate / OpGroupMemberDecorate
        if (op == 15 && wc >= 4) { const char *nm = (const char *)(const void *)&w[i + 3]; ne++; em = w[i + 1]; main_ = memchr(nm, 0, (wc - 3) * 4) && !strcmp(nm, "main"); }   // OpEntryPoint
        if (op == 17 && wc >= 2 && !cap_ok(w[i + 1])) { snprintf(why, wn, "SPIR-V capability %u (its device feature is off, or a later slice)", w[i + 1]); return 0; }   // OpCapability
        if (op == 71 && wc >= 4 && (w[i + 2] == 34 || w[i + 2] == 33)) {   // OpDecorate DescriptorSet / Binding
            int k = 0; while (k < nd && d[k].id != w[i + 1]) k++;
            if (k == nd) { if (nd == MAXID) { snprintf(why, wn, "more than %d resources", MAXID); return 0; } d[nd].id = w[i + 1]; d[nd].set = d[nd].bind = 0; nd++; }
            if (w[i + 2] == 34) d[k].set = i + 3; else d[k].bind = i + 3;
        }
        if (op == 71 && wc >= 4 && w[i + 2] == 32 && w[i + 3]) { if (nx < 8) xi[nx++] = w[i + 1]; else sh->dual = 1; }   // OpDecorate Index, not 0 (past 8 of them: taken as dual-source)
        if (op == 71 && wc >= 4 && w[i + 2] == 30 && nl < MAXID) { lc[nl][0] = w[i + 1]; lc[nl][1] = w[i + 3]; nl++; }   // OpDecorate Location
        if (op == 59 && wc >= 4 && w[i + 3] == 3) {   // OpVariable, storage class Output (the decorations come first): Index 0 (or none) or not, at its location (a built-in has none)
            int x = 0; for (int k = 0; k < nx; k++) x |= xi[k] == w[i + 2];
            sh->dual |= x;
            for (int k = 0; k < nl; k++) if (lc[k][0] == w[i + 2] && lc[k][1] < 8) { if (x) sec |= 1u << lc[k][1]; else pri |= 1u << lc[k][1]; }
        }
        if (op == 54) break;   // OpFunction: the decorations come before it
        i += wc;
    }
    if (sh->stage == 1) sh->out_mask &= ~(sec & ~pri);
    if (ne != 1 || !main_) { snprintf(why, wn, "%d entry points%s", ne, ne == 1 ? ", not main" : ""); return 0; }
    if (em != (sh->stage == 0 ? 0u : sh->stage == 5 ? 5u : 4u)) { snprintf(why, wn, "its entry point's execution model is %u, not the reflection's stage (%s)", em, sh->stage == 0 ? "vertex, 0" : "fragment, 4"); return 0; }
    for (int k = 0; k < nd; k++) {
        if (!d[k].set || !d[k].bind) { snprintf(why, wn, "resource %%%u lacks a set or binding", d[k].id); return 0; }
        ids[k] = d[k].id;
        uint32_t set = w[d[k].set], bi = w[d[k].bind]; int f = -1;
        for (int b = 0; b < sh->nb; b++) if (sh->b[b].set == set && sh->b[b].orig == bi) f = b;
        if (f >= 0 && (sh->b[f].kind == VKN_B_TEXD || sh->b[f].kind == VKN_B_SAMPD || (sh->b[f].kind == VKN_B_TEXU && sh->b[f].n > 1))) ids[k] = 0;   // arrays by design: not refused below
        if (f < 0) { snprintf(why, wn, "resource %%%u (set %u binding %u) is not in the reflection", d[k].id, set, bi); return 0; }
        w[d[k].set] = 0; w[d[k].bind] = sh->b[f].binding;
    }
    for (int b = 0; b < sh->nb; b++) { int f = 0; for (int k = 0; k < nd; k++) f |= w[d[k].bind] == sh->b[b].binding; if (!f) { snprintf(why, wn, "binding %u of the reflection is not in the module", sh->b[b].orig); return 0; } }
    uint32_t a = spv_res_array(w, nw, ids, nd);
    if (a) { snprintf(why, wn, "resource %%%u is an array (one descriptor a binding: arrays are a later slice)", a); return 0; }
    return 1;
}
// c1 (bank 1, the shader's own constants): a host-visible buffer of the bytes the reflection carries
static int c1_make(vkn_shader *sh, const char *hex, char *why, size_t wn) {
    size_t n = strlen(hex) / 2; if (!n) { snprintf(why, wn, "a c1 binding without c1 bytes"); return 0; }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (n + 15) & ~(size_t)15, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkMemoryRequirements mr; int mt; void *m;
    if (vk.vkCreateBuffer(vd.dev, &bi, NULL, &sh->c1_buf)) { sh->c1_buf = VK_NULL_HANDLE; snprintf(why, wn, "its c1 buffer cannot be made"); return 0; }
    vk.vkGetBufferMemoryRequirements(vd.dev, sh->c1_buf, &mr);
    mt = vkn_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
    if (mt < 0 || vk.vkAllocateMemory(vd.dev, &ai, NULL, &sh->c1_mem) || vk.vkBindBufferMemory(vd.dev, sh->c1_buf, sh->c1_mem, 0) || vk.vkMapMemory(vd.dev, sh->c1_mem, 0, VK_WHOLE_SIZE, 0, &m)) {
        snprintf(why, wn, "its c1 buffer's memory cannot be had"); return 0; }   // leaked with the refused module: once per module at most
    memset(m, 0, (size_t)bi.size);
    for (size_t i = 0; i < n; i++) { unsigned v = 0; sscanf(hex + 2 * i, "%2x", &v); ((uint8_t *)m)[i] = (uint8_t)v; }
    vk.vkUnmapMemory(vd.dev, sh->c1_mem);
    sh->c1_n = (uint32_t)bi.size;
    return 1;
}
// A tex_dyn_slots array indexed by a handle the shader read from memory (Paper Mario: The Thousand-Year Door's material table, storage buffer s0, names the S block's reflection
// cube by its own handle, texture ID 393): the translation's index is min(id, 31), so any ID past 31 read element 31. Each such clamp (UMin(x, 31) stored to a local, loaded,
// bitcast, then the array's access chain index) becomes: x below 32 as before (c2's slot indices, draw_drv2), else the slot k whose bound handle has that ID (texture part
// for a texture array, sampler part for a sampler array), from a 32-word push constant table (nvn_vk.c draw_idmap: slot k's handle, as element k holds it), else 31.
// Returns the new module (malloc) and its size, or NULL: nothing to change. DM_VK_IDMAP=0: not done
// A float-declared 2D texture whose reads are used only as bits (each read's components, directly or through a local, go only into OpBitcast to int / uint): the Switch returns an
// integer texture's bits through a float sampler, and Paper Mario: The Thousand-Year Door's reflection masks and lighting read the depth-stencil's stencil view so (floatBitsToInt(x) & 1);
// Vulkan has no float read of a stencil view (it read zeros: no reflection). The binding becomes a uint image (comp 1, so the host binds the stencil view), each read returns uvec4
// under a new id, and an OpBitcast to the old type takes the old id: every use is unchanged. Returns the new module (malloc), or NULL: nothing to change. DM_VK_BITS=0: not done
static int g_ida[4], g_bits_n;   // spv_idmap: modules rewritten, texture clamps, sampler clamps, clamps left as selects (a loop header or continue target); spv_bits: modules rewritten
static int spv_is_int(const uint32_t *w, size_t nw, uint32_t t) {   // t is a 32-bit int type
    for (size_t i = 5; i < nw; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (op == 54) break; if (op == 21 && w[i + 1] == t && w[i + 2] == 32) return 1; i += wc; }
    return 0;
}
static uint32_t *spv_bits(const uint32_t *w, size_t nw, vkn_shader *sh, size_t *onw, int *nfix) {
    enum { MAXV = 8, MAXU = 256 };
    size_t first_fn = 0; uint32_t bound = w[3], tu = 0, tu4 = 0, tf4 = 0, var[MAXV], vimg[MAXV]; int vb[MAXV], nv = 0;
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (op == 54) { first_fn = i; break; }
        if (op == 21 && wc == 4 && w[i + 2] == 32 && !w[i + 3]) tu = w[i + 1];
        if (op == 71 && wc >= 4 && w[i + 2] == 33 && nv < MAXV)   // Binding: a float 2D combined texture of the reflection
            for (int b = 0; b < sh->nb; b++) if (sh->b[b].binding == w[i + 3] && sh->b[b].kind == VKN_B_TEX && !sh->b[b].comp && sh->b[b].vt == VKN_VT_2D) { var[nv] = w[i + 1]; vb[nv++] = b; }
        i += wc;
    }
    if (!first_fn || !tu || !nv) return NULL;
    for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (op == 23 && wc == 4 && w[i + 3] == 4) { if (w[i + 2] == tu) tu4 = w[i + 1]; }   // OpTypeVector uint 4
        i += wc; }
    // Per variable: its pointer → sampled image → image types, and whether every read is used as bits only
    size_t vpos[MAXV] = {0}; int ok[MAXV] = {0}, nfx = 0;
    for (int v = 0; v < nv; v++) {
        uint32_t pt = 0, si = 0;
        for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (op == 59 && w[i + 2] == var[v]) { pt = w[i + 1]; vpos[v] = i; } i += wc; }
        for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (op == 32 && w[i + 1] == pt) si = w[i + 3]; i += wc; }
        for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (op == 27 && w[i + 1] == si) vimg[v] = w[i + 2]; i += wc; }
        if (!vpos[v] || !si || !vimg[v]) continue;
        // ids derived from it: loads (sampled image), OpImage, reads (Fetch 95, SampleImplicit 87, SampleExplicit 88, Gather 96)
        uint32_t L[MAXU], R[MAXU], E[MAXU], X[MAXU], Y[MAXU]; int nl = 0, nr = 0, ne = 0, nx = 0, ny = 0, bad = 0;
        for (size_t i = first_fn; i < nw && !bad; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            if (op == 61 && w[i + 3] == var[v] && nl < MAXU) L[nl++] = w[i + 2];
            if (op == 100) for (int k = 0; k < nl; k++) if (w[i + 3] == L[k] && nl < MAXU) L[nl++] = w[i + 2];
            if (op == 95 || op == 87 || op == 88 || op == 96) for (int k = 0; k < nl; k++) if (w[i + 3] == L[k]) { if (nr < MAXU) R[nr++] = w[i + 2]; tf4 = tf4 ? tf4 : w[i + 1]; if (w[i + 1] != tf4) bad = 1; }
            i += wc; }
        if (!nr || bad) continue;
        // every use of a read: OpCompositeExtract; of an extract: OpBitcast to int / uint, or OpStore into a local whose loads go only into such bitcasts
        for (size_t i = first_fn; i < nw && !bad; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            for (int k = 0; k < nr; k++) for (uint32_t a = 1; a < wc; a++) if (w[i + a] == R[k]) {
                if (op == 81 && a == 3 && ne < MAXU) E[ne++] = w[i + 2];
                else if (!((op == 95 || op == 87 || op == 88 || op == 96) && a == 2)) bad = 1; }
            i += wc; }
        for (size_t i = first_fn; i < nw && !bad; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            for (int k = 0; k < ne; k++) for (uint32_t a = 1; a < wc; a++) if (w[i + a] == E[k]) {
                if (op == 124 && a == 3 && spv_is_int(w, nw, w[i + 1])) {}
                else if (op == 62 && a == 2 && nx < MAXU) X[nx++] = w[i + 1];
                else if (!(op == 81 && a == 2)) bad = 1; }
            i += wc; }
        for (size_t i = first_fn; i < nw && !bad; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;   // the locals: stored only from these extracts; their loads
            for (int k = 0; k < nx; k++) if (op == 62 && w[i + 1] == X[k]) { int f = 0; for (int q = 0; q < ne; q++) f |= w[i + 2] == E[q]; if (!f) bad = 1; }
            for (int k = 0; k < nx; k++) if (op == 61 && w[i + 3] == X[k] && ny < MAXU) Y[ny++] = w[i + 2];
            i += wc; }
        for (size_t i = first_fn; i < nw && !bad; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            for (int k = 0; k < ny; k++) for (uint32_t a = 1; a < wc; a++) if (w[i + a] == Y[k] && !(op == 61 && a == 2) && !(op == 124 && a == 3 && spv_is_int(w, nw, w[i + 1]))) bad = 1;
            i += wc; }
        ok[v] = !bad;
    }
    int any = 0; for (int v = 0; v < nv; v++) any |= ok[v];
    if (!any || !tf4) return NULL;
    { int vec4 = 0; size_t upos = 0;   // the reads' type is a 4-vector (the uvec4 bitcasts to it), and uint is declared before the first variable rewritten
      for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (op == 23 && w[i + 1] == tf4 && w[i + 3] == 4) vec4 = 1; if (op == 21 && w[i + 1] == tu) upos = i; i += wc; }
      for (int v = 0; v < nv; v++) if (ok[v] && vpos[v] < upos) ok[v] = 0;
      if (!vec4) return NULL; }
    // The image type with a uint sampled type, its sampled image and pointer: new ids, declared before the first variable rewritten (uint and its vector before it)
    size_t ins = 0; for (int v = 0; v < nv; v++) if (ok[v] && (!ins || vpos[v] < ins)) ins = vpos[v];
    uint32_t nimg[MAXV], nsi[MAXV], nptr[MAXV], ntu4 = tu4 ? tu4 : bound++;
    size_t cap = nw + 16 + (size_t)nv * 24 + 4096; uint32_t *o = malloc(cap * 4); if (!o) return NULL;
    size_t n = 5; memcpy(o, w, 20);
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (i == ins) {
            if (!tu4) { o[n++] = 4u << 16 | 23; o[n++] = ntu4; o[n++] = tu; o[n++] = 4; }
            for (int v = 0; v < nv; v++) if (ok[v]) {   // an identical uint image type (or one made for an earlier variable) and its sampled image are reused: non-aggregate types are unique
                const uint32_t *fi = NULL; uint32_t fc = 0;
                for (size_t j = 5; j < first_fn; ) { uint32_t o2 = w[j] & 0xffff, c2 = w[j] >> 16; if (o2 == 25 && w[j + 1] == vimg[v]) { fi = w + j; fc = c2; } j += c2; }
                nimg[v] = nsi[v] = 0;
                for (int q = 0; q < v; q++) if (ok[q] && vimg[q] == vimg[v]) { nimg[v] = nimg[q]; nsi[v] = nsi[q]; }
                for (size_t j = 5; j < first_fn && !nimg[v] && fi; ) { uint32_t o2 = w[j] & 0xffff, c2 = w[j] >> 16;
                    if (o2 == 25 && c2 == fc && w[j + 2] == tu && !memcmp(w + j + 3, fi + 3, (fc - 3) * 4)) nimg[v] = w[j + 1]; j += c2; }
                for (size_t j = 5; j < first_fn && nimg[v] && !nsi[v]; ) { uint32_t o2 = w[j] & 0xffff, c2 = w[j] >> 16; if (o2 == 27 && w[j + 2] == nimg[v]) nsi[v] = w[j + 1]; j += c2; }
                if (!nimg[v]) { nimg[v] = bound++; memcpy(o + n, fi, fc * 4); o[n + 1] = nimg[v]; o[n + 2] = tu; n += fc; }
                if (!nsi[v]) { nsi[v] = bound++; o[n++] = 3u << 16 | 27; o[n++] = nsi[v]; o[n++] = nimg[v]; }
                nptr[v] = bound++; o[n++] = 4u << 16 | 32; o[n++] = nptr[v]; o[n++] = 0; o[n++] = nsi[v];   // UniformConstant
            }
        }
        if (n + wc + 8 > cap) { free(o); return NULL; }
        memcpy(o + n, w + i, wc * 4);
        for (int k = 0; k < nv; k++) if (ok[k]) {
            if (op == 59 && w[i + 2] == var[k]) o[n + 1] = nptr[k];   // the variable
            if (op == 61 && w[i + 3] == var[k]) o[n + 1] = nsi[k];    // its loads
        }
        n += wc; i += wc;
    }
    // Second walk over the output: OpImage of a rewritten load gets the uint image type; each read of one returns uvec4 under a new id, bitcast back to the old id
    uint32_t lds[64], lv[64]; int nld = 0;
    for (size_t i = 5; i < n; ) { uint32_t op = o[i] & 0xffff, wc = o[i] >> 16;
        if (op == 61) for (int k = 0; k < nv; k++) if (ok[k] && o[i + 1] == nsi[k] && nld < 64) { lds[nld] = o[i + 2]; lv[nld++] = (uint32_t)k; }
        if (op == 100) for (int k = 0; k < nld; k++) if (o[i + 3] == lds[k] && nld < 64) { o[i + 1] = nimg[lv[k]]; lds[nld] = o[i + 2]; lv[nld++] = lv[k]; break; }
        i += wc; }
    uint32_t *p = malloc((n + (size_t)nld * 8 + 64) * 4); if (!p) { free(o); return NULL; }
    size_t m = 0;
    for (size_t i = 0; i < n; ) {
        uint32_t op = i < 5 ? 0 : o[i] & 0xffff, wc = i < 5 ? 5 : o[i] >> 16;
        int hit = 0; if (op == 95 || op == 87 || op == 88 || op == 96) for (int k = 0; k < nld; k++) hit |= o[i + 3] == lds[k];
        memcpy(p + m, o + i, wc * 4);
        if (hit) { uint32_t old = o[i + 2], nu = bound++; p[m + 1] = ntu4; p[m + 2] = nu; m += wc; p[m++] = 4u << 16 | 124; p[m++] = tf4; p[m++] = old; p[m++] = nu; nfx++; }
        else m += wc;
        i += wc;
    }
    free(o);
    for (int v = 0; v < nv; v++) if (ok[v]) sh->b[vb[v]].comp = 1;
    g_bits_n++;
    p[3] = bound; *onw = m; *nfix = nfx;
    return p;
}
static uint32_t *spv_idmap(const uint32_t *w, size_t nw, const vkn_shader *sh, size_t *onw, int *nfix) {
    enum { MAXA = 16, MAXF = 64 };
    uint32_t bound = w[3], glsl = 0, tu = 0, tb = 0, arr[MAXA], c31[8]; int ak[MAXA], na = 0, nc = 0;
    size_t first_type = 0, first_fn = 0;
    for (size_t i = 5; i < nw; ) {   // globals: GLSL.std.450, uint and bool, uint 31, the TEXD / SAMPD arrays by their (patched) bindings
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (op == 11 && wc >= 3 && !strcmp((const char *)(const void *)&w[i + 2], "GLSL.std.450")) glsl = w[i + 1];
        if (op == 21 && wc == 4 && w[i + 2] == 32 && !w[i + 3]) tu = w[i + 1];
        if (op == 20 && wc == 2) tb = w[i + 1];
        if (op == 43 && wc == 4 && w[i + 1] == tu && tu && w[i + 3] == 31 && nc < 8) c31[nc++] = w[i + 2];
        if (op == 71 && wc >= 4 && w[i + 2] == 33 && na < MAXA)
            for (int b = 0; b < sh->nb; b++) if (sh->b[b].binding == w[i + 3] && (sh->b[b].kind == VKN_B_TEXD || sh->b[b].kind == VKN_B_SAMPD)) { arr[na] = w[i + 1]; ak[na++] = sh->b[b].kind == VKN_B_TEXD; }
        if (!first_type && ((op >= 19 && op <= 39) || op == 43)) first_type = i;
        if (op == 54) { first_fn = i; break; }
        i += wc;
    }
    if (!glsl || !tu || !nc || !na || !first_type || !first_fn) return NULL;
    // The clamps: access chain on an array → its index's bitcast source → the local it is loaded from → the UMin(x, 31) stored there
    size_t fix[MAXF]; int fk[MAXF], nf = 0;
    for (size_t i = first_fn; i < nw; ) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        int a = -1; if (op == 65 && wc >= 5) for (int k = 0; k < na; k++) if (w[i + 3] == arr[k]) a = k;
        if (a >= 0) {
            uint32_t ix = w[i + 4], var = 0;
            for (size_t j = first_fn; j < i; ) { uint32_t o2 = w[j] & 0xffff, c2 = w[j] >> 16; if (o2 == 124 && c2 == 4 && w[j + 2] == ix) ix = w[j + 3]; j += c2; }   // OpBitcast
            for (size_t j = first_fn; j < i; ) { uint32_t o2 = w[j] & 0xffff, c2 = w[j] >> 16; if (o2 == 61 && c2 >= 4 && w[j + 2] == ix) var = w[j + 3]; j += c2; }   // OpLoad
            for (size_t j = first_fn; j < i; ) {
                uint32_t o2 = w[j] & 0xffff, c2 = w[j] >> 16;
                uint32_t val = o2 == 62 && c2 >= 3 && w[j + 1] == var ? w[j + 2] : o2 == 12 && c2 == 7 && w[j + 2] == ix ? ix : 0;   // OpStore into the local, or the index itself
                if (val) for (size_t m = first_fn; m < i; ) { uint32_t o3 = w[m] & 0xffff, c3 = w[m] >> 16;
                    if (o3 == 12 && c3 == 7 && w[m + 2] == val && w[m + 3] == glsl && w[m + 4] == 38) {   // OpExtInst UMin
                        int is31 = 0; for (int q = 0; q < nc; q++) is31 |= w[m + 6] == c31[q];
                        int dup = 0; for (int q = 0; q < nf; q++) dup |= fix[q] == m;
                        if (is31 && !dup && nf < MAXF) { fix[nf] = m; fk[nf++] = ak[a]; } }
                    m += c3; }
                j += c2;
            }
        }
        i += wc;
    }
    if (!nf) return NULL;
    // A clamp's block becomes a selection: the search runs only for x of 32 or more (most pixels take c2's slot indices). Not where the block holds an OpLoopMerge or is a
    // loop's continue target (the header must stay one block): there the search is unconditional selects
    uint32_t blab[MAXF]; int nobr[MAXF];
    for (int q = 0; q < nf; q++) {
        uint32_t lab = 0; size_t e = nw;
        for (size_t i = first_fn; i < fix[q]; ) { if ((w[i] & 0xffff) == 248) lab = w[i + 1]; i += w[i] >> 16; }
        nobr[q] = !lab;
        for (size_t i = fix[q]; i < nw && e == nw; ) { uint32_t o2 = w[i] & 0xffff; if (o2 == 248 || o2 == 56) e = i; else if (o2 == 246) nobr[q] = 1; i += w[i] >> 16; }
        for (size_t i = first_fn; i < nw; ) { if ((w[i] & 0xffff) == 246 && (w[i] >> 16) >= 4 && w[i + 2] == lab) nobr[q] = 1; i += w[i] >> 16; }
        blab[q] = lab;
    }
    uint32_t rmap[MAXF][2]; int nr = 0; size_t myphi[MAXF]; int nmp = 0;
    // New ids: uint constants 0-31, 32, 0xfffff, 20; bool if absent; the push constant block uint[32] and its pointers
    uint32_t ck[32], c32, cmask, c20, tbool = tb ? tb : bound++, tarr, tst, tpst, tpu, vpc;
    for (int k = 0; k < 32; k++) ck[k] = bound++;
    c32 = bound++; cmask = bound++; c20 = bound++; tarr = bound++; tst = bound++; tpst = bound++; tpu = bound++; vpc = bound++;
    size_t per = 7 + 32 * 26 + 5 + 6 + 3 + 4 + 2 + 2 + 2 + 7, cap = nw + 16 + 4 * 36 + 3 + 4 + 3 + 4 + 4 + 4 + (size_t)nf * per;
    uint32_t *o = malloc(cap * 4); if (!o) return NULL;
    size_t n = 0;
#define EMIT(...) do { uint32_t _t[] = {__VA_ARGS__}; memcpy(o + n, _t, sizeof _t); n += sizeof _t / 4; } while (0)
    for (size_t i = 0; i < nw; ) {
        uint32_t wc = i < 5 ? 5 : w[i] >> 16;
        if (i == first_type) {   // the decorations end here: the block's
            EMIT(4u << 16 | 71, tarr, 6, 4); EMIT(3u << 16 | 71, tst, 2); EMIT(5u << 16 | 72, tst, 0, 35, 0);   // ArrayStride 4, Block, member 0 Offset 0
        }
        if (i == first_fn) {     // the types and constants end here
            if (!tb) EMIT(2u << 16 | 20, tbool);
            for (int k = 0; k < 32; k++) EMIT(4u << 16 | 43, tu, ck[k], (uint32_t)k);
            EMIT(4u << 16 | 43, tu, c32, 32); EMIT(4u << 16 | 43, tu, cmask, 0xfffff); EMIT(4u << 16 | 43, tu, c20, 20);
            EMIT(4u << 16 | 28, tarr, tu, c32); EMIT(3u << 16 | 30, tst, tarr);
            EMIT(4u << 16 | 32, tpst, 9, tst); EMIT(4u << 16 | 32, tpu, 9, tu); EMIT(4u << 16 | 59, tpst, vpc, 9);
        }
        int f = -1; for (int q = 0; q < nf; q++) if (fix[q] == i) f = q;
        if (i == 0) { memcpy(o, w, 20); n = 5; }
        else if (f < 0) { memcpy(o + n, w + i, wc * 4); n += wc; }
        else {   // R = x < 32 ? x : the slot whose handle part is x (the lowest), else UMin(x, 31)
            uint32_t res = w[i + 2], x = w[i + 5], prev = bound++, sl = 0, ml = 0;
            EMIT(7u << 16 | 12, tu, prev, glsl, 38, x, w[i + 6]);
            if (!nobr[f]) {   // header: x < 32 goes straight to the merge block
                uint32_t lt = bound++; sl = bound++; ml = bound++;
                EMIT(5u << 16 | 176, tbool, lt, x, c32); EMIT(3u << 16 | 247, ml, 0); EMIT(4u << 16 | 250, lt, ml, sl); EMIT(2u << 16 | 248, sl);
            }
            for (int k = 31; k >= 0; k--) {
                uint32_t p = bound++, e = bound++, pt = bound++, q = bound++, s = bound++;
                EMIT(6u << 16 | 65, tpu, p, vpc, ck[0], ck[k]); EMIT(4u << 16 | 61, tu, e, p);
                if (fk[f]) EMIT(5u << 16 | 199, tu, pt, e, cmask); else EMIT(5u << 16 | 194, tu, pt, e, c20);
                EMIT(5u << 16 | 170, tbool, q, pt, x); EMIT(6u << 16 | 169, tu, s, q, ck[k], prev); prev = s;
            }
            if (!nobr[f]) {   // the merge block takes the rest of the original block: its phis' edges from that block now come from here
                uint32_t hl = blab[f]; for (int r = 0; r < nr; r++) if (rmap[r][0] == blab[f]) hl = rmap[r][1];
                EMIT(2u << 16 | 249, ml); EMIT(2u << 16 | 248, ml); myphi[nmp++] = n; EMIT(7u << 16 | 245, tu, res, x, hl, prev, sl);
                int r = 0; while (r < nr && rmap[r][0] != blab[f]) r++;
                if (r == nr && nr < MAXF) { rmap[r][0] = blab[f]; nr++; }
                if (r < MAXF) rmap[r][1] = ml;
            } else { uint32_t lt = bound++; EMIT(5u << 16 | 176, tbool, lt, x, c32); EMIT(6u << 16 | 169, tu, res, lt, x, prev); }
        }
        i += wc;
    }
#undef EMIT
    for (size_t i = 5; i < n && nr; ) {   // the original phis: an edge from a split block comes from its last merge block
        uint32_t op = o[i] & 0xffff, wc = o[i] >> 16;
        int mine = 0; for (int q = 0; q < nmp; q++) mine |= myphi[q] == i;
        if (op == 245 && !mine) for (uint32_t a = 4; a + 1 <= wc - 1; a += 2) for (int r = 0; r < nr; r++) if (o[i + a] == rmap[r][0]) { o[i + a] = rmap[r][1]; break; }
        i += wc;
    }
    g_ida[0]++; for (int q = 0; q < nf; q++) { g_ida[fk[q] ? 1 : 2]++; g_ida[3] += nobr[q]; }
    o[3] = bound; *onw = n; *nfix = nf;
    return o;
}
static uint32_t g_shid;
// The SPIR-V's SHA-256 against the reflection's spv_sha256: its first 8 bytes, 16 lowercase hex digits (dmsh_build_vk.py: of the decompressed module)
static int sha_ok(const void *spv, size_t n, const char *want) {
    uint8_t d[32]; char hx[17];
    if (strlen(want) != 16) return 0;
    dm_sha256(spv, n, d); for (int i = 0; i < 8; i++) snprintf(hx + 2 * i, 3, "%02x", d[i]);
    return !strcmp(hx, want);
}
// DM_VK_SPV_OVR=<dir> (diagnostics): <dir>/<key>.spv, where there is one, replaces the module once its own passed the reflection's checks, before the patches (an experiment on the
// SPIR-V, e.g. spirv-opt --relax-float-ops, without rebuilding the cache). *ovr: the buffer read, freed by the caller
static void spv_ovr(const char *hk, uint32_t **spv, size_t *nbytes, uint32_t **ovr) {
    static const char *d = (const char *)1; if (d == (const char *)1) d = getenv("DM_VK_SPV_OVR");
    if (!d || !*d) return;
    char fp[1024]; snprintf(fp, sizeof fp, "%s/%s.spv", d, hk);
    FILE *f = fopen(fp, "rb"); if (!f) return;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *b = n > 20 && n % 4 == 0 ? malloc((size_t)n) : NULL;
    if (b && fread(b, 1, (size_t)n, f) == (size_t)n && b[0] == 0x07230203u) { *spv = *ovr = b; *nbytes = (size_t)n; static int nl; if (nl++ < 4) VLOG("shader %s: SPIR-V from %s (DM_VK_SPV_OVR)", hk, fp); }
    else free(b);
    fclose(f);
}
// A module from its reflection text and SPIR-V (words, patched in place). NULL = refused (said once, with why)
static vkn_shader *mod_make(const uint8_t key[8], int stage, const char *json, uint32_t *spv, size_t nbytes, int zst) {
    char hk[17], why[256] = ""; for (int i = 0; i < 8; i++) snprintf(hk + 2 * i, 3, "%02x", key[i]);
    vkn_shader *sh = calloc(1, sizeof *sh); jv j; memset(&j, 0, sizeof j); uint32_t *ovr = NULL;
    if (!sh) die("vkn backend: shader entry allocation failed");
    memcpy(sh->key, hk, 17); sh->stage = stage;
    char want[24]; snprintf(want, sizeof want, "%s_b", hk);
    const char *e = json ? j_val(json, &j, 0) : NULL;
    if (!e || j.t != J_OBJ) snprintf(why, sizeof why, "its reflection cannot be read");
    else if (strcmp(j_s(&j, "status"), "ok")) snprintf(why, sizeof why, "status %s (%s)", j_s(&j, "status"), j_get(&j, "unsupported") && j_get(&j, "unsupported")->n ? "unsupported features" : "see the build");
    else if (strcmp(j_s(&j, "chain"), VKN_CHAIN)) snprintf(why, sizeof why, "chain %s, this host takes %s", j_s(&j, "chain"), VKN_CHAIN);
    else if (strcmp(j_s(&j, "key"), want) || strcmp(j_s(&j, "dm_ctl"), hk)) snprintf(why, sizeof why, "its reflection is of %s", j_s(&j, "key"));
    else if (strcmp(j_s(&j, "stage"), stage == 0 ? "vs" : stage == 1 ? "ps" : stage == 5 ? "cs" : "?")) snprintf(why, sizeof why, "stage %s, the map says %d", j_s(&j, "stage"), stage);
    else if (strcmp(j_s(&j, "entry"), "main")) snprintf(why, sizeof why, "entry point %s", j_s(&j, "entry"));
    else if (!spv) snprintf(why, sizeof why, "no SPIR-V (%s_b.spv%s)", hk, zst ? ": only its .spv.zst is there, and the directory takes the modules decompressed" : "");
    else if ((long)nbytes != j_i(&j, "spv_bytes", -1) || nbytes % 4) snprintf(why, sizeof why, "SPIR-V of %zu bytes, the reflection says %ld", nbytes, j_i(&j, "spv_bytes", -1));
    else if (!sha_ok(spv, nbytes, j_s(&j, "spv_sha256"))) snprintf(why, sizeof why, "its SPIR-V's SHA-256 is not the reflection's spv_sha256 (%.20s)", *j_s(&j, "spv_sha256") ? j_s(&j, "spv_sha256") : "none");   // before the patch rewrites it
    else if (spv_ovr(hk, &spv, &nbytes, &ovr), refl_take(sh, &j, why, sizeof why) && spv_patch(spv, nbytes / 4, sh, why, sizeof why)) {
        for (int b = 0; b < sh->nb && !*why; b++) if (sh->b[b].kind == VKN_B_C1 && !sh->c1_buf) c1_make(sh, j_s(&j, "c1"), why, sizeof why);
        if (!*why) {
            static int idm = -1; if (idm < 0) { const char *x = getenv("DM_VK_IDMAP"); idm = !(x && *x == '0'); }
            size_t mn = nbytes / 4; uint32_t *ms = spv; int nfix = 0;
            static int bits = -1; if (bits < 0) { const char *x = getenv("DM_VK_BITS"); bits = !(x && *x == '0'); }
            if (bits && (stage == 1 || stage == 5)) { int nb2 = 0; uint32_t *p = spv_bits(spv, nbytes / 4, sh, &mn, &nb2); if (p) { ms = p; sh->bits = nb2; static int nl; if (nl++ < 4) VLOG("shader %s: %d texture reads used only as bits read a uint image (a stencil view: first cases)", hk, nb2); } else mn = nbytes / 4; }
            if (idm && sh->dyn && (stage == 1 || stage == 5)) { size_t in = mn; uint32_t *p = spv_idmap(ms, in, sh, &mn, &nfix); if (p) { if (ms != spv) free(ms); ms = p; sh->idmap = nfix; } else mn = in; }
            { const char *dd = getenv("DM_VK_IDMAP_DUMP"); if (dd && *dd && ms != spv) { char fp[1024]; snprintf(fp, sizeof fp, "%s/%s.spv", dd, hk); FILE *f = fopen(fp, "wb"); if (f) { fwrite(ms, 4, mn, f); fclose(f); } } }   // DM_VK_IDMAP_DUMP=<dir>: the rewritten modules, for spirv-val
            VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, mn * 4, ms};
            VkResult r = vk.vkCreateShaderModule(vd.dev, &ci, NULL, &sh->mod);
            if (r) snprintf(why, sizeof why, "vkCreateShaderModule: %d", r);
            if (sh->idmap) { static int nl; if (nl++ < 4) VLOG("shader %s: %d array index clamps look up the bound slots' handles (a handle read from memory; first cases)", hk, nfix); }
            if (ms != spv) free(ms);
        }
    }
    j_free(&j); if (ovr) free(ovr);
    if (*why) { VLOG("[RED] shader %s (%s) refused: %s: its draws are skipped", hk, stage == 0 ? "vertex" : stage == 1 ? "pixel" : stage == 5 ? "compute" : "other", why); atomic_fetch_add_explicit(&g_nrefuse, 1, memory_order_relaxed); free(sh); return NULL; }
    sh->id = ++g_shid; atomic_fetch_add_explicit(&g_nload, 1, memory_order_relaxed);
    return sh;
}
static void idmap_all(void);
vkn_shader *vkn_shader_get(const uint8_t key[8], int stage) {
    shaders_init(); idmap_all();
    void *v = nvn_hm_get(&g_mods, key);
    if (v) return v == REFUSED ? NULL : v;
    vkn_shader *sh = NULL;
    // compute (stage 5) too; DM_VK_COMPUTE=0 refuses it as before (nvn_vk.c op_dispatch)
    if (stage != 0 && stage != 1 && !(stage == 5 && !(getenv("DM_VK_COMPUTE") && *getenv("DM_VK_COMPUTE") == '0'))) { char hk[17]; for (int i = 0; i < 8; i++) snprintf(hk + 2 * i, 3, "%02x", key[i]); VLOG("[RED] shader %s: NVN stage %d (geometry, tessellation and compute are a later slice): its draws are skipped", hk, stage); }
    else if (g_dir_ok) {
        char hk[17], p[1024]; for (int i = 0; i < 8; i++) snprintf(hk + 2 * i, 3, "%02x", key[i]);
        snprintf(p, sizeof p, "%s/%s_b.json", g_dir, hk); char *json = slurp(p, NULL); size_t n = 0;
        snprintf(p, sizeof p, "%s/%s_b.spv", g_dir, hk); char *spv = slurp(p, &n);
        if (!json) VLOG("[RED] shader %s: %s/%s_b.json cannot be read: its draws are skipped", hk, g_dir, hk);
        else { int zst = 0; if (!spv) { snprintf(p, sizeof p, "%s/%s_b.spv.zst", g_dir, hk); zst = !access(p, R_OK); }
               sh = mod_make(key, stage, json, (uint32_t *)(void *)spv, spv ? n : 0, zst); }
        free(json); free(spv);
    }
    nvn_hm_put(&g_mods, key, sh ? (void *)sh : REFUSED);
    return sh;
}
// DM_VK_IDMAP_ALL=1 (diagnostics): at the first shader load, every pixel and compute shader of the folder with a tex_dyn_slots array is loaded too, so spv_idmap rewrites all of them
// (with DM_VK_IDMAP_DUMP, all are written for spirv-val), and the totals are said
static void idmap_all(void) {
    static int done; const char *e = getenv("DM_VK_IDMAP_ALL");
    if (done || !e || *e != '1' || !g_dir_ok) return;
    done = 1; DIR *d = opendir(g_dir); struct dirent *de; int n = 0;
    while (d && (de = readdir(d))) {
        size_t l = strlen(de->d_name); if (l != 23 || strcmp(de->d_name + 16, "_b.json")) continue;
        char p[1024]; snprintf(p, sizeof p, "%s/%s", g_dir, de->d_name); char *j = slurp(p, NULL);
        int st = j && (strstr(j, "_cb0_40000000") || strstr(j, "\"comp\": \"float\"")) ? (strstr(j, "\"stage\": \"ps\"") ? 1 : strstr(j, "\"stage\": \"cs\"") ? 5 : -1) : -1; free(j);
        if (st < 0) continue;
        uint8_t k[8]; for (int i = 0; i < 8; i++) { unsigned v; sscanf(de->d_name + 2 * i, "%2x", &v); k[i] = (uint8_t)v; }
        vkn_shader_get(k, st); n++;
    }
    if (d) closedir(d);
    VLOG("DM_VK_IDMAP_ALL: %d pixel and compute shaders with tex_dyn_slots arrays or float textures loaded (bit reads rewritten in %d): %d rewritten, %d texture and %d sampler clamps, %d of them as selects", n, g_bits_n, g_ida[0], g_ida[1], g_ida[2], g_ida[3]);
}
// The tests' modules (t_vkn.c): a map entry and a module from memory, as if read from the directory
vkn_shader *vkn_shader_add(const uint8_t key[8], int stage, const char *json, const uint32_t *spv, size_t nbytes) {
    shaders_init();
    nvn_hm_put(&g_map, key, (void *)(uintptr_t)(stage + 1));
    uint32_t *w = malloc(nbytes ? nbytes : 4); if (!w) die("vkn backend: SPIR-V copy allocation failed");
    memcpy(w, spv, nbytes);
    vkn_shader *sh = mod_make(key, stage, json, w, nbytes, 0); free(w);
    nvn_hm_put(&g_mods, key, sh ? (void *)sh : REFUSED);
    return sh;
}
