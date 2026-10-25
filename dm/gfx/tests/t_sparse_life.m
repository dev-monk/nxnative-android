// A texture outliving its sparse virtual pool (SHIN MEGAMI TENSEI V finalizes its per-texture pools all the time): it is read only through the pool it was initialized in, found from
// the texture alone (tex_t sparse, hle_nvn.c vpool_tex), so it keeps its last upload once that pool is gone, also when a new pool takes the same pool object or the same addresses
// before its next use; a first upload after the pool is gone reads 0, a re-upload that finds it gone leaves the texture as it was, and nothing reads the unmapped range directly.
// A copy of the texture object still finds its pool after the object is finalized, a texture past its pool's reservation is not taken for the pool reserved there, and the level a
// texture is bound from comes from the mappings its bytes were gathered with
#define nvn_game_fps nvn_game_fps_mtl_default
#include "../nvn/nvn_mtl.m"
#undef nvn_game_fps
#define g_ft g_ft_hle
#include "../nvn/hle_nvn.c"
#undef g_ft
uint64_t g_t0_ns;
void *front_metal_layer(void) { return NULL; }
module_t *module_of(const void *p) { (void)p; return NULL; }
module_t g_mod[MAX_MOD]; int g_nmod;
void svc_thread_dump(void) {}
uint64_t now_ns(void) { return bp_now(); }
static int checks, failed;
#define CHECK(c, label) do { checks++; if (!(c)) { failed++; fprintf(stderr, "FAIL: %s\n", label); } } while (0)

enum { PAGE = 0x10000, W = 16, H = 16 };
static void finish(void) {
    CHECK(mtl_wait(mtl_sub_commit(), 5000000000ull), "GPU submission completed");
    CHECK(!mtl_last_commit_error(), "GPU submission has no error");
}
// Every byte of level 0 is v (the pages are filled with one byte, so the block-linear layout does not matter)
static int all(id<MTLTexture> mt, uint8_t v) {
    if (!mt) return 0;
    id<MTLBuffer> b = [g_dev newBufferWithLength:W * H * 4 options:MTLResourceStorageModeShared];
    mtl_sub_begin();
    [blit() copyFromTexture:mt sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:b destinationOffset:0 destinationBytesPerRow:W * 4 destinationBytesPerImage:W * H * 4];
    finish();
    const uint8_t *p = b.contents; for (int i = 0; i < W * H * 4; i++) if (p[i] != v) return 0;
    return 1;
}
static id<MTLTexture> resolve(const tex_t *t) { mtl_sub_begin(); id<MTLTexture> r = tex_mtl(t); finish(); return r; }
static void map_at(pool_t *s, pool_t *phys, int64_t poff, int64_t voff) {
    map_req m = {.phys = (NVNmemoryPool *)phys, .poff = poff, .voff = voff, .size = PAGE};
    CHECK(h_nvnMemoryPoolMapVirtual((NVNmemoryPool *)s, 1, (const NVNmappingRequest *)&m), "a page mapped");
}
static void map_page(pool_t *s, pool_t *phys, int64_t poff) { map_at(s, phys, poff, 0); }
static void bare_pool(pool_t *s, uint64_t size) {   // sparse, nothing mapped
    pool_builder_t pb = {.size = size, .flags = 0x200};
    CHECK(h_nvnMemoryPoolInitialize((NVNmemoryPool *)s, (const NVNmemoryPoolBuilder *)&pb), "sparse pool initialized");
}
static void sparse_pool(pool_t *s, pool_t *phys, int64_t poff) {
    pool_builder_t pb = {.size = DM_GAME_APP_POOL + 0x100000, .flags = 0x200};   // larger than the application pool: sparse with any profile
    CHECK(h_nvnMemoryPoolInitialize((NVNmemoryPool *)s, (const NVNmemoryPoolBuilder *)&pb), "sparse pool initialized");
    map_page(s, phys, poff);
}
static void texture_lv(tex_t *t, pool_t *s, int flags, int64_t off, int w, int h, int levels) {
    tex_builder_t tb; h_nvnTextureBuilderSetDefaults((NVNtextureBuilder *)&tb);
    h_nvnTextureBuilderSetTarget((NVNtextureBuilder *)&tb, NVN_TEXTURE_TARGET_2D); h_nvnTextureBuilderSetFormat((NVNtextureBuilder *)&tb, NVN_FORMAT_RGBA8);
    h_nvnTextureBuilderSetSize2D((NVNtextureBuilder *)&tb, w, h); h_nvnTextureBuilderSetFlags((NVNtextureBuilder *)&tb, flags);
    if (levels > 1) h_nvnTextureBuilderSetLevels((NVNtextureBuilder *)&tb, levels);
    h_nvnTextureBuilderSetStorage((NVNtextureBuilder *)&tb, (NVNmemoryPool *)s, off);
    CHECK(h_nvnTextureInitialize((NVNtexture *)t, (const NVNtextureBuilder *)&tb), "texture initialized");
    CHECK(t->sparse, "a texture in a sparse pool is marked sparse");
}
static void texture(tex_t *t, pool_t *s, int flags) { texture_lv(t, s, flags, 0, W, H, 1); }
static void fini(pool_t *s) { h_nvnMemoryPoolFinalize((NVNmemoryPool *)s); }
int main(void) { @autoreleasepool {
    mtl_init();
    uint8_t *mem = mmap(NULL, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0), *other = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    CHECK(mem != MAP_FAILED && other != MAP_FAILED, "physical storage allocated"); if (mem == MAP_FAILED || other == MAP_FAILED) abort();
    memset(mem, 0xa1, PAGE); memset(mem + PAGE, 0xb2, PAGE); memset(other, 0xc3, PAGE);
    pool_t phys, s; tex_t a, a2, c, d, e, f, g;
    pool_builder_t pb = {.mem = mem, .size = 2 * PAGE, .flags = 0x24};
    CHECK(h_nvnMemoryPoolInitialize((NVNmemoryPool *)&phys, (const NVNmemoryPoolBuilder *)&pb), "physical pool initialized");

    // Pool gone, then the texture used, then a pool made in the same object: the old texture keeps what it was uploaded with
    sparse_pool(&s, &phys, 0); texture(&a, &s, 0);
    id<MTLTexture> a1 = resolve(&a); CHECK(all(a1, 0xa1), "first upload reads the mapped page");
    fini(&s);
    CHECK(resolve(&a) == a1, "texture of a finalized pool keeps its upload");
    sparse_pool(&s, &phys, PAGE);
    CHECK(resolve(&a) == a1 && all(a1, 0xa1), "a pool made in the same object later does not replace the old texture's upload");
    fini(&s);

    // A pool made in the same object before the old texture is used again, remapped since (its mapping generation moved): the texture is not read through the new pool
    sparse_pool(&s, &phys, 0); texture(&a2, &s, 0);
    uint8_t *old_mem = a2.mem;
    id<MTLTexture> b1 = resolve(&a2); CHECK(all(b1, 0xa1), "first upload reads the mapped page (2)");
    fini(&s); sparse_pool(&s, &phys, PAGE); map_page(&s, &phys, PAGE);
    CHECK(s.mem == old_mem, "the new pool takes the old one's addresses (the case tested)");
    CHECK(resolve(&a2) == b1 && all(b1, 0xa1), "a pool made in the same object before the next use is not read for the old texture");

    // A texture first used after its pool is gone and a new pool took the pool object reads 0, not the new pool's pages
    texture(&e, &s, 0);
    fini(&s); sparse_pool(&s, &phys, PAGE);
    id<MTLTexture> e1 = resolve(&e);
    CHECK(all(e1, 0), "a first use after the pool was replaced reads 0, not the new pool");
    CHECK(resolve(&e) == e1, "and is not uploaded again at each use");

    // A texture in a live pool reads it; after the pool is gone an upload reads 0 (the range is unmapped: read directly it would fault), a re-upload is refused
    texture(&c, &s, 0); texture(&d, &s, 0); texture(&f, &s, 0);
    id<MTLTexture> c1 = resolve(&c); CHECK(all(c1, 0xb2), "texture of the second pool reads its page");
    id<MTLTexture> f1 = resolve(&f);
    map_page(&s, &phys, 0);   // f's pages remapped: dirty
    fini(&s);
    uint64_t st;
    mtl_sub_begin(); id<MTLTexture> c0 = pkg_upload(&c, &st, nil, 0, NULL); finish();
    CHECK(all(c0, 0), "a first upload after the pool is gone reads 0");
    mtl_sub_begin(); id<MTLTexture> fr = pkg_upload(&f, &st, nil, 1, NULL); finish();
    CHECK(fr == nil, "a re-upload that finds the pool gone is refused");
    CHECK(resolve(&f) == f1 && all(f1, 0xb2), "a dirty texture whose pool went keeps its upload");
    CHECK(resolve(&c) == c1, "the texture keeps its last upload");
    id<MTLTexture> d1 = resolve(&d);
    CHECK(all(d1, 0), "a first use after the pool is gone uploads 0");
    CHECK(resolve(&d) == d1, "and is not uploaded again at each use");

    // An image texture (not read through the page table) whose pool object became a non-virtual pool: its first use does not read its old, unmapped storage (tex_init_fill),
    // which faults on any read here (a PROT_NONE page at the old address: unmapped, the range may be taken by anything), and it starts as 0
    sparse_pool(&s, &phys, 0); texture(&g, &s, NVN_TEXTURE_FLAGS_IMAGE);
    fini(&s);
    pool_builder_t nb = {.mem = other, .size = PAGE, .flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | 0x20};
    CHECK(h_nvnMemoryPoolInitialize((NVNmemoryPool *)&s, (const NVNmemoryPoolBuilder *)&nb), "the same pool object initialized as a non-virtual pool");
    uint8_t *stale = mmap(g.mem, PAGE, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);   // a hint, not MAP_FIXED: whatever took the range since is left alone
    CHECK(stale == g.mem, "the old range is free to map again");
    CHECK(all(resolve(&g), 0), "an image texture of a finalized sparse pool starts as 0, without reading its old range");

    h_nvnTextureFinalize((NVNtexture *)&a); h_nvnTextureFinalize((NVNtexture *)&a2); h_nvnTextureFinalize((NVNtexture *)&c); h_nvnTextureFinalize((NVNtexture *)&d);
    h_nvnTextureFinalize((NVNtexture *)&e); h_nvnTextureFinalize((NVNtexture *)&f); h_nvnTextureFinalize((NVNtexture *)&g);

    // A copy of a texture object first used after the object was finalized, its pool still live (the game copies texture objects by value; the replay resolves a registered copy)
    pool_t s2; tex_t h; sparse_pool(&s2, &phys, 0); texture(&h, &s2, 0);
    tex_t hc = h; h_nvnTextureFinalize((NVNtexture *)&h);
    CHECK(all(resolve(&hc), 0xa1), "a copy first used after its object was finalized reads the live pool");
    fini(&s2);

    // A texture past its pool's reservation (the page table reaches further: iOS reserves 1 GB, the Mac 64 GB), where a later reservation can land: it reads its own pool.
    // The other pool (near) is put in the host's list by hand, at an address inside the first one's range
    pool_t big; tex_t p; const uint64_t G = 1ull << 30; const int64_t poff = 68 * G;
    bare_pool(&big, 72 * G); map_at(&big, &phys, 0, poff); texture_lv(&p, &big, 0, poff, W, H, 1);
    struct vpool *near = calloc(1, sizeof *near);
    near->base = big.mem + 66 * G; near->len = near->size = 4 * G; near->pt = calloc((near->size + (2u << 20) - 1) >> 21, sizeof *near->pt);
    pthread_mutex_lock(&g_vp_mu);
    int put = big.vp && big.vp->len < big.vp->size && g_nvp < g_capvp;
    if (put) { int at = vps_find((uint64_t)(uintptr_t)near->base) + 1; memmove(&g_vps[at + 1], &g_vps[at], (size_t)(g_nvp - at) * sizeof *g_vps); g_vps[at] = near; g_nvp++; }
    pthread_mutex_unlock(&g_vp_mu);
    CHECK(put, "the pool reaches past its reservation, and the near pool is listed there (the case tested)");
    CHECK(all(resolve(&p), 0xa1), "a texture past its pool's reservation reads its own pool, not the pool reserved there");
    pthread_mutex_lock(&g_vp_mu);
    int at = vps_find((uint64_t)(uintptr_t)near->base), gone = put && at >= 0 && g_vps[at] == near;
    if (gone) { memmove(&g_vps[at], &g_vps[at + 1], (size_t)(g_nvp - 1 - at) * sizeof *g_vps); g_nvp--; }
    pthread_mutex_unlock(&g_vp_mu);
    CHECK(!put || gone, "the other pool taken out of the list again");
    free(near->pt); free(near); h_nvnTextureFinalize((NVNtexture *)&p); fini(&big);

    // A texture whose level 0 is unmapped is bound from level 1, and that level comes from the mappings its bytes were gathered with: a pool gone right after does not change it
    pool_t s3; tex_t m3; bare_pool(&s3, DM_GAME_APP_POOL + 0x100000); map_at(&s3, &phys, PAGE, PAGE);
    texture_lv(&m3, &s3, 0, 0, 128, 128, 3);
    CHECK(nvn_tex_level_offset(&m3.d, 1) == PAGE && m3.L.size <= 2 * PAGE, "level 1 starts on the second page (the case tested)");
    id<MTLTexture> mv = resolve(&m3);
    CHECK(mv && mv.parentTexture && mv.parentRelativeLevel == 1, "bound from level 1, the first one mapped");
    unsigned vb = 9; mtl_sub_begin(); id<MTLTexture> mu = pkg_upload(&m3, &st, nil, 1, &vb); finish();
    fini(&s3);
    NvPkg *pk = [NvPkg new]; pk->tex = mu; pk->sparse = 1; pk->vbase = -1; sparse_view(pk, &m3, vb);
    CHECK(mu && vb == 1 && pk->vbase == 1 && pk->view.parentRelativeLevel == 1, "the level is the one gathered with the bytes, not read from the pool after it went");
    pk = nil; h_nvnTextureFinalize((NVNtexture *)&m3);
    fini(&s); fini(&phys); defer_drain(); munmap(mem, 2 * PAGE); munmap(other, PAGE); if (stale != MAP_FAILED) munmap(stale, PAGE);
    printf("t_sparse_life: %d checks, %d failed\n", checks, failed); return failed != 0;
} }
