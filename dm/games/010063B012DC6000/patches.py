# SHIN MEGAMI TENSEI V (010063B012DC6000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of a module's NSO build ID;
# CUTS, OBS and FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
CUTS = {
  # SHIN MEGAMI TENSEI V 1.0.0 (v0) main, Unreal Engine 4.23 (++UE4+Release-4.23):
  # - Its FMallocBinned3 takes 37 GB of the engine's address region (74 pools of 512 MB, pool index = offset >> 29; the alias region was mapped up to +38.5 GB at boot with a 64 GB one), which an
  #   iPhone cannot give. The allocator is made as DRAGON QUEST VII Reimagined's: GCreateMalloc mallocs 0x1c38 bytes on the first allocation, constructs them (main+0x1713920 → main+0x17098b0)
  #   and stores GMalloc (main+0x6abc8b8). The constructor call is cut, and observation gate OBS_UE_MALLOC (OBS below) puts the host allocator's vtable into the new object, which is still in
  #   x19 (dm/host/hle_uemalloc.c). Nothing outside the allocator reads its pool base (main+0x6a8f568: every reference is in main+0x170a0ec … +0x170e504) or calls its methods directly (they
  #   are reached only through the vtable main+0x57e6238); the one call into its code from outside (main+0x1872d70 → +0x170aa50) builds a bit tree in the caller's own object
  # - nninitStartup (main+0x1885750) allocates the engine's region only when -SwitchVirtualAddressSpaceSize= gave no size: `if (!size) { size = 62 GB; AllocateAddressRegion(&base, 62 GB); }`,
  #   so with the option the base stayed 0 and the engine's page allocator (main+0x1872af0) would hand out [0, size). The branch now skips only the default, and the region gets the size
  #   the option gave (the lazy initialisation at main+0x1872a40 has the same test, but nninitStartup runs first and sets the size)
  '4828bbb9e91e802470fec7ac68397bd1': [
    (0x1713920, 'e4d7ff97', '1f2003d5', 'GCreateMalloc: bl 0x17098b0 (FMallocBinned3 constructor, 37 GB of address space) → nop; the host allocator\'s vtable is put in at the gate after it (hle_uemalloc.c)'),
    (0x1885ae8, 'e80000b5', '480000b5', 'nninitStartup: cbnz x8, 0x1885b04 (a size given: no region at all) → cbnz x8, 0x1885af0 (a size given: keep it, still allocate the region)'),
    (0x1885af8, 'e11361b2', 'e10308aa', 'nninitStartup: mov x1, #0xf80000000 (the region always 62 GB) → mov x1, x8 (the size the option gave, else the 62 GB default)'),
  ],
}
OBS = {
  # SHIN MEGAMI TENSEI V 1.0.0 main: right after the cut FMallocBinned3 constructor call (CUTS above), x19 = the new allocator object; the host writes its vtable there (hle_uemalloc.c). The
  # instruction right after the call (adrp) is PC-relative, so the gate takes the next one, the store of GMalloc; nothing calls the object in between
  '4828bbb9e91e802470fec7ac68397bd1': [
    (0x1713928, 21, 'f9045eb3', 'GCreateMalloc: the allocator object is in x19, before it is used: the host puts its vtable in; orig str x19, [x21, #0x8b8]'),
  ],
}
