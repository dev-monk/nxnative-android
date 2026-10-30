// Vulkan backend internals (docs/android/PLAN.md Phase 2), shared by nvn_vk.c (the mtl_* set: render targets, passes, submission, completion), vk_dev.c (loading, driver choice, instance,
// device, validation) and vk_present.c (surface, swapchain, present thread). Vulkan is reached through the function table below, filled from vkGetInstanceProcAddr / vkGetDeviceProcAddr
// of a library opened at run time (the system loader, or libadrenotools' for a driver package): nothing links libvulkan, so one host binary runs on any of them
#pragma once
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include "../nvn/nvn_tex.h"   // nvn_texdesc

#define VLOG(...) do { fprintf(stderr, "[dm] [vkn] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#define VRED_ONCE(...) do { static _Atomic int said_; if (!atomic_exchange(&said_, 1)) VLOG("[RED] " __VA_ARGS__); } while (0)

// The platform surface structures are not in the vendored headers (vulkan_core.h only: dm/third_party/README.md); their layouts as the registry has them
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; void *window; } vkn_android_surface_ci;   // VkAndroidSurfaceCreateInfoKHR (ANativeWindow *window)
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; const void *pLayer; } vkn_metal_surface_ci;   // VkMetalSurfaceCreateInfoEXT (const CAMetalLayer *pLayer)
// VkPhysicalDevicePortabilitySubsetFeaturesKHR (a beta extension: vulkan_beta.h, not vendored), as the registry lays it out; MoltenVK has it, KosmicKrisp, Adreno and Turnip do not
#define VKN_STYPE_PORTABILITY_SUBSET_FEATURES ((VkStructureType)1000163000)
typedef struct { VkStructureType sType; void *pNext; VkBool32 constantAlphaColorBlendFactors, events, imageViewFormatReinterpretation, imageViewFormatSwizzle, imageView2DOn3DImage,
                 multisampleArrayImage, mutableComparisonSamplers, pointPolygons, samplerMipLodBias, separateStencilMaskRef, shaderSampleRateInterpolationFunctions, tessellationIsolines,
                 tessellationPointMode, triangleFans, vertexAttributeAccessBeyondStride; } vkn_portability_features;
typedef VkResult (VKAPI_PTR *vkn_create_surface_fn)(VkInstance, const void *, const VkAllocationCallbacks *, VkSurfaceKHR *);

#define VKN_GLOBAL(X) X(vkCreateInstance) X(vkEnumerateInstanceVersion) X(vkEnumerateInstanceLayerProperties) X(vkEnumerateInstanceExtensionProperties)
#define VKN_INSTANCE(X) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFormatProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) \
    X(vkGetDeviceProcAddr) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR) X(vkCreateDebugUtilsMessengerEXT) X(vkDestroyDebugUtilsMessengerEXT)
#define VKN_DEVICE(X) X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkQueueWaitIdle) X(vkQueueSubmit2) X(vkQueuePresentKHR) X(vkCreateSwapchainKHR) \
    X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdPipelineBarrier2) \
    X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdClearAttachments) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDraw) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateSampler) X(vkDestroySampler) X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateGraphicsPipelines) X(vkCreateComputePipelines) X(vkCmdDispatch) X(vkDestroyPipeline) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) X(vkFlushMappedMemoryRanges) X(vkCmdCopyBuffer) X(vkCmdCopyImageToBuffer) X(vkGetBufferDeviceAddress) X(vkCmdCopyBufferToImage) \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdDrawIndexed) X(vkCmdSetBlendConstants) \
    X(vkCmdSetStencilReference) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) X(vkCmdSetDepthBias) X(vkCmdPushDescriptorSetKHR) X(vkCreatePipelineCache) \
    X(vkDestroyPipelineCache)
typedef struct {
#define VKN_PFN(n) PFN_##n n;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    VKN_GLOBAL(VKN_PFN) VKN_INSTANCE(VKN_PFN) VKN_DEVICE(VKN_PFN)
#undef VKN_PFN
    vkn_create_surface_fn create_surface;   // vkCreateAndroidSurfaceKHR or vkCreateMetalSurfaceEXT, whichever this platform's instance enabled
} vkn_fn;
extern vkn_fn vk;

// The device (vk_dev.c, made once by vkn_dev_init): NULL handles when it could not be made, and the backend runs as the null one does (said in red)
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qfam;
    VkPhysicalDeviceProperties props; VkPhysicalDeviceDriverProperties drv; VkPhysicalDeviceMemoryProperties mem;
    int validation;                      // DM_VK_VALIDATION=1: VK_LAYER_KHRONOS_validation and the messenger are on
    const char *surface_ext;             // VK_KHR_android_surface / VK_EXT_metal_surface; NULL = none, nothing is shown
    int absent;                          // vkn_dev_init failed for want of Vulkan itself (no loader library, or a loader with no driver), not a broken one (t_vkn's only skip)
    int bda;                             // bufferDeviceAddress is on (Vulkan 1.3 requires it): shadow buffers have device addresses (nvn_vk.c mtl_gpu_addr)
    int aniso, mirror, minmax;           // samplerAnisotropy, samplerMirrorClampToEdge, samplerFilterMinmax are on (where the device has them: nvn_vk.c samp_make says a sampler that needs a missing one)
    int port, port_bias, port_swz;       // VK_KHR_portability_subset is on (required where the driver has it), with its samplerMipLodBias / imageViewFormatSwizzle features; every feature it
                                         // offers is enabled, and nvn_vk.c avoids a use whose feature is absent (a LOD bias of 0, an identity view swizzle: said once)
    int push; uint32_t max_push;         // VK_KHR_push_descriptor is on, and its maxPushDescriptors (the draws' one set: nvn_vk.c)
    int clip_ctl;                        // VK_EXT_depth_clip_control's depthClipControl is on (near = -w: mtl_set_depth_zero)
    int depth_clamp, indep_blend, dual_src, bias_clamp, fill_ns;   // depthClamp, independentBlend, dualSrcBlend, depthBiasClamp, fillModeNonSolid are on
    int gather_ext, draw_params, sample_rate;   // shaderImageGatherExtended, shaderDrawParameters, sampleRateShading are on (vk_shader.c refuses a module declaring a capability whose feature is off)
    uint32_t sg_ops, sg_stages, sg_size;   // VkPhysicalDeviceSubgroupProperties (vk_shader.c cap_ok takes the GroupNonUniform* capabilities it reports; DM_VK_SUBGROUP=0: none)
    int bc_native;                       // textureCompressionBC is on (else BC textures are decoded on the CPU: vk_bc.c)
    int nofmt_w, nofmt_r;                // shaderStorageImageWriteWithoutFormat / ReadWithoutFormat are on (DM_VK_NOFMT=0: off)
} vkn_dev_t;
extern vkn_dev_t vd;
int vkn_dev_init(void);                  // 1 = a device; logs the driver's identity
uint64_t vkn_validation_count(void);     // warnings and errors the messenger has had
int vkn_mem_type(uint32_t bits, VkMemoryPropertyFlags want);   // a memory type index in bits with want, -1 = none

// Host images of NVN textures (nvn_vk.c), by the texture's serial, in plain device memory (PLAN.md 2.2: render-target pools have no shadow): one image per serial, whether it was first made as a
// render target or as a sampled texture (PLAN.md 2.3, slice 2b), with every usage its format offers
typedef struct vkn_img {
    uint64_t serial; VkImage img; VkDeviceMemory mem; VkFormat fmt, sfmt; VkImageAspectFlags aspect;   // sfmt: the format the present samples it as (UNORM for an sRGB target: bytes pass through)
    uint32_t w, h, layers, levels, nvn_fmt;
    int cube;                            // made cube-compatible (square, layers a multiple of 6): sampled through cube views where a shader or a view asks
    uint32_t depth;                      // a 3D texture's slices (1 for the others): a VK_IMAGE_TYPE_3D image, sampled through 3D views
    VkImageLayout layout;                // as of the last submission handed to the queue (under g_qmu); the present restores it after sampling
    VkImageLayout rec;                   // as the submission being recorded leaves it (submit thread); published to layout at the commit
    uint64_t rec_sub;                    // the submission rec belongs to
    VkPipelineStageFlags2 sr_st;         // while rec is the sampled layout: the stages made ready for it (nvn_vk.c tex_ready)
    uint64_t use_nvn, use_pres;          // the NVN timeline / present timeline value of its last use: destroyed once both have completed
    struct { uint32_t key; VkImageView v; } views[16]; int nviews;   // attachment views by level << 16 | layer, and the present's (key ~0)
    float clear[4]; int cleared;         // the last colour clear landed on it (the present logs it: what the screen should show)
    VkImageUsageFlags usage;             // what it was made with
    int gpu;                             // made for a render target (TextureInitialize in a CPU-no-access pool, or SET_RT), not for sampling
    // Sampled textures (nvn_vk.c, submit thread): the storage in guest memory and its description, the upload's state, and the views
    const uint8_t *gmem; uint64_t size; nvn_texdesc d;
    int once;                            // a CPU-sourced texture in a CPU-no-access pool: uploaded once at its first use (the game cannot write it again), never checked again
    int up, alw;                         // up: its content was uploaded from guest memory (stamp is that upload's); alw: it lies in a page another pool shares, never protected: uploaded at every use
    uint64_t stamp;                      // nvn_dirty generation of the last upload (every use checks its pages against it: nvn_vk.c tex_ready)
    const void *span;                    // its nvn_dirty range (the pool's)
    uint8_t *edge[2]; uint32_t edge_n[2];   // its bytes in the pages it shares with other data at each end, as last uploaded: a write there that leaves them as they were uploads nothing (nvn_mtl.m pkg_edges_*)
    struct { uint64_t key; VkImageView v; } sviews[8]; int nsviews;   // sampled views by format, levels, layers, swizzle and aspect (nvn_vk.c img_sview)
    int dead;                            // released by the game, destruction pending
    VkImage fbi; VkDeviceMemory fbm; VkImageView fbv, fbsv;   // a copy of level 0, layer 0 for draws that sample the pass's own target (nvn_vk.c fb_copy); fbsv: its stencil view
    uint64_t fb_ep; uint8_t fb_ok;       // which parts of the copy are current (bit 0 colour or depth, bit 1 stencil: cleared by what writes them), in the same write epoch and frame (fb_ep)
    struct vkn_img *next;                // the table's bucket chain, then the pending-destruction list
} vkn_img;
vkn_img *vkn_img_get(uint64_t serial);  // under vkn_tmu
extern pthread_mutex_t vkn_tmu;          // the image table
extern pthread_mutex_t vkn_qmu;          // the queue (vkQueueSubmit2, vkQueuePresentKHR, vkQueueWaitIdle) and the published layouts
VkImageView vkn_img_view(vkn_img *e, uint32_t level, uint32_t layer, int present);   // under vkn_tmu
void vkn_sweep(void);                    // destroys released images, and freed pools' shadow buffers, whose last uses have completed
// Strategy B (nvn_vk.c, PLAN.md 2.2 "As built, slice 2a"): a game address → its pool's shadow VkBuffer (as void *), the offset in it and its device address (0 without bufferDeviceAddress).
// 0 = no pool with a shadow holds [addr, addr+len): a pool gets one when a replayed record first reads it as a buffer
int mtl_gpu_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off, uint64_t *gpu);
// The GPU lost to this run (VK_ERROR_DEVICE_LOST, or a submission the queue refused): said once in red; from then on nothing reaches the device (a lost device's queue would keep failing,
// and its command buffers stay pending for ever): NVN submissions complete in order without it (the null backend's behaviour) and presents are retired unshown. Any thread
extern _Atomic int vkn_lost;
void vkn_gpu_lost(const char *where, VkResult r);

// The NVN timeline (nvn_vk.c) and the present's own (vk_present.c): never mixed, an NVN fence value is only ever an NVN submission's
uint64_t vkn_pres_done(void);
int vkn_origin_ll(void);                 // NVN's window origin is lower left (mtl_set_origin_lower_left)
// The present thread (vk_present.c): a ticket per present, done once its frame is handed to the queue; log: say what this present shows; shot: a PNG path (DM_NVN_SHOT picked this present,
// nvn_mtl.m's numbering: the present count, which is not the ticket's) or NULL: the window texture is read back and written there before the ticket is done
uint64_t vkn_present_enqueue(uint64_t serial, uint32_t w, uint32_t h, const int crop[4], int log, const char *shot);
void vkn_present_wait(uint64_t ticket);
void vkn_present_flush(void);            // every ticket so far done, and the GPU through their present work
int vkn_present_drain(uint64_t ns);      // the same within ns: 1 = done

// Shader modules (vk_shader.c, PLAN.md 2.4, slice 2c): slice 0's SPIR-V of this translation chain (dmsh_build_vk.py; a module of another is refused), by the shader's key (hle_nvn.c prog_key)
#define VKN_CHAIN "2efb3d91898e"   // dmsh_build_vk.py --version: moves with the Metal chain (upstream ecbeae1 changed dmsh_build.py, dmsh_chain.py and the lock); spv_add_caps.py and spv_barrier_exit.py joined it
#define VKN_MAXB 32
enum { VKN_B_UBO = 1, VKN_B_C1, VKN_B_ZERO, VKN_B_TEX, VKN_B_SUPPORT, VKN_B_SBUF, VKN_B_IMG, VKN_B_DRV0, VKN_B_TEXD, VKN_B_SAMPD, VKN_B_DRV2, VKN_B_TEXU, VKN_B_UBOP };   // UBOP: uniform slot reg bound as a storage buffer (compute's uniforms as device pointers, nvn_mtl.m NVN_RB_UBOP)   // TEXU: texture handles in a uniform buffer (slot reg, byte off; n > 1: an array, element k at off + 8k)   // TEXD / SAMPD: n textures / samplers picked at run time by slot (tex_dyn_slots); DRV2: driver bank c2 (draw_drv2)   // VKN_B_DRV0: driver bank c0 the shader reads (nvn_vk.c draw_drv0)   // VKN_B_IMG: a storage image, NVN image slot reg (compute)   // also Ryujinx's support_buffer (nvn_vk.c g_supbuf), an NVN storage buffer slot (reg; rw: written). A binding: an NVN uniform slot (reg), the shader's c1 bank, a driver bank it reads nothing of (zeros), a combined texture slot (reg)
enum { VKN_S_TEX = 1, VKN_S_SAMP, VKN_S_ID };              // its sampler: the sampler half of combined slot sreg, separate sampler slot sreg, the constant sampler ID sreg
enum { VKN_VT_2D, VKN_VT_2D_ARRAY, VKN_VT_CUBE, VKN_VT_3D };
typedef struct { uint8_t kind, skind, vt, comp, set, rw, n; uint16_t orig, binding, reg, sreg, off; } vkn_sbind;   // n: an array binding's length (TEXD, SAMPD)   // binding: in the draws' set (pixel ones + 32); comp 0 float 1 uint 2 sint
typedef struct vkn_shader {
    char key[17]; int stage;             // NVN stage: 0 vertex, 1 pixel
    VkShaderModule mod; uint32_t id;     // id: a small number, for the pipeline keys
    int nb; vkn_sbind b[VKN_MAXB];
    int nin; struct { uint8_t loc, ct, n; } in[16];   // vertex inputs: location, component type (as comp), components
    uint32_t out_mask; uint8_t out_ct[8], out_n[8];   // pixel outputs written, by location, and their component types and components
    uint32_t io_mask; uint8_t io_ct[32], io_n[32];    // the interface between the stages, by location: a vertex shader's outputs, a pixel shader's inputs (component type as comp, components)
    uint8_t io_first[32], io_nd[32], io_dims[32][4];  // and the variable on each: its first location and array lengths (outermost first; io_nd 0: no array, io_first the location)
    int dyn;                                          // has TEXD / SAMPD bindings: c2 holds slot indices for it (draw_drv2)
    int idmap;                                        // TEXD / SAMPD indices looked up in the slots' handles (vk_shader.c spv_idmap): the 32-word push constant table (nvn_vk.c draw_idmap)
    int dual;                                         // a pixel shader with an Output decorated Index other than 0 (dual-source blending's second colour)
    VkBuffer c1_buf; VkDeviceMemory c1_mem; uint32_t c1_n;
    VkDescriptorSetLayout cdsl; VkPipelineLayout cpl; VkPipeline cpso; int cbad;   // a compute shader's own layout and pipeline (nvn_vk.c op_dispatch); cbad: they cannot be made
} vkn_shader;
int vkn_shmap_stage(const uint8_t key[8]);           // the shader map's NVN stage of a key, -1 = not in it
vkn_shader *vkn_shader_get(const uint8_t key[8], int stage);   // NULL = refused or missing (said once)
vkn_shader *vkn_shader_add(const uint8_t key[8], int stage, const char *json, const uint32_t *spv, size_t nbytes);   // t_vkn's modules
void vkn_shader_stats(uint64_t *loaded, uint64_t *refused);
