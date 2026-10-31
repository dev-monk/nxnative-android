# Android and Vulkan backend switches

Environment variables read by the Android front end (`dm/android/front`), the Vulkan backend (`dm/gfx/vkn`) and the Vulkan shader build. In the app they go in `env.txt` (one `NAME=value` per line; [PLAN.md 1.7](PLAN.md#17-paths-and-settings)); for the headless `dmhost` they are ordinary environment variables. ★ marks the switches added with Paper Mario: The Thousand-Year Door's bring-up on Mali.

Most compat paths are on by default, and `=0` turns one off. Each logs which path runs, so a regression can be bisected without a rebuild.

## Front end and host

| Switch | Default | What it does |
|---|---|---|
| `DM_OPMODE` | the profile's `op_mode` | `0` handheld, `1` docked at start (`hle_oe.c`). |
| `DM_HOTKEY_OPMODE` ★ | off | `1`: Select (−) + right-stick press toggles handheld / docked while the game runs (`hotkey.c`). Off by default because another game may use the combination. |
| `DM_STICK_DEADZONE` | 0.15 | Stick dead zone, a fraction from 0 to 0.9. |
| `DM_INPUT_LOG` | off | `1`: log each key press and what reached the controller. |
| `DM_FPS` | the game's | `30` or `60`: pin the frame rate (test switch). |

## Vulkan backend: compat paths (on; `=0` off)

| Switch | `=0` gives |
|---|---|
| `DM_VK_COMPUTE` ★ | Compute dispatches not run (the shaders are refused, as before). |
| `DM_VK_PASS_KEEP` ★ | Every SET_RT and ordering barrier ends the render pass. On: a pass stays open across a SET_RT of the same targets or of some of them with the same depth, and across ordering barriers. |
| `DM_VK_SPLIT` ★ | `0`: one submission per QueueSubmitCommands. Default `400`: a frame goes to the GPU in parts every 400 draws, at a target change. |
| `DM_VK_POOL_INDEX` ★ | Memory pools found by a walk over all of them, not the sorted index. |
| `DM_VK_ROBUST` ★ | Robust buffer / image access off. |
| `DM_VK_SUBGROUP` ★ | Subgroup operations not used (modules that need them are refused). |
| `DM_VK_ASTC` ★ | No native ASTC textures. |
| `DM_VK_BC_NATIVE` ★ | Native BC formats not used, even where the device has them. |
| `DM_VK_BCDEC` ★ | No CPU decode of BC textures on devices without BC (they read as zeros). |
| `DM_VK_NOFMT` ★ | Storage images without format: the features stay off, and the modules that need them are refused. |
| `DM_VK_NOACC_UP` ★ | Packaged textures in CPU-no-access pools are not read (zeros). |
| `DM_VK_COPY` ★ | Texture copies and clears not done. |
| `DM_VK_FEEDBACK` ★ | A draw that samples a target of its own pass reads zeros instead of a copy. |
| `DM_VK_3D`, `DM_VK_3D_IMG` ★ | 3D textures refused / without host images. |
| `DM_VK_CUBE` ★ | Cube maps left out. |
| `DM_VK_STORAGE_IMG` ★ | Images get no storage usage (compute stores to them fail). Default: an image gets storage usage only when a dispatch binds it, remade in place with its contents (no storage usage keeps framebuffer compression, AFBC, on Mali). `1`: every image with the format feature gets it at creation, as before. |
| `DM_VK_CS_FRAG` ★ | Every dispatch runs as a dispatch. Default: a compute shader whose SPIR-V provably writes each storage image once at its own pixel (no barriers, workgroup memory or image reads) runs as a fragment pass over those images as colour attachments, so they keep AFBC. `<dir>`: load `<dir>/<key>.spv` as the fragment shader instead (diagnostics). |
| `DM_VK_CLEAR_FOLD` ★ | Full clears with no draw after them are recorded where they are. On: they become the next pass's load op. |
| `DM_VK_AREA` ★ | Passes cover their whole targets. On: a pass's render area is the part its draws can reach (viewports and scissors), rounded out to 32 pixels, when no attachment is cleared by its load op. |
| `DM_VK_PCACHE` ★ | No pipeline cache file. On: `<DM_VK_SHADERS>/pipelines.vkpc`, written when the cache grew. |
| `DM_VK_SRSTAGE` ★ | A drawn target is made ready for every stage. On: only for the stage that samples it. |

| `DM_VK_SUPPORT`, `DM_VK_SBUF`, `DM_VK_UBOP`, `DM_VK_TEXU`, `DM_VK_TEXD`, `DM_VK_DRV0`, `DM_VK_DRV2` ★ | That binding kind refused, as before: support buffer, storage buffers, uniform slots read as storage, texture handles in uniform buffers, dynamic texture slots (profile key `tex_dyn_slots`), driver banks c0 / c2. |
| `DM_VK_PUSH` | `0`: per-slot descriptor pools even where the device has push descriptors. |

## Vulkan backend: device and diagnostics

| Switch | What it does |
|---|---|
| `DM_VK_SHADERS` | The SPIR-V folder (the app sets it to its package's). |
| `DM_VK_DRIVER`, `DM_VK_DRIVER_DIR`, `DM_VK_DRIVER_LIB`, `DM_VK_HOOK_DIR`, `DM_VK_LOADER`, `DM_VK_EXPECT_INFO` | Driver loading ([PLAN.md 2.1](PLAN.md#21-device-driver-loading-present)). |
| `DM_VK_VALIDATION` | `1`: validation layers. `2` ★: plus synchronization validation. |
| `DM_VK_ANISO` ★ | `<n>`: anisotropy at most n (`1`: none). |
| `DM_VK_DS_STORE` ★ | `1`: depth / stencil a pass cannot write is not stored (no gain on Mali). |
| `DM_VK_HOT` ★ | `1`: buffer pages written again within a few submissions are copied at every submission, unprotected (no gain on Mali, one hang seen). |
| `DM_VK_STORAGE_SER` ★ | `<serial>[,<serial>…]`: only those images get storage usage, at creation (measurement only). |
| `DM_VK_SPV_OVR` ★ | `<dir>`: `<dir>/<key>.spv` replaces a module once the loader's checks pass. |
| `DM_VK_SPV_DUMP` ★ | `<dir>`: every module as handed to the driver, after the loader's patches, plus `<key>_fs.spv` for compute shaders run as passes. |
| `DM_VK_PRESENT_TEST` | `1`: present a test pattern instead of the game's frames. |
| `DM_VK_PASSLOG` ★ | `<present>`: every target change, draw, dispatch, copy and clear of that frame, and where each pass ended. |
| `DM_VK_SHOT_SER` ★ | `<serial>[,<serial>…]`: `DM_NVN_SHOT` screenshots take that texture instead of the window. |
| `DM_VK_DIAG` ★ | Draw-state overrides, comma-separated: `solidps` (magenta pixel shader), `nodepth`, `nostencil`, `nocull`, `csclear` (storage images magenta before a dispatch); bounds that give a wrong picture: `rdiscard` (nothing rasterized), `dsnone` (depth / stencil never stored), `fbstale` (feedback copies not refreshed), `texlod` (textures from mip 4); `afbc` logs whether each image format and usage is compressed. Diagnostics only. |
| `DM_VK_CS_SKIP` ★ | Compute shader keys (substring match) not dispatched. |

## Vulkan shader build (`dmsh_build_vk.py`)

| Switch | What it does |
|---|---|
| `DMSH_VK_NOFMT` ★ | `0`: `spv_add_caps.py` declares no StorageImage{Read,Write}WithoutFormat. |
| `DMSH_VK_BARRIER_EXIT` ★ | `0`: `spv_barrier_exit.py` leaves early returns before workgroup barriers as they are (Mali then loses the device on such shaders). |
| `DMSH_KEEP_WORK` | `1`: keep the work directory (translator output before the passes). |

## Game profile (patch time)

| Switch | What it does |
|---|---|
| `DM_TTYD_HANDHELD_RES`, `DM_TTYD_DOCKED_RES` ★ | Paper Mario: The Thousand-Year Door 1.0.1: the 3D render size, `WxH` (up to 1152x648 / 1600x900; default unchanged). Read by its `patches.py` when the modules are patched. 960x540 handheld is the fastest setting on the RG 476H. |
