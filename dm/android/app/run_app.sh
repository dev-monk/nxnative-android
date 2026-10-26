#!/bin/bash
# Install the app (build_app.sh) on the connected phone, give it the game files and run it (docs/android/PLAN.md Phase 1): run_app.sh <module dir> <RomFS dir> [options]
#   <module dir>  the rewritten modules (dm/tools/patch_nso.py --target android: <name>.patched.bin and its .tsv files), copied into the app's files/mods through /data/local/tmp/nxhost-app and run-as;
#                 - for none: a game's APK (scripts/build_apk_from_nsp.py) carries its own, with its Vulkan shaders, and unpacks them itself
#   <RomFS dir>   the game's RomFS, pushed (adb push --sync: what changed) to the app's external files directory, /sdcard/Android/data/<package>/files/romfs; - for none (already there)
#   The package is the APK's (aapt2 dump packagename): org.nxnative.host for build_app.sh's development APK, org.nxnative.t<title ID> for a game's; APP_PKG names it when the APK is not at hand
#   --secs N        wait N seconds after the activity started (default 60), then pull files/log.txt (and the process's logcat and each thread's blocked signals)
#   --env K=V       a line of files/env.txt (repeatable); --env-file F puts F's lines first. env.txt is rewritten on every run (empty without either)
#   --fresh         start from an empty save and temp directory
#   --key T:CODE[~MS]  press a key T seconds after the start returned, held MS ms (default 150, DM_KEYS' hold; repeatable), e.g. --key 25:KEYCODE_BUTTON_B. Sent with adb shell input
#                   keyevent --duration (a gamepad's for BUTTON_ and DPAD_ codes, else a keyboard's): a plain keyevent goes down and up within a millisecond, which one game read may miss
#   --out DIR       where the pulled files go (default ${TMPDIR:-/tmp}/nxhost-app-run, outside the work tree)
#   --no-install    keep the installed APK; --keep  leave the app running afterwards (by default it is force-stopped once the log is pulled)
#   --driver D      the Vulkan backend's APK (build_app.sh --gpu vkn: build/nxhost-vkn.apk, unless APP_APK says otherwise) on driver D: system, 842 or turnip (DM_VK_DRIVER, first in env.txt).
#                   A package driver is unpacked into the app's files/drivers/D/, where dm/gfx/vkn/vk_dev.c opens it through libadrenotools, as dm/android/probe/shader/run_shader.sh installs
#                   it: ZIP_842 (default /tmp/qcom_drv/v842/8Elite2-842.6.zip; DM_VK_EXPECT_INFO=${Q842_INFO:-E031.50} too, its driverID being the stock driver's), ZIP_TURNIP
#                   (/tmp/qcom_drv/turnip_a8xx/turnip_a8xx.zip). Driver packages are the user's own files: never in the repository
# The game files go to the phone only. Touches nothing but the APK's package (org.nxnative.* only), its directories and /data/local/tmp/nxhost-app; never clears logcat.
# adb shell joins its arguments into one command line for the phone's shell, so nothing from the command line or the module directory reaches one unchecked: module file names must match
# ^[A-Za-z0-9._-]+$ (and not start with -), key codes ^[A-Z0-9_]+$, times and holds plain numbers, the pid digits
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ANDROID_HOME=${ANDROID_HOME:-$HOME/Library/Android/sdk}
ADB=${ADB:-$ANDROID_HOME/platform-tools/adb}
BT=${APP_BUILD_TOOLS:-$ANDROID_HOME/build-tools/36.1.0}
APK=${APP_APK:-}
STAGE=/data/local/tmp/nxhost-app
die() { echo "error: $*" >&2; exit 1; }
now() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }
[ $# -ge 2 ] || die "usage: run_app.sh <module dir> <RomFS dir> [--secs N] [--env K=V]... [--env-file F] [--fresh] [--key T:CODE[~MS]]... [--out DIR] [--no-install] [--keep] [--driver system|842|turnip]"
MODS=$1 ROMFS=$2; shift 2
SECS=60 OUT=${TMPDIR:-/tmp}/nxhost-app-run FRESH=0 INSTALL=1 KEEP=0 ENVF="" DRV=""; ENVS=(); KEYS=()
while [ $# -gt 0 ]; do
    case $1 in
        --secs) SECS=$2; shift 2 ;; --env) ENVS+=("$2"); shift 2 ;; --env-file) ENVF=$2; shift 2 ;; --fresh) FRESH=1; shift ;;
        --key) KEYS+=("$2"); shift 2 ;; --out) OUT=$2; shift 2 ;; --no-install) INSTALL=0; shift ;; --keep) KEEP=1; shift ;; --driver) DRV=$2; shift 2 ;;
        *) die "unknown option $1" ;;
    esac
done
[ "$MODS" = - ] || ls "$MODS"/*.patched.bin >/dev/null 2>&1 || die "$MODS holds no rewritten modules (*.patched.bin; - for an APK that carries its own)"
FILES=(); NAMES=()   # the module directory's files, each name checked: they are spelled out in a command line run on the phone (below)
[ "$MODS" = - ] || for f in "$MODS"/*; do
    n=${f##*/}
    [[ $n =~ ^[A-Za-z0-9._-]+$ && $n != -* ]] || die "module file name '$n' has characters other than A-Z a-z 0-9 . _ - (or starts with -): rename it"
    [ -f "$f" ] && [ ! -L "$f" ] || die "$f is not a regular file"
    FILES+=("$f"); NAMES+=("$n")
done
[[ $SECS =~ ^[0-9]+$ ]] || die "--secs takes whole seconds, not '$SECS'"
for k in ${KEYS[@]+"${KEYS[@]}"}; do
    [[ $k =~ ^[0-9]+(\.[0-9]+)?:[A-Z0-9_]+(~[0-9]+)?$ ]] || die "--key takes T:CODE[~MS] (T seconds, CODE like KEYCODE_BUTTON_B, MS milliseconds), not '$k'"
done
[ "$ROMFS" = - ] || [ -d "$ROMFS" ] || die "no RomFS directory $ROMFS"
ZIP=""; DENV=()
case $DRV in
    "") ;; system) DENV=(DM_VK_DRIVER=system) ;;
    842) ZIP=${ZIP_842:-/tmp/qcom_drv/v842/8Elite2-842.6.zip}; DENV=(DM_VK_DRIVER=842 "DM_VK_EXPECT_INFO=${Q842_INFO:-E031.50}") ;;
    turnip) ZIP=${ZIP_TURNIP:-/tmp/qcom_drv/turnip_a8xx/turnip_a8xx.zip}; DENV=(DM_VK_DRIVER=turnip) ;;
    *) die "--driver takes system, 842 or turnip, not '$DRV'" ;;
esac
[ -z "$ZIP" ] || [ -f "$ZIP" ] || die "no driver package $ZIP"
[ -n "$APK" ] || if [ -n "$DRV" ]; then APK=$HERE/build/nxhost-vkn.apk; else APK=$HERE/build/nxhost.apk; fi
[ -z "$ENVF" ] || [ -f "$ENVF" ] || die "no env file $ENVF"
PKG=${APP_PKG:-}
if [ -z "$PKG" ]; then if [ -f "$APK" ]; then PKG=$("$BT/aapt2" dump packagename "$APK" 2>/dev/null | tr -d '\r') || die "aapt2 cannot read the package of $APK"; else PKG=org.nxnative.host; fi; fi
[[ $PKG =~ ^org\.nxnative\.[A-Za-z][A-Za-z0-9_]*$ ]] || die "package '$PKG' is not org.nxnative.<name>: this script touches only those"
ACT=$PKG/org.nxnative.host.HostActivity   # the class keeps its own package under a game's (the short form .HostActivity would name <package>.HostActivity)
echo "package $PKG ($APK)"
mkdir -p "$OUT"

if [ $INSTALL = 1 ]; then
    [ -f "$APK" ] || die "no $APK: build it first (dm/android/app/build_app.sh <GAME>)"
    # Play Protect may hold a sideloaded install for a prompt on the phone: one retry
    "$ADB" install -r -t "$APK" >/dev/null || { echo "install failed, retrying once (approve the prompt on the phone if one is shown)"; "$ADB" install -r -t "$APK" >/dev/null; }
fi
if [ -n "$DRV" ]; then   # --driver needs the Vulkan backend's APK installed (--no-install, or APP_APK, may leave the null one: it has no libadrenotools.so)
    PMP=$("$ADB" shell pm path $PKG || true); IAPK=$(printf '%s\n' "$PMP" | tr -d '\r' | sed -n 's/^package://p' | head -1)   # captured whole: under pipefail an early exit in a pipe would fail it
    [ -n "$IAPK" ] || die "$PKG is not installed"
    LST=$("$ADB" shell "unzip -l $IAPK" || true)
    grep -q 'lib/arm64-v8a/libadrenotools\.so' <<< "$LST" || die "the installed $PKG is not the Vulkan backend's APK (no lib/arm64-v8a/libadrenotools.so in $IAPK): build_app.sh <GAME> --gpu vkn, and install it"
fi
"$ADB" shell am force-stop $PKG
# The external files directory: the app's getExternalFilesDir() makes it with the app's ownership on its first start; one adb made first may not be the app's to read
EXT=/sdcard/Android/data/$PKG/files
if ! "$ADB" shell "test -d $EXT"; then
    "$ADB" shell am start -W -n $ACT >/dev/null; t=0
    until "$ADB" shell "test -d $EXT" || [ $t -ge 20 ]; do sleep 1; t=$((t + 1)); done
    "$ADB" shell am force-stop $PKG
fi
"$ADB" shell "test -d $EXT" || die "the app made no $EXT"
# The RomFS: each top-level entry into romfs/ (a directory pushed onto an existing one would land inside it)
if [ "$ROMFS" != - ]; then
    "$ADB" shell "mkdir -p $EXT/romfs"
    TOP=(); for e in "$ROMFS"/*; do TOP+=("$e"); done
    "$ADB" push --sync "${TOP[@]}" "$EXT/romfs/" | tail -1
fi
# The modules: staged in /data/local/tmp (the app cannot read it, run-as can), then copied into files/mods, which holds exactly this set
if [ "$MODS" != - ]; then
    "$ADB" shell "mkdir -p $STAGE/mods"
    "$ADB" push --sync "${FILES[@]}" "$STAGE/mods/" | tail -1
    "$ADB" shell "run-as $PKG sh -c 'rm -rf files/mods && mkdir -p files/mods && cd $STAGE/mods && cp -- ${NAMES[*]} /data/user/0/$PKG/files/mods/'"
fi
# The driver package: staged as the modules are, then unpacked into files/drivers/<driver>/ (a clean directory, so no file of another package version stays)
if [ -n "$ZIP" ]; then
    "$ADB" shell "mkdir -p $STAGE/drivers"
    "$ADB" push "$ZIP" "$STAGE/drivers/$DRV.zip" | tail -1
    "$ADB" shell "run-as $PKG sh -c 'rm -rf files/drivers/$DRV && mkdir -p files/drivers/$DRV && cp $STAGE/drivers/$DRV.zip files/drivers/$DRV.zip && cd files/drivers/$DRV && unzip -o -q ../$DRV.zip && rm ../$DRV.zip'"
    echo "driver $DRV: $("$ADB" shell "run-as $PKG ls files/drivers/$DRV" | tr -d '\r' | tr '\n' ' ')"
fi
# env.txt: the first line that sets a variable wins (front_android.c env_file), so --driver's lines go first
{ [ ${#DENV[@]} -eq 0 ] || printf '%s\n' "${DENV[@]}"; [ -z "$ENVF" ] || cat "$ENVF"; [ ${#ENVS[@]} -eq 0 ] || printf '%s\n' "${ENVS[@]}"; } | "$ADB" shell "run-as $PKG sh -c 'cat > files/env.txt'"
echo "env.txt:"; "$ADB" shell "run-as $PKG cat files/env.txt" | sed 's/^/  /'
[ $FRESH = 0 ] || "$ADB" shell "run-as $PKG sh -c 'rm -rf files/save files/temp'"

# Wake the screen first: an activity started on a sleeping display gets no window, so the host waits for its surface (and key presses then end in an ANR). The app's KEEP_SCREEN_ON
# (front_android.c) keeps it on once the window shows, but cannot wake it. A keyguard with a PIN or pattern still needs the user
"$ADB" shell input keyevent KEYCODE_WAKEUP
"$ADB" shell dumpsys window | grep -q 'isKeyguardShowing=true' && echo "note: the lock screen is showing: unlock the device, or the app gets no window until then"
"$ADB" shell am start -W -n $ACT | grep -E "Status|TotalTime" || true
T0=$(now); echo "started at $(date +%T)"
for k in ${KEYS[@]+"${KEYS[@]}"}; do
    at=${k%%:*} code=${k#*:} hold=150
    case $code in *~*) hold=${code#*~}; code=${code%%~*} ;; esac
    case $code in *BUTTON_*|*DPAD_*) src=gamepad ;; *) src=keyboard ;; esac
    ( sleep "$at"; s=$(now); "$ADB" shell input $src keyevent --duration "$hold" "$code"; e=$(now)
      echo "key $code ($src, held $hold ms) sent at $(echo "$s - $T0" | bc) s (returned at $(echo "$e - $T0" | bc) s)" ) &
done
sleep "$SECS"; wait
PID=$("$ADB" shell pidof $PKG | tr -d '\r' || true); PID=${PID%% *}
[[ $PID =~ ^[0-9]*$ ]] || die "pidof answered '$PID'"
"$ADB" exec-out "run-as $PKG cat files/log.txt" > "$OUT/log.txt"
if [ -n "$PID" ]; then
    "$ADB" logcat -d --pid="$PID" > "$OUT/logcat.txt" || true
    # Each thread's blocked signals, as the kernel has them (the host's own readback is in the log: "signals: thread")
    "$ADB" shell "run-as $PKG sh -c 'for t in /proc/$PID/task/*; do echo \$(cat \$t/comm) \$(grep SigBlk \$t/status); done'" > "$OUT/sigblk.txt" || true
else echo "the app is not running any more"; fi
[ $KEEP = 1 ] || "$ADB" shell am force-stop $PKG
echo "pulled into $OUT: $(ls "$OUT" | tr '\n' ' ')"
grep -E "thread slots|signals:|AAudio stream|front end: (key|window)|hasRoom|boot: (game|modules|package)|\[vkn\] (driver|swapchain|surface|presenting|present [0-9]+:|GPU work)|\[RED\]|FATAL|CRASH" "$OUT/log.txt" | cut -c1-300 || true
grep -E "^\[dm\] \[nvn\] frame [0-9]+:" "$OUT/log.txt" | tail -1 | cut -c1-80 || true
echo "[RED] lines: $(grep -c '\[RED\]' "$OUT/log.txt" || true)"
[ -z "$DRV" ] || echo "validation messages: $(grep -c '\[vkn\] validation:' "$OUT/log.txt" || true)"
