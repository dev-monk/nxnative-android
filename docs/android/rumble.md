# Android rumble

The game's `nn::hid` vibration (`hle_hid.c` → `hle_haptics.c`) reaches an Android vibrator through `HapticsRouter`. It goes to player 1's controller when that controller has a motor, and to the device's built-in motor otherwise. Built into `libdmhost.so` and the app's `classes.dex`. The code compiles and the host tests pass, but it has not yet been run on a device.

## Pieces

| File | Role |
|---|---|
| `dm/host/haptics_android.[ch]` | Portable dispatcher. The guest-side sink only queues two-motor snapshots (256 max; a STOP is never dropped). A worker thread calls out, so no JNI runs under the HID lock. Route tokens, generations and epochs reject stale output. |
| `dm/host/haptics_android_jni.c` | Bridge from the worker to `HapticsRouter.applyFromNative`. Also has `dm_haptics_android_register`, which binds the natives with `RegisterNatives`. |
| `dm/android/app/java/dev/nxnative/haptics/HapticsRouter.java` | Public-API vibrator routing. Built-in motor via `VibratorManager`; a controller's motor via `InputDevice.getVibratorManager()`. Modes Auto / Built-in / Controller / Off, an on/off switch and a gain of 0..3, persisted in the `nxnative.rumble` SharedPreferences. Output is mono (the two Switch motors combined). A motor without amplitude control gets duty-cycled pulses. |
| `dm/android/front/front_android.c` (`haptics_init`) | Runs in `onCreate` before the host thread starts. Binds the router's natives and `HostActivity.nativeHapticsLog`, then calls `HostActivity.startHaptics`. |
| `dm/android/app/java/org/nxnative/host/HostActivity.java` | Creates the router. Picks player 1's controller and re-picks on hotplug. Pauses rumble unless the activity is resumed and has focus. Closes the router before `onDestroy` ends the process. |
| `dm/android/app/AndroidManifest.xml` | `android.permission.VIBRATE` (a normal permission, granted at install). |
| `dm/host/tests/t_haptics_android.c` | Host test of the dispatcher (`make GAME=<id> build/t_haptics_android`). |
| `dm/android/rumble-probe/` | Optional standalone Gradle app that drives the same backend with left/right/both/sustained/stop buttons and shows a capability report. Needs Android SDK 35 and AGP 8.7.3. It is not part of `build_app.sh`. |

The NativeActivity loads `libdmhost.so` itself, not through `System.loadLibrary`. ART therefore never looks up the exported `Java_dev_nxnative_haptics_*` names in it, and the host binds them through `RegisterNatives`. The exported names are still there for the probe, which does load its library with `System.loadLibrary`.

## Routing

All controllers are merged into player 1 (`front_android.c`). `HostActivity.hapticsPick` gives player 1 the first non-virtual gamepad or joystick that has a vibrator. If there is none, it passes `-1`, and Auto mode falls back to the built-in motor. Players 2 to 8 get no output. The router never sends one output to two players.

## Switches and log

- `DM_RUMBLE=0` in `env.txt`: no router is started, and the game sees no vibration output (as before this change).
- `log.txt` gets lines prefixed `rumble:`:
  - at start, the capability report: SDK, build, mode and strength, built-in vibrator IDs with amplitude support, gamepads with their motors, and each player's route;
  - `player 1's controller: ... → Player 1: built-in, amplitude` (or `controller #N, ...`) on each change.
- `boot: [RED] rumble did not start` when binding or start failed. The game then runs without rumble.
- Router errors go to logcat under the tag `NxHaptics`.

## Not done yet

- There is no in-app menu for the router's mode, on/off or strength. The methods `setOutputMode`, `setEnabled` and `setStrength` exist and persist their values, but nothing in the UI calls them yet.
- Output is a mono amplitude approximation, not Switch HD rumble frequency and not separate left and right motors.
- There is no kernel `FF_RUMBLE` path. A motor that is not exposed through the public vibrator API shows as unavailable.

## Testing on a device

1. Install the APK. Start a game that rumbles.
2. Check the `rumble:` lines in `log.txt`, and run `adb logcat -s NxHaptics`.
3. Test the built-in motor, then a USB or Bluetooth controller connected mid-game.
4. Check that rumble stops on Home, when the notification shade opens, and when the input box appears. It should start again with the game's next vibration command.
