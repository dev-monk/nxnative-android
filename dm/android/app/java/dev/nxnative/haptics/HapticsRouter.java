package dev.nxnative.haptics;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.SharedPreferences;
import android.hardware.input.InputManager;
import android.media.AudioAttributes;
import android.os.Build;
import android.os.CombinedVibration;
import android.os.VibrationAttributes;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.os.VibratorManager;
import android.util.Log;
import android.view.InputDevice;

/**
 * Public-API Android rumble transport for the game's existing nn::hid / dm_haptics backend.
 * Requires Android 8/API 26+, with controller vibrator routing on Android 12/API 31+.
 * The native library containing haptics_android_jni.c must already be loaded.
 *
 * Route generation tokens, rather than unstable Android InputDevice IDs, reject stale
 * JNI deliveries. Every device has one owner: no accidental player 2 -> handheld fallback.
 */
public final class HapticsRouter implements InputManager.InputDeviceListener, AutoCloseable {
    private static final String TAG = "NxHaptics";
    private static final int PLAYERS = 8;
    private static final int INACTIVE = 1;
    private static final int INTERRUPTED = 2;
    public static final int AUTO = 0, BUILTIN = 1, CONTROLLER = 2, OFF = 3;
    private static final long[] HOLD_MS = { 0, 1000 };
    private static final AudioAttributes AUDIO = new AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_GAME).build();

    private final Context app;
    private final InputManager input;
    private final SharedPreferences preferences;
    private final int[] playerInput = new int[PLAYERS];
    private final Route[] routes = new Route[PLAYERS];
    private boolean started, closed, paused, interrupted, enabled = true;
    private int mode = AUTO, nextToken = 1;
    private float gain = 1.0f;

    private static final class Route {
        final int token, deviceId;
        final String identity;
        final VibratorManager manager;
        final Vibrator vibrator;
        final boolean amplitude;
        int lastAmplitude = -1;
        long lastEpoch;
        Route(int token, int deviceId, String identity, VibratorManager manager,
              Vibrator vibrator, boolean amplitude) {
            this.token = token;
            this.deviceId = deviceId;
            this.identity = identity;
            this.manager = manager;
            this.vibrator = vibrator;
            this.amplitude = amplitude;
        }
        void cancel() {
            try {
                if (manager != null) manager.cancel();
                else if (vibrator != null) vibrator.cancel();
            } catch (RuntimeException error) {
                Log.w(TAG, "Vibrator cancellation failed", error);
            }
            lastAmplitude = 0;
        }
        void play(int requested) {
            // A binary device must not turn tiny amplitude requests into 100% continuous buzz.
            // Duty-cycled 20 ms pulses are intentionally capped at 25% for weak signals.
            int amplitude = this.amplitude ? requested : requested >= 192 ? 255 :
                    (requested >= 64 ? 96 : requested >= 20 ? 32 : 0);
            if (amplitude == lastAmplitude) return;
            if (amplitude == 0) { cancel(); return; }
            VibrationEffect effect;
            if (this.amplitude) {
                effect = VibrationEffect.createWaveform(HOLD_MS, new int[]{0, amplitude}, 1);
            } else {
                // Full-power pulse density is proportional to the requested coarse strength.
                int on = amplitude >= 255 ? 60 : amplitude >= 96 ? 15 : 5;
                int off = amplitude >= 255 ? 0 : amplitude >= 96 ? 45 : 55;
                // Repeat the on/off pair (do not repeat the initial 0-delay segment).
                effect = VibrationEffect.createWaveform(new long[]{0, on, off}, 1);
            }
            if (manager != null) {
                if (Build.VERSION.SDK_INT >= 33)
                    manager.vibrate(CombinedVibration.createParallel(effect),
                            VibrationAttributes.createForUsage(VibrationAttributes.USAGE_MEDIA));
                else manager.vibrate(CombinedVibration.createParallel(effect));
            } else if (vibrator != null) {
                if (Build.VERSION.SDK_INT >= 33)
                    vibrator.vibrate(effect, VibrationAttributes.createForUsage(VibrationAttributes.USAGE_MEDIA));
                else vibrator.vibrate(effect, AUDIO);
            }
            lastAmplitude = amplitude;
        }
    }

    public HapticsRouter(Context context) {
        app = context.getApplicationContext();
        input = (InputManager) app.getSystemService(Context.INPUT_SERVICE);
        preferences = app.getSharedPreferences("nxnative.rumble", Context.MODE_PRIVATE);
        enabled = preferences.getBoolean("enabled", true);
        mode = preferences.getInt("mode", AUTO);
        gain = preferences.getFloat("gain", 1.0f);
        if (!Float.isFinite(gain) || gain < 0 || gain > 3) gain = 1.0f;
        if (mode < AUTO || mode > OFF) mode = AUTO;
        for (int p = 0; p < PLAYERS; p++) playerInput[p] = -1;
    }

    /** Host calls after System.loadLibrary(), before launching the emulated game. */
    public boolean start() {
        synchronized (this) {
            if (closed || started) return false;
        }
        boolean permitted = app.getPackageManager().checkPermission(
                android.Manifest.permission.VIBRATE, app.getPackageName()) == PackageManager.PERMISSION_GRANTED;
        if (!nativeStart(this, permitted)) return false;
        synchronized (this) { started = true; }
        if (input != null) input.registerInputDeviceListener(this, null);
        refresh();
        nativeGain(gain);
        return true;
    }

    /** The input layer must call this whenever it assigns an Android gamepad to an NX player. */
    public synchronized void setPlayerInputDevice(int player, int androidDeviceId) {
        if (player < 0 || player >= PLAYERS || closed) return;
        playerInput[player] = androidDeviceId;
        refresh();
    }
    public synchronized void setOutputMode(int outputMode) {
        if (outputMode < AUTO || outputMode > OFF || closed) return;
        mode = outputMode;
        preferences.edit().putInt("mode", mode).apply();
        refresh();
    }
    public synchronized void setEnabled(boolean on) {
        enabled = on;
        preferences.edit().putBoolean("enabled", on).apply();
        refresh();
    }
    public synchronized void setStrength(float strength) {
        gain = Float.isFinite(strength) ? Math.max(0, Math.min(3, strength)) : 0;
        preferences.edit().putFloat("gain", gain).apply();
        if (started && !closed) nativeGain(gain);
    }

    /** Activity.onPause/onStop and onWindowFocusChanged(false); clears persistent guest state. */
    public synchronized void pause() {
        paused = true;
        for (Route r : routes) if (r != null) r.cancel();
        if (started && !closed) nativeSuspend(INACTIVE, true);
    }
    /** Activity.onResume or restored focus. No previous game vibration is replayed. */
    public synchronized void resume() {
        paused = false;
        if (started && !closed) {
            refresh();
            nativeSuspend(INACTIVE, false);
        }
    }
    public synchronized void setInterrupted(boolean on) {
        interrupted = on;
        if (on) for (Route r : routes) if (r != null) r.cancel();
        if (started && !closed) nativeSuspend(INTERRUPTED, on);
    }

    /** Executes under the Java routing lock on the dedicated native dispatcher thread. */
    @SuppressWarnings("unused")
    private synchronized void applyFromNative(int player, int token, long epoch,
                                              int left, int right, int mono) {
        if (closed || paused || interrupted || !enabled || player < 0 || player >= PLAYERS) return;
        Route r = routes[player];
        // A queued STOP for a previous token may not cancel an entirely new output.
        if (r == null || r.token != token) return;
        if (epoch == 0 && left == 0 && right == 0 && mono == 0) { r.cancel(); return; }
        if (epoch < r.lastEpoch) return;
        if (epoch > r.lastEpoch) { r.cancel(); r.lastEpoch = epoch; }
        // Android 14 baseline: mono amplitude approximation. Left and right are retained
        // in the native contract for validated independent actuators on later devices.
        try { r.play(mono); }
        catch (RuntimeException e) {
            Log.e(TAG, "[RED] vibration output failed for player " + player, e);
            r.cancel();
        }
    }

    private Route builtIn(int token) {
        if (Build.VERSION.SDK_INT >= 31) {
            VibratorManager manager = (VibratorManager) app.getSystemService(Context.VIBRATOR_MANAGER_SERVICE);
            if (manager == null) return null;
            int[] ids = manager.getVibratorIds();
            if (ids.length == 0) return null;
            boolean amplitude = true, has = false;
            for (int id : ids) {
                Vibrator v = manager.getVibrator(id);
                if (v.hasVibrator()) { has = true; amplitude &= v.hasAmplitudeControl(); }
            }
            return has ? new Route(token, -1, "builtin", manager, null, amplitude) : null;
        }
        Vibrator vibrator = (Vibrator) app.getSystemService(Context.VIBRATOR_SERVICE);
        return vibrator != null && vibrator.hasVibrator() ?
                new Route(token, -1, "builtin", null, vibrator, vibrator.hasAmplitudeControl()) : null;
    }
    private Route controller(int id, int token) {
        if (id < 0 || Build.VERSION.SDK_INT < 31) return null;
        InputDevice device = InputDevice.getDevice(id);
        if (device == null) return null;
        int sources = device.getSources();
        if ((sources & InputDevice.SOURCE_GAMEPAD) != InputDevice.SOURCE_GAMEPAD &&
            (sources & InputDevice.SOURCE_JOYSTICK) != InputDevice.SOURCE_JOYSTICK) return null;
        VibratorManager manager = device.getVibratorManager();
        if (manager == null || manager.getVibratorIds().length == 0) return null;
        boolean amplitude = true, has = false;
        for (int vibratorId : manager.getVibratorIds()) {
            Vibrator v = manager.getVibrator(vibratorId);
            if (v.hasVibrator()) { has = true; amplitude &= v.hasAmplitudeControl(); }
        }
        String identity = "controller:" + device.getDescriptor() + ":" + device.getVendorId() + ":" + device.getProductId();
        return has ? new Route(token, id, identity, manager, null, amplitude) : null;
    }
    private Route resolve(int player, int token) {
        if (!enabled || mode == OFF) return null;
        Route selected = null;
        if (mode != BUILTIN) selected = controller(playerInput[player], token);
        if (selected == null && player == 0 && mode != CONTROLLER) selected = builtIn(token);
        // Do not drive the same controller/output for two players.
        if (selected != null) for (int p = 0; p < player; p++) {
            Route other = routes[p];
            if (other != null && other.identity.equals(selected.identity)) return null;
        }
        return selected;
    }
    /** Called on hardware changes, gamepad assignment or settings changes. */
    public synchronized void refresh() {
        if (!started || closed) return;
        for (int p = 0; p < PLAYERS; p++) {
            Route old = routes[p];
            Route next = resolve(p, nextToken++);
            boolean same = old != null && next != null && old.identity.equals(next.identity) &&
                           old.amplitude == next.amplitude && old.deviceId == next.deviceId;
            if (same) continue;
            if (old != null) old.cancel();
            routes[p] = next;
            nativeRoute(p, next != null ? next.token : 0, next != null);
        }
    }
    public synchronized String capabilities() {
        StringBuilder out = new StringBuilder("Android SDK ").append(Build.VERSION.SDK_INT);
        out.append("\nBuild: ").append(Build.DISPLAY);
        out.append("\nRumble: ").append(enabled ? "ON" : "OFF")
           .append(", mode=").append(mode).append(", strength=").append(gain);
        if (Build.VERSION.SDK_INT >= 31) {
            VibratorManager manager = (VibratorManager) app.getSystemService(Context.VIBRATOR_MANAGER_SERVICE);
            if (manager != null) {
                out.append("\nInternal vibrator IDs:");
                for (int id : manager.getVibratorIds()) {
                    Vibrator v = manager.getVibrator(id);
                    out.append(" ").append(id).append("[present=").append(v.hasVibrator())
                       .append(",amplitude=").append(v.hasAmplitudeControl()).append("]");
                }
            }
        }
        if (input != null) for (int id : input.getInputDeviceIds()) {
            InputDevice d = input.getInputDevice(id);
            if (d == null || ((d.getSources() & InputDevice.SOURCE_GAMEPAD) != InputDevice.SOURCE_GAMEPAD &&
                              (d.getSources() & InputDevice.SOURCE_JOYSTICK) != InputDevice.SOURCE_JOYSTICK)) continue;
            out.append("\nGamepad ").append(id).append(" descriptor-hash=")
               .append(d.getDescriptor() == null ? 0 : d.getDescriptor().hashCode());
            if (Build.VERSION.SDK_INT >= 31) {
                VibratorManager cm = d.getVibratorManager();
                out.append(", motors=");
                if (cm != null) for (int motorId : cm.getVibratorIds()) {
                    Vibrator motor = cm.getVibrator(motorId);
                    out.append(motorId).append("(").append(motor.hasAmplitudeControl()).append(") ");
                }
            }
        }
        for (int p = 0; p < PLAYERS; p++) {
            Route r = routes[p];
            out.append("\nPlayer ").append(p + 1).append(": ");
            out.append(r == null ? "unavailable" :
                    (r.deviceId < 0 ? "built-in" : "controller #" + r.deviceId) +
                    (r.amplitude ? ", amplitude" : ", binary"));
        }
        if (started) out.append("\nNative queue overflows: ").append(nativeOverflows());
        return out.toString();
    }
    @Override public void onInputDeviceAdded(int deviceId) { refresh(); }
    @Override public void onInputDeviceRemoved(int deviceId) { refresh(); }
    @Override public void onInputDeviceChanged(int deviceId) { refresh(); }

    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            for (int p = 0; p < PLAYERS; p++) {
                if (routes[p] != null) routes[p].cancel();
                routes[p] = null;
            }
        }
        if (input != null && started) input.unregisterInputDeviceListener(this);
        // The Java routing lock is released before nativeStop joins the JNI worker.
        if (started) nativeStop();
    }

    private static native boolean nativeStart(HapticsRouter router, boolean enabled);
    private static native void nativeRoute(int player, int token, boolean available);
    private static native void nativeGain(float gain);
    private static native void nativeSuspend(int reason, boolean on);
    private static native void nativeStop();
    private static native long nativeOverflows();
}
