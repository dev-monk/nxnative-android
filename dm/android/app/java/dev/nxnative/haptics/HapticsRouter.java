package dev.nxnative.haptics;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.SharedPreferences;
import android.hardware.input.InputManager;
import android.media.AudioAttributes;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
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
    // Games update vibration every few milliseconds (TTYD: every 5 ms). Each vibrate() restarts the effect, and a Bluetooth controller re-sent one every 5 ms never spins up
    // (felt on an Xbox controller, RG 476H). So at most one vibrate() per SEND_MS, the strength in steps of STEP; a stop goes at once
    private static final long SEND_MS = 40;
    private static final int STEP = 32;
    private HandlerThread pacer;
    private Handler pace;
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
        // Send pacing (deliver): the latest strength waiting for its turn, -1 none; when the last one went out; whether a flush is posted
        int pending = -1;
        long lastSent = -SEND_MS;
        boolean scheduled;
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
            pending = -1;
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
        pacer = new HandlerThread("rumble-pace");
        pacer.start();
        pace = new Handler(pacer.getLooper());
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
        if (closed || paused || interrupted || !enabled || player < 0 || player >= PLAYERS) {
            drop("closed=" + closed + " paused=" + paused + " interrupted=" + interrupted + " enabled=" + enabled + " player=" + player);
            return;
        }
        Route r = routes[player];
        // A queued STOP for a previous token may not cancel an entirely new output.
        if (r == null || r.token != token) { drop("route " + (r == null ? "none" : "token " + r.token) + ", delivery for token " + token); return; }
        if (epoch == 0 && left == 0 && right == 0 && mono == 0) { r.cancel(); return; }
        if (epoch < r.lastEpoch) return;
        if (epoch > r.lastEpoch) { r.cancel(); r.lastEpoch = epoch; }
        // Android 14 baseline: mono amplitude approximation. Left and right are retained
        // in the native contract for validated independent actuators on later devices.
        deliver(r, mono);
    }
    // Under the routing lock. A strength change within SEND_MS of the last send waits for a flush, which sends the strongest one since
    private void deliver(Route r, int mono) {
        int q = mono <= 0 ? 0 : r.amplitude ? Math.min(255, Math.max(STEP, (mono + STEP / 2) / STEP * STEP)) : mono;
        if (q == 0) { r.cancel(); return; }
        long now = SystemClock.uptimeMillis();
        if (!r.scheduled && now - r.lastSent >= SEND_MS) { send(r, q, now); return; }
        r.pending = Math.max(r.pending, q);   // the strongest of the wait: a peak shorter than SEND_MS (a menu tick) is still felt
        if (r.scheduled || pace == null) return;
        r.scheduled = true;
        final Route route = r;
        pace.postAtTime(new Runnable() { public void run() { flush(route); } }, r.lastSent + SEND_MS);
    }
    private synchronized void flush(Route r) {
        r.scheduled = false;
        int q = r.pending; r.pending = -1;
        boolean current = false;
        for (Route x : routes) if (x == r) current = true;
        if (q < 0 || !current || closed || paused || interrupted || !enabled) return;
        send(r, q, SystemClock.uptimeMillis());
    }
    private void send(Route r, int q, long now) {
        r.lastSent = now;
        if (drops < 64) { drops++; Log.d(TAG, "send " + q + " on " + r.identity + (r.amplitude ? " (amplitude)" : " (binary)")); }
        try { r.play(q); }
        catch (RuntimeException e) {
            Log.e(TAG, "[RED] vibration output failed on " + r.identity, e);
            r.cancel();
        }
    }

    private int drops;   // the first 64 drops and plays go to logcat (NxHaptics)
    private void drop(String why) { if (drops < 64) { drops++; Log.d(TAG, "dropped: " + why); } }

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
    /** Diagnostic: three buzzes on player's route, 1.5 s apart (a one-shot at the default strength; one held waveform at full strength; the waveform re-sent every 5 ms as the
     *  game does). Runs on its own thread; each step goes to out. */
    public void selfTest(final int player, final java.util.function.Consumer<String> out) {
        new Thread(new Runnable() { public void run() {
            try {
                Thread.sleep(3000);
                Route r; synchronized (HapticsRouter.this) { r = routes[player]; }
                if (r == null) { out.accept("self-test: player " + (player + 1) + " has no route"); return; }
                out.accept("self-test 1/3: one-shot 800 ms, default strength, on " + r.identity);
                VibrationEffect one = VibrationEffect.createOneShot(800, VibrationEffect.DEFAULT_AMPLITUDE);
                if (r.manager != null) r.manager.vibrate(CombinedVibration.createParallel(one)); else r.vibrator.vibrate(one);
                Thread.sleep(1500); r.cancel();
                out.accept("self-test 2/3: the router's held waveform at 255 for 800 ms");
                synchronized (HapticsRouter.this) { r.lastAmplitude = -1; r.play(255); }
                Thread.sleep(800); synchronized (HapticsRouter.this) { r.cancel(); }
                Thread.sleep(700);
                out.accept("self-test 3/3: a strength every 5 ms for 800 ms, 255 falling to 100, as the game sends, through the send pacing");
                for (int i = 0; i < 160; i++) { synchronized (HapticsRouter.this) { deliver(r, 255 - i); } Thread.sleep(5); }
                synchronized (HapticsRouter.this) { r.cancel(); }
                out.accept("self-test done");
            } catch (Throwable t) { out.accept("self-test failed: " + t); }
        } }, "rumble-selftest").start();
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
        if (pacer != null) pacer.quitSafely();
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
