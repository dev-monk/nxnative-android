// The app's activity (dm/android/app, docs/android/PLAN.md 1.6): NativeActivity, which runs libdmhost.so (dm/android/front/front_android.c), plus the software keyboard's input box and the
// touch gamepad (TouchPad, a window of its own over this one).
// front_text_input (front.h) calls showTextInput on a host-owned thread; the box is an AlertDialog with an EditText, so the system keyboard types into it as into any text field (its committed
// text, composition and suggestions included), and its answer goes back through nativeTextDone, which front_android.c registers in onCreate. The box keeps OK disabled until the text reaches
// the game's minimum length, as the Switch applet does (front_mac.m brings its box back instead). Rumble: HapticsRouter (dev.nxnative.haptics), started by front_android.c's onCreate through
// startHaptics. Built by build_app.sh (javac, d8) into the APK's classes.dex
package org.nxnative.host;

import android.app.AlertDialog;
import android.app.NativeActivity;
import android.content.DialogInterface;
import android.hardware.input.InputManager;
import android.text.Editable;
import android.text.InputFilter;
import android.text.InputType;
import android.text.TextWatcher;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.TextView;
import dev.nxnative.haptics.HapticsRouter;
import java.nio.charset.StandardCharsets;

public class HostActivity extends NativeActivity {
    // The box's answer, once per box: ok false for a cancel. The text goes as UTF-8, and the strings come in as UTF-8, because JNI's own strings are modified UTF-8
    private static native void nativeTextDone(long token, boolean ok, byte[] utf8);

    private static String str(byte[] b) { return b == null ? "" : new String(b, StandardCharsets.UTF_8); }

    // Any thread (front_android.c calls it from a host-owned one); the box is made on the UI thread. maxChars and minChars count UTF-16 units, 0 for none
    public void showTextInput(final long token, byte[] title, byte[] message, byte[] hint, byte[] initial, final int maxChars, final int minChars) {
        final String t = str(title), m = str(message), h = str(hint), i = str(initial);
        runOnUiThread(new Runnable() { public void run() { show(token, t, m, h, i, maxChars, minChars); } });
    }

    private void show(final long token, String title, String message, String hint, String initial, int maxChars, final int minChars) {
        // The window going away first (finishing, or destroyed before the runnable ran): no box, a cancel (d.show would throw on the UI thread, which nothing here catches)
        if (isFinishing() || isDestroyed()) { nativeTextDone(token, false, null); return; }
        final boolean[] done = { false };
        final EditText f = new EditText(this);
        f.setSingleLine(true);
        f.setInputType(InputType.TYPE_CLASS_TEXT);
        f.setImeOptions(EditorInfo.IME_ACTION_DONE | EditorInfo.IME_FLAG_NO_EXTRACT_UI);   // the box stays in view in landscape, rather than a full-screen editor
        if (maxChars > 0) f.setFilters(new InputFilter[] { new InputFilter.LengthFilter(maxChars) });
        f.setText(initial);
        f.setHint(hint);
        f.selectAll();
        FrameLayout pad = new FrameLayout(this);
        int px = (int) (20 * getResources().getDisplayMetrics().density);
        pad.setPadding(px, px / 2, px, 0);
        pad.addView(f);
        AlertDialog.Builder b = new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
            .setTitle(title.isEmpty() ? "Input" : title)
            .setView(pad)
            .setPositiveButton(android.R.string.ok, null)
            .setNegativeButton(android.R.string.cancel, null);
        if (!message.isEmpty()) b.setMessage(message);
        final AlertDialog d = b.create();
        d.setCanceledOnTouchOutside(false);   // the theme's default closes it on a touch outside: a stray tap on the game would lose the text (the Switch applet has no such way out)
        final Runnable ok = new Runnable() { public void run() {
            if (done[0] || f.length() < minChars) return;
            done[0] = true; d.dismiss();
            nativeTextDone(token, true, f.getText().toString().getBytes(StandardCharsets.UTF_8));
        } };
        // Cancel, back, or the box gone any other way: a cancel, given once
        d.setOnDismissListener(new DialogInterface.OnDismissListener() { public void onDismiss(DialogInterface x) {
            if (done[0]) return;
            done[0] = true; nativeTextDone(token, false, null);
        } });
        f.setOnEditorActionListener(new TextView.OnEditorActionListener() { public boolean onEditorAction(TextView v, int action, KeyEvent e) {
            boolean enter = action == EditorInfo.IME_ACTION_DONE || (e != null && e.getKeyCode() == KeyEvent.KEYCODE_ENTER && e.getAction() == KeyEvent.ACTION_DOWN);
            if (enter) ok.run();
            return enter;
        } });
        d.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE);
        try { d.show(); }
        catch (WindowManager.BadTokenException e) {   // the activity's window went between the check above and here
            if (!done[0]) { done[0] = true; nativeTextDone(token, false, null); }
            return;
        }
        final Button okb = d.getButton(AlertDialog.BUTTON_POSITIVE);
        okb.setOnClickListener(new android.view.View.OnClickListener() { public void onClick(android.view.View v) { ok.run(); } });   // not the builder's: that one would close the box on a short text
        okb.setEnabled(f.length() >= minChars);
        f.addTextChangedListener(new TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int c, int n) {}
            public void onTextChanged(CharSequence s, int a, int c, int n) {}
            public void afterTextChanged(Editable s) { okb.setEnabled(s.length() >= minChars); }
        });
        f.requestFocus();
    }

    // ---- The touch gamepad (TouchPad): a panel window over this one, made once the window is attached (its token) ----
    // The natives front_android.c registers in onCreate (touch_init): the release count, the whole touch state after each touch event, TouchPad's log lines (DM_INPUT_LOG), and a device
    // removed (the controller's sticks, hat and triggers let go)
    static native int nativeTouchEpoch();
    static native void nativeTouchSet(int buttons, int lx, int ly, float[] game, int n);
    static native void nativeTouchLog(byte[] utf8);
    static native void nativeControllerRemoved(int id, boolean last);
    private TouchPad pad;

    @Override public void onAttachedToWindow() {
        super.onAttachedToWindow();
        if (pad == null) pad = TouchPad.attach(this);
    }
    // Focus lost (the input box, the notification shade, another app in multi-window) or paused: the native side has let go of everything (input_release), and the controls show nothing
    // held; the panel lets touches through while the window lacks focus (TouchPad.focus)
    @Override public void onWindowFocusChanged(boolean has) {
        super.onWindowFocusChanged(has);
        if (pad != null) pad.focus(has);
        focused = has; hapticsActive();
    }
    @Override protected void onPause() {
        super.onPause();
        if (pad != null) pad.release();
        resumed = false; hapticsActive();
    }
    @Override protected void onResume() {
        super.onResume();
        resumed = true; hapticsActive();
    }
    // front_android.c's onDestroy ends the process inside super.onDestroy: the motors stop first
    @Override protected void onDestroy() {
        if (haptics != null) { haptics.close(); haptics = null; }
        super.onDestroy();
    }

    // ---- Rumble (docs/android/rumble.md): the game's nn::hid vibration through HapticsRouter, to player 1's controller when it has a motor, else the built-in motor ----
    // front_android.c binds HapticsRouter's natives on hapticsRouterClass (this class loader's), then calls startHaptics, both in onCreate before the host thread starts. nativeHapticsLog: a
    // line in log.txt (rumble: ...)
    static native void nativeHapticsLog(byte[] utf8);
    private HapticsRouter haptics;
    private int hapticsPlayer1 = -2;   // the input device given as player 1's, -1 none, -2 not yet given
    private boolean resumed, focused = true, hapticsOn = true;
    static Class<?> hapticsRouterClass() { return HapticsRouter.class; }
    private static void hlog(String s) { nativeHapticsLog(s.getBytes(StandardCharsets.UTF_8)); }
    boolean startHaptics() {
        try {
            HapticsRouter h = new HapticsRouter(this);
            if (!h.start()) { hlog("[RED] HapticsRouter.start failed (the native dispatcher did not start)"); return false; }
            haptics = h;
            hapticsPick();
            getSystemService(InputManager.class).registerInputDeviceListener(new InputManager.InputDeviceListener() {
                public void onInputDeviceAdded(int id) { hapticsPick(); }
                public void onInputDeviceRemoved(int id) { hapticsPick(); }
                public void onInputDeviceChanged(int id) { hapticsPick(); }
            }, null);
            hlog(h.capabilities());
            if (new java.io.File(getFilesDir(), "rumble_test").exists())   // diagnostic: files/rumble_test present, three buzzes on player 1 three seconds after start
                h.selfTest(0, new java.util.function.Consumer<String>() { public void accept(String l) { hlog(l); } });
            return true;
        } catch (Throwable t) {   // never stops the game: it runs without rumble
            hlog("[RED] rumble could not start: " + t);
            haptics = null;
            return false;
        }
    }
    // Every controller is player 1 (front_android.c): the first non-virtual gamepad with a motor drives player 1's rumble; none, and HapticsRouter's Auto mode uses the built-in motor
    private void hapticsPick() {
        if (haptics == null) return;
        int pick = -1; String name = null;
        for (int id : InputDevice.getDeviceIds()) {
            InputDevice d = InputDevice.getDevice(id);
            if (d == null || d.isVirtual()) continue;
            int s = d.getSources();
            if ((s & InputDevice.SOURCE_GAMEPAD) != InputDevice.SOURCE_GAMEPAD && (s & InputDevice.SOURCE_JOYSTICK) != InputDevice.SOURCE_JOYSTICK) continue;
            if (d.getVibratorManager().getVibratorIds().length == 0) continue;
            pick = id; name = d.getName(); break;
        }
        if (pick == hapticsPlayer1) return;
        hapticsPlayer1 = pick;
        haptics.setPlayerInputDevice(0, pick);
        String r = haptics.capabilities(); int at = r.indexOf("\nPlayer 1: ");
        hlog("player 1's controller: " + (pick < 0 ? "none with a motor" : "#" + pick + " '" + name + "'") + " → " + (at < 0 ? "?" : r.substring(at + 1, r.indexOf('\n', at + 1) < 0 ? r.length() : r.indexOf('\n', at + 1))));
    }
    // Rumble only while resumed and in focus (the input box, the notification shade or another window in front stop it); a game command after that starts it again, nothing is replayed
    private void hapticsActive() {
        boolean on = resumed && focused;
        if (haptics == null || on == hapticsOn) return;
        hapticsOn = on;
        if (on) haptics.resume(); else haptics.pause();
    }
}
