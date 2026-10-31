package dev.nxnative.probe;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.graphics.Color;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import dev.nxnative.haptics.HapticsRouter;

/** Hardware capability / repeat / stop regression probe. No game files required. */
public final class MainActivity extends Activity {
    static { System.loadLibrary("nxrumble_probe"); }
    private static native void nativeSend(int motor, float amplitude);
    private static native void nativeBoth(float amplitude);
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Runnable stopTask = () -> nativeBoth(0);
    private HapticsRouter haptics;
    private TextView diagnostic;
    private float level = .65f;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        haptics = new HapticsRouter(this);
        ScrollView scroll = new ScrollView(this);
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(26, 26, 26, 26);
        scroll.addView(body);
        TextView heading = new TextView(this);
        heading.setText("NXNative Android Rumble Probe");
        heading.setTextSize(23);
        heading.setTextColor(Color.BLACK);
        body.addView(heading);
        diagnostic = new TextView(this);
        diagnostic.setTextIsSelectable(true);
        diagnostic.setPadding(0, 18, 0, 18);
        body.addView(diagnostic);
        SeekBar strength = new SeekBar(this);
        strength.setProgress(65); strength.setMax(100);
        strength.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar bar, int progress, boolean fromUser) {
                level = progress / 100f;
            }
            @Override public void onStartTrackingTouch(SeekBar bar) { }
            @Override public void onStopTrackingTouch(SeekBar bar) { }
        });
        body.addView(strength);
        button(body, "Left motor (200 ms)", v -> pulse(0));
        button(body, "Right motor (200 ms)", v -> pulse(1));
        button(body, "Both motors (2 sec)", v -> {
            handler.removeCallbacks(stopTask); nativeBoth(level);
            handler.postDelayed(stopTask, 2000);
        });
        button(body, "Sustained ON (must stop manually)", v -> {
            handler.removeCallbacks(stopTask); nativeBoth(level);
        });
        button(body, "STOP / cancel", v -> stop());
        button(body, "Refresh capability report", v -> updateReport());
        button(body, "Use built-in motor", v -> {
            stop(); haptics.setOutputMode(HapticsRouter.BUILTIN); updateReport();
        });
        button(body, "Use assigned controller", v -> {
            stop(); haptics.setOutputMode(HapticsRouter.CONTROLLER); updateReport();
        });
        button(body, "Use auto route", v -> {
            stop(); haptics.setOutputMode(HapticsRouter.AUTO); updateReport();
        });
        button(body, "Assign first connected gamepad to P1", v -> {
            for (int id : android.view.InputDevice.getDeviceIds()) {
                android.view.InputDevice d = android.view.InputDevice.getDevice(id);
                if (d != null && (d.getSources() & android.view.InputDevice.SOURCE_GAMEPAD) == android.view.InputDevice.SOURCE_GAMEPAD) {
                    stop(); haptics.setPlayerInputDevice(0, id); break;
                }
            }
            updateReport();
        });
        setContentView(scroll);
        if (!haptics.start()) diagnostic.setText("Failed to start JNI haptics engine");
        else updateReport();
    }
    private void button(LinearLayout root, String label, View.OnClickListener listener) {
        Button b = new Button(this); b.setText(label); b.setOnClickListener(listener); root.addView(b);
    }
    private void pulse(int motor) {
        handler.removeCallbacks(stopTask); nativeBoth(0); nativeSend(motor, level);
        handler.postDelayed(stopTask, 200);
    }
    private void stop() { handler.removeCallbacks(stopTask); nativeBoth(0); }
    private void updateReport() { diagnostic.setText(haptics.capabilities()); }
    @Override protected void onPause() {
        stop(); haptics.pause(); super.onPause();
    }
    @Override protected void onResume() {
        super.onResume(); if (haptics != null) haptics.resume();
    }
    @Override protected void onDestroy() {
        stop(); haptics.close(); handler.removeCallbacksAndMessages(null);
        super.onDestroy();
    }
}
