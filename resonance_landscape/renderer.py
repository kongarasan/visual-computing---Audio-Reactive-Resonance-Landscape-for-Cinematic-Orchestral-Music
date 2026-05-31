"""
ModernGL renderer for the Audio-Reactive Resonance Landscape.

Scene (styled to match the reference artwork):
  - a single thin neon ring, blue at the top fading to magenta at the bottom
  - a brilliant central light flare on the horizon with a horizontal lens streak
  - rounded wireframe-mesh "mountain" waves that flow across the whole width
    (low ripples through the center, taller rounded peaks on the sides),
    blue mesh net + magenta crest rims + sparkle points, on a black field
  - two side clusters of thin neon equalizer bars (FFT spectrum)
  - a reflective floor that mirrors everything with concentric ripples
  - pure black (0,0,0) background everywhere else

Per-frame audio features arrive as uniforms; spectrum / waveform as 1D textures.
Runs headless via a standalone OpenGL context (EGL) or the native display.

Course 6178S - Seminar in Visual Computing, Group 08
"""

from __future__ import annotations

import numpy as np
import moderngl


VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 uv;
out vec4 frag;

uniform float u_time;
uniform vec2  u_res;
uniform float u_amp;
uniform float u_bass;
uniform float u_onset;
uniform int   u_nbands;
uniform sampler2D u_spectrum;
uniform sampler2D u_waveform;

const float PI = 3.14159265;
const vec3 BLUE = vec3(0.15, 0.45, 1.00);
const vec3 CYAN = vec3(0.30, 0.85, 1.00);
const vec3 PINK = vec3(1.00, 0.18, 0.95);
const vec3 PURP = vec3(0.55, 0.25, 1.00);

float band(sampler2D tex, float x01) {
    return texture(tex, vec2(clamp(x01, 0.0, 1.0), 0.5)).r;
}
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float bump(float a, float c, float w) {
    float d = (a - c) / w;
    return exp(-d * d);
}

// rounded mountain profile: low ripples near center, taller peaks on the sides
float mountains(float xn, float t, float react) {
    float a = abs(xn);
    float h = 0.0;
    h += 0.26 * bump(a, 0.52, 0.14);
    h += 0.20 * bump(a, 0.82, 0.13);
    h += 0.13 * bump(a, 0.30, 0.09);
    h += 0.06 * bump(a, 0.12, 0.10);          // low ripple through the ring base
    h += 0.03 * sin(xn * 9.0 + t * 0.6);      // gentle motion
    h = max(h, 0.0);
    h *= (0.55 + 0.95 * react);
    return h;
}

vec3 scene(vec2 p) {
    float ar = u_res.x / u_res.y;
    float xn = p.x / ar;
    vec3 col = vec3(0.0);

    // ===== side equalizer bars (two clusters only) =======================
    {
        float a = abs(xn);
        float cluster = step(0.17, a) * step(a, 0.46);   // only this band
        if (cluster > 0.5 && p.y >= 0.0) {
            float n = 12.0;                               // fewer, wider bars
            float fb = (a - 0.17) / (0.46 - 0.17) * n;    // index within cluster
            int bi = int(floor(fb));
            float h = band(u_spectrum, (float(bi) + 0.5) / n);
            h = 0.10 + h * 0.62;                          // a touch taller
            float fx = fract(fb);
            float barw = step(0.12, fx) * step(fx, 0.88); // wider, proper rectangles
            float inbar = step(p.y, h) * barw;
            float tip = exp(-abs(p.y - h) * 9.0) * barw;
            vec3 bcol = mix(BLUE, PINK, clamp(p.y / max(h, 1e-3), 0.0, 1.0));
            col += bcol * (inbar * 0.75 + tip * 0.9);      // more solid fill
        }
    }

    // ===== rounded mesh-net mountains (flow through the whole width) ======
    {
        float react = band(u_waveform, abs(xn));
        float h = mountains(xn, u_time, react) * (0.75 + 0.5 * u_amp);
        if (p.y >= 0.0 && p.y <= h) {
            float edge = h - p.y;                           // 0 at crest
            float crest = exp(-edge * 22.0);                // magenta rim
            // wireframe net: lines that follow the surface
            float lh = 1.0 - abs(fract((h - p.y) * 30.0) - 0.5) * 2.0;
            float lv = 1.0 - abs(fract(p.x * 42.0) - 0.5) * 2.0;
            float mesh = max(smoothstep(0.86, 1.0, lh),
                             smoothstep(0.90, 1.0, lv));
            float spk = step(0.95, hash2(floor(vec2(p.x * 80.0, p.y * 80.0))));
            // body stays mostly dark (lets black show through), net is bright
            vec3 net = mix(BLUE, CYAN, clamp(p.y / max(h, 1e-3), 0.0, 1.0));
            col += net * mesh * 1.2;
            col += BLUE * 0.05;                             // faint body tint
            col += PINK * crest * 1.4;                      // bright crest rim
            col += CYAN * spk * 1.0;                        // sparkle points
        }
    }

    // ===== neon ring (blue top -> magenta bottom) ========================
    {
        float ringCY = 0.40;
        float ringR  = 0.52 + 0.015 * u_bass;
        float d = abs(length(vec2(p.x, p.y - ringCY)) - ringR);
        float core = exp(-d * 150.0);
        float bloom = exp(-d * 34.0) * 0.22;
        float ang = atan(p.y - ringCY, p.x);
        float f = sin(ang) * 0.5 + 0.5;                     // 1 top, 0 bottom
        vec3 rcol = mix(PINK, CYAN, f);
        col += rcol * (core * (1.3 + 0.7 * u_onset) + bloom) * (1.0 + 0.4 * u_bass);
    }

    // ===== central light flare + horizontal lens streak ==================
    {
        float r = length(p);
        float core = exp(-r * 34.0);
        float glow = exp(-r * 8.0) * (0.30 + 0.35 * u_bass);
        float streak = exp(-abs(p.y) * 80.0) * exp(-abs(p.x) * 2.0) * 0.8;
        col += vec3(0.85, 0.93, 1.0) * (core * (1.6 + 2.5 * u_onset) + glow);
        col += CYAN * streak * (0.9 + 0.6 * u_amp);
    }

    return col;
}

void main() {
    vec2 aspect = vec2(u_res.x / u_res.y, 1.0);
    vec2 p = (uv - 0.5) * 2.0 * aspect;
    p.y += 0.30;                                // horizon in lower third

    vec3 col;
    if (p.y >= 0.0) {
        col = scene(p);
    } else {
        vec2 m = vec2(p.x, -p.y);
        vec3 refl = scene(m);
        float depth = -p.y;
        float fade = exp(-depth * 2.0);
        float ripple = 1.0 + 0.05 * sin(p.x * 28.0 + u_time * 2.0) * depth;
        col = refl * fade * 0.55 * ripple;
        // concentric ripple rings radiating from the center on the floor
        float fr = length(vec2(p.x, depth * 2.4));
        float rip = 0.5 + 0.5 * sin(fr * 44.0 - u_time * 3.0);
        col += CYAN * smoothstep(0.80, 1.0, rip) * exp(-fr * 3.0) * 0.4;
    }

    // vivid neon: gentle tonemap that keeps saturation, pure black stays black
    col *= (0.9 + 0.4 * u_amp);
    col = col / (col + vec3(0.7));
    col = pow(col, vec3(0.80));
    frag = vec4(col, 1.0);
}
"""


class ResonanceRenderer:
    def __init__(self, width=1280, height=720, n_bands=48):
        self.width = width
        self.height = height
        self.n_bands = n_bands

        self.ctx = self._make_context()
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )

        self.tex_spectrum = self.ctx.texture((n_bands, 1), 1, dtype="f4")
        self.tex_waveform = self.ctx.texture((n_bands, 1), 1, dtype="f4")
        for t in (self.tex_spectrum, self.tex_waveform):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False

        self._set("u_res", (float(width), float(height)))
        self._set("u_nbands", n_bands)
        self._set("u_spectrum", 0)
        self._set("u_waveform", 1)

    def _set(self, name, value):
        """Set a uniform if it exists (the GLSL compiler may drop unused ones)."""
        try:
            self.prog[name].value = value
        except KeyError:
            pass

    @staticmethod
    def _make_context():
        errors = []
        for kwargs in ({"backend": "egl"}, {}):
            label = kwargs.get("backend", "default")
            try:
                return moderngl.create_standalone_context(**kwargs)
            except Exception as e:
                errors.append(label + ": " + str(e))
        raise RuntimeError(
            "Could not create an OpenGL context. On a headless server install "
            "EGL/Mesa (e.g. apt install libegl1 libgl1-mesa-dri). Tried: "
            + " | ".join(errors)
        )

    def render(self, t, amp, bass, onset, spectrum, waveform):
        self.tex_spectrum.write(np.ascontiguousarray(spectrum, dtype="f4").tobytes())
        self.tex_waveform.write(np.ascontiguousarray(waveform, dtype="f4").tobytes())
        self.tex_spectrum.use(location=0)
        self.tex_waveform.use(location=1)

        self._set("u_time", float(t))
        self._set("u_amp", float(amp))
        self._set("u_bass", float(bass))
        self._set("u_onset", float(onset))

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        data = self.fbo.read(components=3, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(img).copy()

    def release(self):
        for o in (self.vao, self.vbo, self.tex_spectrum, self.tex_waveform,
                  self.fbo, self.prog, self.ctx):
            try:
                o.release()
            except Exception:
                pass
