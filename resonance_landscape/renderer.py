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
uniform float u_melody;
uniform float u_treble;
uniform float u_onset;
uniform int   u_nbands;
uniform sampler2D u_spectrum;
uniform sampler2D u_waveform;

const float PI   = 3.14159265;
const vec3  BLUE = vec3(0.08, 0.30, 1.00);
const vec3  CYAN = vec3(0.18, 0.72, 1.00);
const vec3  PINK = vec3(0.92, 0.08, 0.92);
const vec3  PURP = vec3(0.48, 0.12, 0.95);
const vec3  WHITE = vec3(0.92, 0.96, 1.00);

float band(sampler2D tex, float x01) {
    return texture(tex, vec2(clamp(x01, 0.0, 1.0), 0.5)).r;
}
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float bump(float a, float c, float w) { float d = (a-c)/w; return exp(-d*d); }

// Smooth wave mountains that TRAVEL inward toward the ring, shrinking to a
// line at the glowing center -- like driving through a cave / archway. The
// rounded ridges scroll from the outer edges in toward the middle each frame.
float mountains(float xn, float t, float react) {
    float a = abs(xn);
    // rising envelope: 0 at center (a line) -> full toward the edges
    float env  = smoothstep(0.03, 0.42, a);
    // amplitude grows outward (small near the ring, big at the edges)
    float grow = clamp(a, 0.0, 1.15);

    // traveling ridge field: argument (a*k + t*speed) makes the pattern
    // scroll toward smaller a, i.e. the waves flow inward into the ring.
    float k     = 18.0;     // ~5 rounded ridges per side
    float speed = 1.6;      // inward flow speed
    float ridge = 0.5 + 0.5 * cos(a * k + t * speed);
    ridge = pow(ridge, 1.4);                       // round the crests
    float h = ridge * (0.08 + 0.30 * grow);

    // slower second layer for an organic, rolling-wave feel
    h += (0.5 + 0.5 * cos(a * k * 0.5 + t * speed * 0.6)) * 0.05 * grow;

    h *= env;                                      // taper to a line at center
    // gentle beat swell, also enveloped so the center stays flat
    h += 0.10 * u_onset * env * bump(a, 0.70, 0.35);
    h = max(h, 0.0);
    h *= (0.60 + 0.90 * react) * (0.65 + 0.55 * u_amp);
    return h;
}

vec3 scene(vec2 p, float ar) {
    float xn  = p.x / ar;   // aspect-corrected x, range +-1
    vec3  col = vec3(0.0);

    // Pre-compute mountain height at this x for bar occlusion
    float mReact = band(u_waveform, abs(xn));
    float mHeight = mountains(xn, u_time, mReact);

    // === Soft blue ambient glow (inside ring area) =======================
    {
        float dist = length(vec2(xn, p.y - 0.30));
        col += BLUE * exp(-dist * 1.6) * 0.20 * (0.4 + 0.6 * u_bass);
    }

    // === Equalizer bars — span the mountain region; mountains occlude them
    //     so a cluster of bars shows in each valley between peaks ==========
    {
        float a = abs(xn);
        if (a >= 0.12 && a <= 0.96 && p.y >= mHeight) {
            float n      = 24.0;                       // bars across the whole span
            float fb     = (a - 0.12) / (0.96 - 0.12) * n;
            int   bi     = int(floor(fb));
            float specIdx = (float(bi) + 0.5) / n;
            float h      = band(u_spectrum, specIdx);
            // shrink bars near the ring, grow them toward the outer edges
            float barEnv = smoothstep(0.12, 0.45, a);
            h = (0.06 + h * 0.42) * barEnv;
            float fx     = fract(fb);
            // crisp rectangle: gap between bars, hard top edge
            float barw   = step(0.18, fx) * step(fx, 0.82);
            float inbar  = step(p.y, h) * barw;
            // flat solid CYAN — no gradient, no tip glow
            col += CYAN * inbar * 0.75;
        }
    }

    // === Mesh-net mountains (full width, drawn after bars to cover them) =
    {
        if (p.y >= 0.0 && p.y <= mHeight) {
            float h     = mHeight;
            float edge  = h - p.y;
            float crest = exp(-edge * 30.0);
            // wireframe: iso-height lines + vertical lines
            float lh = 1.0 - abs(fract((h - p.y) * 26.0) - 0.5) * 2.0;
            float lv = 1.0 - abs(fract(p.x * 38.0) - 0.5) * 2.0;
            float mesh = max(smoothstep(0.85, 1.0, lh),
                             smoothstep(0.88, 1.0, lv));
            float spk = step(0.965, hash2(floor(vec2(p.x * 72.0, p.y * 72.0))));
            // Opaque black body first — this occludes any bars underneath
            col = vec3(0.0);
            // mesh net: deep purple at the base -> blue -> cyan at the crest
            float up = clamp(p.y / max(h, 1e-3), 0.0, 1.0);
            vec3 netcol = mix(mix(PURP, BLUE, up), CYAN, up * up);
            col += netcol * mesh * 1.15;
            col += PURP * 0.12;                              // richer purple body fill
            col += PINK * crest * (1.5 + 0.9 * u_onset);
            col += CYAN * spk * (0.9 + 0.5 * u_treble);
        }
    }

    // === Large neon ring (cyan top -> magenta bottom, true circle) =======
    {
        float ringCY = 0.30;
        float ringR  = 0.42 + 0.012 * u_bass + 0.007 * u_onset;
        // aspect-corrected distance for a proper circle
        float dist  = length(vec2(xn, p.y - ringCY));
        float d     = abs(dist - ringR);
        float core  = exp(-d * 170.0);
        float bloom = exp(-d * 38.0) * 0.28;
        float ang   = atan(p.y - ringCY, xn);
        float f     = sin(ang) * 0.5 + 0.5;    // 1 = top (cyan), 0 = bottom (pink)
        vec3  rcol  = mix(PINK, CYAN, f);
        float pulse = 1.2 + 0.8 * u_onset + 0.3 * u_melody;
        col += rcol * (core * pulse + bloom) * (1.0 + 0.38 * u_bass);
    }

    // === Central light flare + horizontal + vertical streaks =============
    {
        float r = length(vec2(xn, p.y));
        float core    = exp(-r * 38.0);
        float glow    = exp(-r * 9.0) * (0.22 + 0.32 * u_bass);
        float hstreak = exp(-abs(p.y) * 88.0) * exp(-abs(xn) * 2.2) * 0.75;
        float vstreak = exp(-abs(xn) * 130.0) * exp(-max(p.y, 0.0) * 4.5) * 0.45;
        col += WHITE * (core * (1.7 + 2.4 * u_onset) + glow);
        col += CYAN  * hstreak * (0.85 + 0.5 * u_amp);
        col += WHITE * vstreak * (0.55 + 0.4 * u_onset);
    }

    return col;
}

void main() {
    float ar = u_res.x / u_res.y;
    vec2  p  = (uv - 0.5) * 2.0 * vec2(ar, 1.0);
    p.y     += 0.28;    // horizon in lower third

    vec3 col;
    if (p.y >= 0.0) {
        col = scene(p, ar);
    } else {
        // Mirror reflection
        vec2  m     = vec2(p.x, -p.y);
        vec3  refl  = scene(m, ar);
        float depth = -p.y;
        float fade  = exp(-depth * 2.2);
        float ripple = 1.0 + 0.04 * sin(p.x * 30.0 + u_time * 2.2) * depth;
        col = refl * fade * 0.52 * ripple;

        // Perspective grid on floor (horizontal + vertical lines)
        float xn = p.x / ar;
        float gz = fract(depth * 12.0 / (depth + 0.25));
        float gx = fract(xn * 5.5);
        float gridH = smoothstep(0.88, 1.0, 1.0 - abs(gz * 2.0 - 1.0));
        float gridV = smoothstep(0.90, 1.0, 1.0 - abs(gx * 2.0 - 1.0));
        col += PURP * max(gridH, gridV) * exp(-depth * 3.2) * 0.40;  // stronger purple grid
        col += PURP * exp(-depth * 2.5) * 0.06;                      // faint purple floor wash

        // Concentric ripple rings from center
        float fr  = length(vec2(xn, depth * 2.2));
        float rip = 0.5 + 0.5 * sin(fr * 42.0 - u_time * 3.0);
        col += CYAN * smoothstep(0.82, 1.0, rip) * exp(-fr * 3.8) * 0.32;
    }

    // Neon tonemap: vivid saturation, pure black stays black
    col *= (0.85 + 0.45 * u_amp);
    col  = col / (col + vec3(0.65));
    col  = pow(col, vec3(0.78));
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
        self._set("u_melody", 0.0)
        self._set("u_treble", 0.0)

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

    def render(self, t, amp, bass, melody, treble, onset, spectrum, waveform):
        self.tex_spectrum.write(np.ascontiguousarray(spectrum, dtype="f4").tobytes())
        self.tex_waveform.write(np.ascontiguousarray(waveform, dtype="f4").tobytes())
        self.tex_spectrum.use(location=0)
        self.tex_waveform.use(location=1)

        self._set("u_time", float(t))
        self._set("u_amp", float(amp))
        self._set("u_bass", float(bass))
        self._set("u_melody", float(melody))
        self._set("u_treble", float(treble))
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
