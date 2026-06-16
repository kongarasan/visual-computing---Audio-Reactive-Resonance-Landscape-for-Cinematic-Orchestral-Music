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
uniform float u_flow;     // accumulated inward-flow phase (beat-driven speed)
uniform float u_ripple;   // accumulated floor-ripple phase (bass-driven speed)
uniform int   u_nbands;
uniform sampler2D u_spectrum;
uniform sampler2D u_waveform;

const float PI   = 3.14159265;
const vec3  BLUE = vec3(0.08, 0.30, 1.00);
const vec3  CYAN = vec3(0.18, 0.72, 1.00);
const vec3  PINK = vec3(0.92, 0.08, 0.92);
const vec3  PURP = vec3(0.48, 0.12, 0.95);
const vec3  WHITE = vec3(0.92, 0.96, 1.00);
const vec3  TEAL = vec3(0.10, 0.95, 0.65);   // melody-rich (strings/winds)
const vec3  WARM = vec3(1.00, 0.55, 0.18);   // treble-rich (cymbals/brass)

// Music-driven palette: melody pulls the body toward teal-green, treble warms
// the highlights toward orange/white. Matches the "color follows the music" idea.
vec3 dynBody() { return mix(BLUE, TEAL, clamp(u_melody, 0.0, 1.0) * 0.55); }
vec3 dynCrest(){ return mix(PINK, WARM, clamp(u_treble, 0.0, 1.0) * 0.60); }
vec3 dynHi()   { return mix(CYAN, WHITE, clamp(u_treble, 0.0, 1.0) * 0.50); }

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

    // FORWARD-TUNNEL motion: ridges are born small near the center (the ring)
    // and stream OUTWARD, growing as they sweep toward the screen edges -- the
    // sensation of the camera flying forward into the cave. u_flow is the
    // accumulated, beat-driven phase, so you accelerate on the music.
    float k     = 18.0;     // ~5 rounded ridges per side
    float ridge = 0.5 + 0.5 * cos(a * k - u_flow);
    ridge = pow(ridge, 1.4);                       // round the crests
    float h = ridge * (0.08 + 0.30 * grow);

    // slower second layer for an organic, rolling-wave feel
    h += (0.5 + 0.5 * cos(a * k * 0.5 - u_flow * 0.6)) * 0.05 * grow;

    h *= env;                                      // taper to a line at center
    // gentle beat swell, also enveloped so the center stays flat
    h += 0.10 * u_onset * env * bump(a, 0.70, 0.35);
    h = max(h, 0.0);
    h *= (0.60 + 0.90 * react) * (0.65 + 0.55 * u_amp);
    return h;
}


vec3 scene(vec2 p, float ar, float isRefl) {
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
            // the bar lattice drifts OUTWARD with the tunnel motion (u_flow/k),
            // so bars stream past you as the camera flies forward.
            float drift  = u_flow / 18.0;
            float fb     = (a - 0.12) / (0.96 - 0.12) * n - drift;
            int   bi     = int(floor(fb));
            float specIdx = fract((float(bi) + 0.5) / n);
            float h      = band(u_spectrum, specIdx);
            // shrink bars near the ring, grow them toward the outer edges
            float barEnv = smoothstep(0.12, 0.45, a);
            h = (0.06 + h * 0.42) * barEnv;
            float fx     = fract(fb);
            // crisp rectangle: medium bar width, medium gap, hard top edge
            float barw   = step(0.25, fx) * step(fx, 0.75);
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
            // mesh net: deep purple base -> music-driven body -> bright crest.
            // melody shifts the body teal, treble warms the crest highlight.
            float up = clamp(p.y / max(h, 1e-3), 0.0, 1.0);
            vec3 netcol = mix(mix(PURP, dynBody(), up), dynHi(), up * up);
            col += netcol * mesh * 1.15;
            col += PURP * 0.12;                              // richer purple body fill
            col += dynCrest() * crest * (1.5 + 0.9 * u_onset);
            col += dynHi() * spk * (0.9 + 0.5 * u_treble);
        }
    }

    // === Neon resonance ring (perfect circle) + radiating resonance ========
    {
        float ringCY = 0.30;
        float ringR  = 0.42 + 0.012 * u_bass + 0.007 * u_onset;
        // aspect-corrected distance -> a perfect circle regardless of aspect
        float dist  = length(vec2(xn, p.y - ringCY));
        float rr    = dist - ringR;            // signed distance from the ring
        float d     = abs(rr);
        // crisp, even ring core + soft bloom (uniform all the way around)
        float core  = exp(-d * 190.0);
        float bloom = exp(-d * 40.0) * 0.26;
        float ang   = atan(p.y - ringCY, xn);
        float f     = sin(ang) * 0.5 + 0.5;    // 1 = top, 0 = bottom
        vec3  rcol  = mix(dynCrest(), dynHi(), f);
        float pulse = 1.2 + 0.8 * u_onset + 0.3 * u_melody;
        col += rcol * (core * pulse + bloom) * (1.0 + 0.38 * u_bass);

        // RESONANCE: concentric halos radiating OUTWARD from the ring only
        // (rr > 0), animated with the bass-driven phase (u_ripple) and pulsing
        // on the beat. Skipped in the mirror reflection (isRefl) so it stays in
        // the real space outside the circle.
        float outside = smoothstep(0.0, 0.02, rr);          // 0 inside, 1 outside
        float reson   = 0.5 + 0.5 * sin(rr * 34.0 - u_ripple);
        float resMask = smoothstep(0.55, 1.0, reson) * exp(-rr * 4.5) * outside;
        col += rcol * resMask * (0.16 + 0.40 * u_onset + 0.28 * u_bass)
                    * (1.0 - isRefl);
    }

    // === Central light flare + horizontal + vertical streaks =============
    {
        float r = length(vec2(xn, p.y));
        float core    = exp(-r * 38.0);
        float glow    = exp(-r * 9.0) * (0.22 + 0.32 * u_bass);
        float hstreak = exp(-abs(p.y) * 88.0) * exp(-abs(xn) * 2.2) * 0.75;
        float vstreak = exp(-abs(xn) * 130.0) * exp(-max(p.y, 0.0) * 4.5) * 0.45;
        // flare warms toward orange/white as treble rises
        col += mix(WHITE, WARM, u_treble * 0.35) * (core * (1.7 + 2.4 * u_onset) + glow);
        col += dynHi() * hstreak * (0.85 + 0.5 * u_amp);
        col += WHITE * vstreak * (0.55 + 0.4 * u_onset);

        // Radial speed-streaks shooting OUTWARD from the flare -> forward motion.
        // Streaks are keyed to angle (so they're stable rays) and pulse outward
        // along r with the tunnel phase u_flow; faint, fade near the center.
        float ang2 = atan(p.y, xn);
        float rays = hash2(vec2(floor(ang2 * 12.0), 0.0));        // random ray per sector
        float along = 0.5 + 0.5 * sin(r * 30.0 - u_flow + rays * 6.28);
        float streak = smoothstep(0.75, 1.0, along) * smoothstep(0.05, 0.5, r) * exp(-r * 2.0);
        col += dynHi() * streak * 0.10 * (0.6 + 0.8 * u_bass);
    }

    return col;
}

void main() {
    float ar = u_res.x / u_res.y;
    vec2  p  = (uv - 0.5) * 2.0 * vec2(ar, 1.0);
    p.y     += 0.28;    // horizon in lower third

    vec3 col;
    if (p.y >= 0.0) {
        col = scene(p, ar, 0.0);                 // real scene
    } else {
        // ===== GLASSY MIRROR FLOOR ==============================================
        vec2  m     = vec2(p.x, -p.y);          // clean mirror, no distortion
        vec3  refl  = scene(m, ar, 1.0);         // reflection: no resonance halos
        float depth = -p.y;
        float fade  = exp(-depth * 2.2);
        col = refl * fade * 0.52;

        // Perspective grid on floor (horizontal + vertical lines)
        float xn = p.x / ar;
        float gz = fract(depth * 12.0 / (depth + 0.25));
        float gx = fract(xn * 5.5);
        float gridH = smoothstep(0.88, 1.0, 1.0 - abs(gz * 2.0 - 1.0));
        float gridV = smoothstep(0.90, 1.0, 1.0 - abs(gx * 2.0 - 1.0));
        col += PURP * max(gridH, gridV) * exp(-depth * 3.2) * 0.40;  // purple grid
        col += PURP * exp(-depth * 2.5) * 0.06;                      // faint floor wash

        // Concentric ripple rings from center, travelling outward at the
        // bass-driven speed (u_ripple): race out on heavy bass, crawl when quiet.
        float fr  = length(vec2(xn, depth * 2.2));
        float rip = 0.5 + 0.5 * sin(fr * 42.0 - u_ripple);
        col += CYAN * smoothstep(0.82, 1.0, rip) * exp(-fr * 3.8) * 0.32;
    }

    // Neon tonemap: vivid saturation, pure black stays black
    col *= (0.85 + 0.45 * u_amp);
    col  = col / (col + vec3(0.65));
    col  = pow(col, vec3(0.78));
    frag = vec4(col, 1.0);
}
"""

# Post-process bloom: sample the rendered scene, blur its bright pixels with a
# Gaussian kernel, and add that glow back. This gives the whole scene a soft,
# cinematic neon halo instead of the per-element fake glows.
BLOOM_SHADER = """
#version 330
in vec2 uv;
out vec4 frag;
uniform sampler2D u_scene;
uniform vec2  u_res;
uniform float u_strength;

void main() {
    vec3 base = texture(u_scene, uv).rgb;
    vec2 px   = 1.0 / u_res;
    vec3 bloom = vec3(0.0);
    float total = 0.0;
    // 9x9 Gaussian over the bright parts only (smoothstep threshold)
    for (int j = -4; j <= 4; j++) {
        for (int i = -4; i <= 4; i++) {
            vec2 o = vec2(float(i), float(j)) * px * 2.2;
            vec3 s = texture(u_scene, uv + o).rgb;
            float b = max(s.r, max(s.g, s.b));
            float bright = smoothstep(0.55, 1.0, b);
            float w = exp(-float(i * i + j * j) / 8.0);
            bloom += s * bright * w;
            total += w;
        }
    }
    bloom /= max(total, 1e-3);
    vec3 col = base + bloom * u_strength;
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
        self.prog_bloom = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=BLOOM_SHADER
        )

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")
        self.vao_bloom = self.ctx.simple_vertex_array(
            self.prog_bloom, self.vbo, "in_pos"
        )

        # Pass 1 renders the scene into this offscreen texture; pass 2 (bloom)
        # samples it and writes the final image into self.fbo.
        self.scene_tex = self.ctx.texture((width, height), 3, dtype="f1")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_tex.repeat_x = False
        self.scene_tex.repeat_y = False
        self.fbo_scene = self.ctx.framebuffer(color_attachments=[self.scene_tex])
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )

        self.tex_spectrum = self.ctx.texture((n_bands, 1), 1, dtype="f4")
        self.tex_waveform = self.ctx.texture((n_bands, 1), 1, dtype="f4")
        for t in (self.tex_spectrum, self.tex_waveform):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False

        # bloom program uniforms (set once)
        self.prog_bloom["u_res"].value = (float(width), float(height))
        self.prog_bloom["u_scene"].value = 2
        self.prog_bloom["u_strength"].value = 0.85

        self._set("u_res", (float(width), float(height)))
        self._set("u_nbands", n_bands)
        self._set("u_spectrum", 0)
        self._set("u_waveform", 1)
        self._set("u_melody", 0.0)
        self._set("u_treble", 0.0)

        # accumulated phases + last timestamp (for beat/bass-driven speeds)
        self._flow = 0.0
        self._ripple = 0.0
        self._last_t = None

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

        # Accumulate the inward-flow phase. Base drift + extra speed on beats
        # (bass + onset) so the mountains surge faster with the music. Phase is
        # integrated frame-by-frame, so it never jumps when the speed changes.
        if self._last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, float(t) - self._last_t)
        self._last_t = float(t)
        flow_speed = 1.4 + 6.0 * float(bass) + 4.0 * float(onset)
        self._flow += dt * flow_speed
        # Floor ripple speed scales with bass: slow when quiet, fast on heavy bass.
        ripple_speed = 3.0 + 28.0 * float(bass)
        self._ripple += dt * ripple_speed

        self._set("u_time", float(t))
        self._set("u_flow", float(self._flow))
        self._set("u_ripple", float(self._ripple))
        self._set("u_amp", float(amp))
        self._set("u_bass", float(bass))
        self._set("u_melody", float(melody))
        self._set("u_treble", float(treble))
        self._set("u_onset", float(onset))

        # Pass 1: render the scene into the offscreen texture.
        self.fbo_scene.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Pass 2: bloom composite -> final framebuffer.
        self.scene_tex.use(location=2)
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao_bloom.render(moderngl.TRIANGLE_STRIP)

        data = self.fbo.read(components=3, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(img).copy()

    def release(self):
        for o in (self.vao, self.vao_bloom, self.vbo, self.tex_spectrum,
                  self.tex_waveform, self.scene_tex, self.fbo_scene, self.fbo,
                  self.prog, self.prog_bloom, self.ctx):
            try:
                o.release()
            except Exception:
                pass
