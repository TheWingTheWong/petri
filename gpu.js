// Petri — WebGL2 GPGPU engine.
//
// Particle state lives in two RGBA32F textures (x, y, vx, vy) that are
// ping-ponged each tick. One fragment shader invocation per particle sums
// forces from every other particle (brute force O(n²): no neighbour structure
// to maintain, and GPUs eat that for breakfast). The same math as sim.js, so
// the CPU engine is a drop-in fallback and a reference for testing.

const QUAD_VS = `#version 300 es
in vec2 aPos; void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const STEP_FS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uState; uniform sampler2D uType;
uniform int uN; uniform int uW; uniform int uK;
uniform float uMatrix[64];
uniform float uRMax; uniform float uBeta; uniform float uForce; uniform float uDt; uniform float uDamp;
uniform vec4 uStir;
out vec4 outState;
float wrapd(float d) { return d > 0.5 ? d - 1.0 : (d < -0.5 ? d + 1.0 : d); }
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  int i = c.y * uW + c.x;
  vec4 s = texelFetch(uState, c, 0);
  if (i >= uN) { outState = s; return; }
  int ti = int(texelFetch(uType, c, 0).r + 0.5);
  vec2 f = vec2(0.0);
  float r2max = uRMax * uRMax;
  for (int j = 0; j < uN; j++) {
    if (j == i) continue;
    ivec2 cj = ivec2(j % uW, j / uW);
    vec2 pj = texelFetch(uState, cj, 0).xy;
    vec2 d = vec2(wrapd(pj.x - s.x), wrapd(pj.y - s.y));
    float d2 = dot(d, d);
    if (d2 == 0.0 || d2 >= r2max) continue;
    float dist = sqrt(d2);
    float r = dist / uRMax;
    float a = uMatrix[ti * uK + int(texelFetch(uType, cj, 0).r + 0.5)];
    float fr = r < uBeta ? r / uBeta - 1.0 : a * (1.0 - abs(2.0 * r - 1.0 - uBeta) / (1.0 - uBeta));
    f += d / dist * fr;
  }
  f *= uRMax * uForce;
  if (uStir.z > 0.0) {
    vec2 d = vec2(wrapd(uStir.x - s.x), wrapd(uStir.y - s.y));
    float dd = length(d);
    if (dd < uStir.z && dd > 1e-6) f += d / dd * uStir.w * (1.0 - dd / uStir.z);
  }
  vec2 v = s.zw * uDamp + f * uDt;
  vec2 p = s.xy + v * uDt + 1.0;
  outState = vec4(p - floor(p), v);
}`;

const POINTS_VS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uState; uniform sampler2D uType;
uniform int uW; uniform vec2 uRes; uniform vec3 uView; uniform float uSize;
uniform vec3 uColors[8];
out vec3 vCol;
void main() {
  ivec2 c = ivec2(gl_VertexID % uW, gl_VertexID / uW);
  vec2 p = texelFetch(uState, c, 0).xy;
  vCol = uColors[int(texelFetch(uType, c, 0).r + 0.5)];
  vec2 px = uView.xy + p * uView.z;
  gl_Position = vec4(px.x / uRes.x * 2.0 - 1.0, 1.0 - px.y / uRes.y * 2.0, 0.0, 1.0);
  gl_PointSize = uSize;
}`;

const POINTS_FS = `#version 300 es
precision highp float;
in vec3 vCol; uniform float uGlow; out vec4 o;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float d = dot(q, q);
  if (d > 1.0) discard;
  float core = smoothstep(0.16, 0.0, d);
  float halo = (1.0 - d) * uGlow;
  o = vec4(vCol * (core + halo), 1.0);
}`;

const FADE_FS = `#version 300 es
precision highp float; uniform vec4 uColor; out vec4 o; void main() { o = uColor; }`;

function compile(gl, vs, fs) {
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.bindAttribLocation(prog, 0, 'aPos');
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
  const loc = new Proxy({}, { get: (c, name) => (name in c ? c[name] : (c[name] = gl.getUniformLocation(prog, name))) });
  return { prog, loc };
}

const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

export class GpuSim {
  static supported() {
    try {
      const gl = document.createElement('canvas').getContext('webgl2');
      return !!(gl && gl.getExtension('EXT_color_buffer_float'));
    } catch { return false; }
  }

  constructor(canvas, colors) {
    const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false, alpha: false, premultipliedAlpha: false });
    if (!gl || !gl.getExtension('EXT_color_buffer_float')) throw new Error('WebGL2 float render targets unavailable');
    this.gl = gl; this.canvas = canvas;
    this.colors = new Float32Array(colors.flatMap(hexToRgb));
    this.stepP = compile(gl, QUAD_VS, STEP_FS);
    this.ptsP = compile(gl, POINTS_VS, POINTS_FS);
    this.fadeP = compile(gl, QUAD_VS, FADE_FS);
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.emptyVao = gl.createVertexArray();
    this.cur = 0;
  }

  _tex(w, h, internal, format, data) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.FLOAT, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    return t;
  }

  /** Upload a CPU world (from sim.js createWorld) to the GPU. */
  load(world) {
    const gl = this.gl;
    this.dispose();
    const n = world.n, W = Math.ceil(Math.sqrt(n)), H = Math.ceil(n / W);
    const state = new Float32Array(W * H * 4), type = new Float32Array(W * H);
    for (let i = 0; i < n; i++) {
      state.set([world.x[i], world.y[i], world.vx[i], world.vy[i]], i * 4);
      type[i] = world.type[i];
    }
    Object.assign(this, { n, W, H });
    this.state = [this._tex(W, H, gl.RGBA32F, gl.RGBA, state), this._tex(W, H, gl.RGBA32F, gl.RGBA, state)];
    this.type = this._tex(W, H, gl.R32F, gl.RED, type);
    this.fbo = this.state.map((t) => {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('float framebuffer incomplete');
      return f;
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.cur = 0;
    this.world = world;
  }

  dispose() {
    const gl = this.gl;
    (this.state || []).forEach((t) => gl.deleteTexture(t));
    (this.fbo || []).forEach((f) => gl.deleteFramebuffer(f));
    if (this.type) gl.deleteTexture(this.type);
    this.state = this.fbo = this.type = null;
  }

  step(stir = null) {
    const gl = this.gl, w = this.world, { prog, loc } = this.stepP;
    const src = this.cur, dst = 1 - src;
    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[dst]);
    gl.viewport(0, 0, this.W, this.H);
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.state[src]); gl.uniform1i(loc.uState, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.type); gl.uniform1i(loc.uType, 1);
    gl.uniform1i(loc.uN, this.n); gl.uniform1i(loc.uW, this.W); gl.uniform1i(loc.uK, w.k);
    const m = new Float32Array(64); m.set(w.matrix);
    gl.uniform1fv(loc.uMatrix, m);
    gl.uniform1f(loc.uRMax, w.rMax); gl.uniform1f(loc.uBeta, w.beta); gl.uniform1f(loc.uForce, w.forceFactor);
    gl.uniform1f(loc.uDt, w.dt); gl.uniform1f(loc.uDamp, Math.pow(0.5, w.dt / w.friction));
    gl.uniform4f(loc.uStir, stir ? stir.x : 0, stir ? stir.y : 0, stir ? stir.radius : 0, stir ? stir.strength : 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.cur = dst;
    w.steps++;
  }

  /** Copy GPU state back into the CPU world arrays (for metrics, photos, engine switches). */
  readback() {
    const gl = this.gl, w = this.world;
    const buf = new Float32Array(this.W * this.H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[this.cur]);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    for (let i = 0; i < this.n; i++) { w.x[i] = buf[i * 4]; w.y[i] = buf[i * 4 + 1]; w.vx[i] = buf[i * 4 + 2]; w.vy[i] = buf[i * 4 + 3]; }
    return w;
  }

  /** Render particles into the dish. view = {cx, cy, R} in canvas pixels. */
  draw({ cx, cy, R }, { trails = true, clear = false, size = 6, glow = 0.16, bg = [4 / 255, 5 / 255, 7 / 255] } = {}) {
    const gl = this.gl, c = this.canvas;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, c.width, c.height);
    if (clear || !trails) {
      gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
    } else {
      gl.useProgram(this.fadeP.prog);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform4f(this.fadeP.loc.uColor, bg[0], bg[1], bg[2], 0.32);
      gl.bindVertexArray(this.vao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    const { prog, loc } = this.ptsP;
    gl.useProgram(prog);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.state[this.cur]); gl.uniform1i(loc.uState, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.type); gl.uniform1i(loc.uType, 1);
    gl.uniform1i(loc.uW, this.W); gl.uniform2f(loc.uRes, c.width, c.height);
    gl.uniform3f(loc.uView, cx - R, cy - R, 2 * R);
    gl.uniform1f(loc.uSize, size); gl.uniform1f(loc.uGlow, glow);
    gl.uniform3fv(loc.uColors, this.colors);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.POINTS, 0, this.n);
    gl.disable(gl.BLEND);
  }
}
