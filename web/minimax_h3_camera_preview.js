// Vertical body coordinates use the preview mannequin's 0.00 ground plane and
// 2.59 head top. The crop is solved from these semantic boundaries rather
// than approximated with a camera-distance lookup table.
const SHOT_FRAMING = {
  extreme_close_up: { bottom: 2.02, top: 2.50, label: "eyes and facial detail" },
  close_up: { bottom: 1.73, top: 2.62, label: "head and shoulders" },
  medium_close_up: { bottom: 1.34, top: 2.62, label: "chest / shoulders upward" },
  medium_shot: { bottom: .94, top: 2.62, label: "waist upward" },
  medium_wide_shot: { bottom: .48, top: 2.62, label: "knees upward" },
  cowboy_shot: { bottom: .58, top: 2.62, label: "mid-thigh upward" },
  medium_full_shot: { bottom: .27, top: 2.62, label: "below the knees upward" },
  full_shot: { bottom: -.04, top: 2.66, label: "entire body" },
  wide_shot: { bottom: -.52, top: 3.12, label: "full body with environment" },
  extreme_wide_shot: { bottom: -1.55, top: 4.15, label: "small subject in a broad environment" },
  establishing_shot: { bottom: -2.15, top: 4.75, label: "environment-dominant establishing frame" },
  insert_shot: { bottom: 1.02, top: 1.70, label: "isolated hand or object detail" },
  detail_shot: { bottom: 2.08, top: 2.47, label: "small facial detail" },
  two_shot: { bottom: -.18, top: 2.78, label: "two subjects" },
  three_shot: { bottom: -.36, top: 2.96, label: "three subjects" },
  group_shot: { bottom: -.65, top: 3.25, label: "group composition" },
};

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const mix = (a, b, t) => a + (b - a) * t;
const mix3 = (a, b, t) => a.map((value, index) => mix(value, b[index], t));
const sub3 = (a, b) => a.map((value, index) => value - b[index]);
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = value => {
  const length = Math.hypot(...value) || 1;
  return value.map(component => component / length);
};
const ease = t => t * t * (3 - 2 * t);

function anglePosition(angle, distance, targetY) {
  const state = { position: [0, targetY, distance], target: [0, targetY, 0], roll: 0 };
  switch (angle) {
    case "low_angle": state.position = [0, targetY - distance * .34, distance * .94]; break;
    case "high_angle": state.position = [0, targetY + distance * .38, distance * .92]; break;
    case "overhead": state.position = [0, targetY + distance * .78, distance * .62]; break;
    case "top_down": state.position = [0, targetY + distance, .05]; break;
    case "birds_eye": case "aerial": state.position = [distance * .48, targetY + distance * .72, distance * .5]; break;
    case "worms_eye": state.position = [0, .04, distance]; break;
    case "ground_level": state.position = [0, .12, distance]; break;
    case "dutch_angle": state.roll = -0.28; break;
    case "over_shoulder": state.position = [1.2, 1.7, distance]; break;
    case "pov": state.position = [0, 1.6, distance * .8]; break;
    case "three_quarter": state.position = [distance * .65, 1.45, distance * .72]; break;
    case "profile": state.position = [distance, 1.45, 0]; break;
    case "rear": state.position = [0, 1.45, -distance]; break;
  }
  return state;
}

function cameraState(presets = {}) {
  const shot = presets.camera_shot || "medium_shot";
  const framing = SHOT_FRAMING[shot] || SHOT_FRAMING.medium_shot;
  const fov = shot === "extreme_close_up" || shot === "detail_shot" ? 35 : 45;
  const targetY = (framing.bottom + framing.top) / 2;
  const verticalSpan = framing.top - framing.bottom;
  const distance = (verticalSpan / 2) / Math.tan(fov * Math.PI / 360);
  const state = anglePosition(presets.camera_angle || "eye_level", distance, targetY);
  state.distance = distance;
  state.fov = fov;
  state.shot = shot;
  state.framingLabel = framing.label;
  return state;
}

function motionState(start, target, motion, amount) {
  const result = { ...target, position: [...target.position], target: [...target.target] };
  const d = amount;
  switch (motion) {
    case "push_in": result.position[2] -= d; break;
    case "pull_out": result.position[2] += d; break;
    case "truck_left": case "dolly_left": result.position[0] -= d; result.target[0] -= d * .35; break;
    case "truck_right": case "dolly_right": result.position[0] += d; result.target[0] += d * .35; break;
    case "pedestal_up": case "crane_up": result.position[1] += d; result.target[1] += d * .3; break;
    case "pedestal_down": case "crane_down": result.position[1] -= d; result.target[1] -= d * .3; break;
    case "pan_left": result.target[0] -= d; break;
    case "pan_right": result.target[0] += d; break;
    case "tilt_up": result.target[1] += d * .65; break;
    case "tilt_down": result.target[1] -= d * .65; break;
    case "roll_clockwise": result.roll += .4; break;
    case "roll_counterclockwise": result.roll -= .4; break;
    case "zoom_in": result.fov = Math.max(24, target.fov - 15); break;
    case "zoom_out": result.fov = Math.min(75, target.fov + 17); break;
    case "dolly_zoom_in": result.position[2] -= d; result.fov = Math.min(72, target.fov + 13); break;
    case "dolly_zoom_out": result.position[2] += d; result.fov = Math.max(25, target.fov - 12); break;
    case "orbit_left": case "arc": {
      const radius = Math.hypot(result.position[0], result.position[2]);
      const angle = Math.atan2(result.position[0], result.position[2]) - .8;
      result.position[0] = Math.sin(angle) * radius; result.position[2] = Math.cos(angle) * radius; break;
    }
    case "orbit_right": {
      const radius = Math.hypot(result.position[0], result.position[2]);
      const angle = Math.atan2(result.position[0], result.position[2]) + .8;
      result.position[0] = Math.sin(angle) * radius; result.position[2] = Math.cos(angle) * radius; break;
    }
    case "tracking": case "follow": result.position[0] += d * .55; result.target[0] += d * .55; break;
  }
  return result;
}

function interpolateState(start, end, t, motion) {
  let position = mix3(start.position, end.position, t);
  let target = mix3(start.target, end.target, t);
  if (motion === "handheld" || motion === "shake_slightly" || motion === "shake_strongly") {
    const strength = motion === "shake_strongly" ? .14 : motion === "handheld" ? .055 : .035;
    position = [position[0] + Math.sin(t * 43) * strength, position[1] + Math.sin(t * 31) * strength, position[2]];
    target = [target[0] + Math.sin(t * 37) * strength, target[1] + Math.cos(t * 29) * strength, target[2]];
  }
  return { position, target, roll: mix(start.roll, end.roll, t), fov: mix(start.fov, end.fov, t) };
}

function projector(camera, width, height) {
  const forward = norm3(sub3(camera.target, camera.position));
  let right = norm3(cross3(forward, [0, 1, 0]));
  let up = norm3(cross3(right, forward));
  if (camera.roll) {
    const cosine = Math.cos(camera.roll), sine = Math.sin(camera.roll);
    const rolledRight = right.map((value, i) => value * cosine + up[i] * sine);
    up = up.map((value, i) => value * cosine - right[i] * sine);
    right = rolledRight;
  }
  const focal = height * .5 / Math.tan((camera.fov || 45) * Math.PI / 360);
  return point => {
    const relative = sub3(point, camera.position);
    const z = dot3(relative, forward);
    if (z <= .03) return null;
    return [width / 2 + dot3(relative, right) * focal / z, height / 2 - dot3(relative, up) * focal / z, z];
  };
}

function line(ctx, project, a, b, color, width = 1, dash = []) {
  const pa = project(a), pb = project(b);
  if (!pa || !pb) return;
  ctx.beginPath(); ctx.setLineDash(dash); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.setLineDash([]);
}

function polygon(ctx, project, points, fill, stroke) {
  const projected = points.map(project);
  if (projected.some(point => !point)) return;
  ctx.beginPath(); projected.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
  ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
}

function box(ctx, project, center, size, fill = "rgba(105,185,255,.36)") {
  const [x, y, z] = center, [w, h, d] = size;
  const p = [[x-w/2,y-h/2,z-d/2],[x+w/2,y-h/2,z-d/2],[x+w/2,y+h/2,z-d/2],[x-w/2,y+h/2,z-d/2],
    [x-w/2,y-h/2,z+d/2],[x+w/2,y-h/2,z+d/2],[x+w/2,y+h/2,z+d/2],[x-w/2,y+h/2,z+d/2]];
  [[4,5,6,7],[0,1,2,3],[0,4,7,3],[1,5,6,2],[3,2,6,7]].forEach(face => polygon(ctx, project, face.map(i => p[i]), fill, "rgba(170,220,255,.72)"));
}

function drawMannequin(ctx, project) {
  box(ctx, project, [0, 1.25, 0], [.7, 1.25, .24]);
  box(ctx, project, [-.48, 1.25, 0], [.18, 1.12, .12], "rgba(178,151,255,.35)");
  box(ctx, project, [.48, 1.25, 0], [.18, 1.12, .12], "rgba(178,151,255,.35)");
  box(ctx, project, [-.22, .35, 0], [.22, .75, .16], "rgba(178,151,255,.35)");
  box(ctx, project, [.22, .35, 0], [.22, .75, .16], "rgba(178,151,255,.35)");
  const head = project([0, 2.25, 0]);
  const headRight = project([.34, 2.25, 0]);
  const headTop = project([0, 2.59, 0]);
  if (head && headRight && headTop) {
    const radiusX = Math.hypot(headRight[0] - head[0], headRight[1] - head[1]);
    const radiusY = Math.hypot(headTop[0] - head[0], headTop[1] - head[1]);
    const radius = clamp(Math.max(radiusX, radiusY), 10, 220);
    ctx.beginPath(); ctx.arc(head[0], head[1], radius, 0, Math.PI * 2);
    ctx.fillStyle = "#d7b18c"; ctx.fill(); ctx.strokeStyle = "#ffe0bd"; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

function drawGrid(ctx, project) {
  for (let i = -6; i <= 6; i++) {
    const major = i === 0;
    line(ctx, project, [i, 0, -6], [i, 0, 6], major ? "rgba(101,185,255,.5)" : "rgba(130,145,165,.2)", major ? 1.5 : 1);
    line(ctx, project, [-6, 0, i], [6, 0, i], major ? "rgba(255,120,130,.42)" : "rgba(130,145,165,.2)", major ? 1.5 : 1);
  }
}

function drawCameraRig(ctx, project, state, color = "#ffd166") {
  const p = state.position, forward = norm3(sub3(state.target, p));
  const right = norm3(cross3(forward, [0,1,0])), up = norm3(cross3(right, forward));
  box(ctx, project, p, [.36,.25,.42], "rgba(255,209,102,.5)");
  const far = .9, halfY = Math.tan(state.fov * Math.PI / 360) * far, halfX = halfY * 1.5;
  const center = p.map((v,i) => v + forward[i] * far);
  const corners = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([sx,sy]) => center.map((v,i) => v + right[i]*halfX*sx + up[i]*halfY*sy));
  corners.forEach(corner => line(ctx, project, p, corner, color, 1));
  corners.forEach((corner,index) => line(ctx, project, corner, corners[(index+1)%4], color, 1));
  line(ctx, project, p, state.target, "rgba(255,209,102,.55)", 1, [4,4]);
}

export class CameraPresetPreview {
  constructor(canvas, statusElement) {
    this.canvas = canvas; this.statusElement = statusElement; this.ctx = canvas.getContext("2d", { alpha: false });
    this.view = "scene"; this.playing = false; this.progress = 0; this.lastTime = 0; this.raf = 0; this.visible = false;
    this.start = cameraState({}); this.end = cameraState({}); this.motion = "none"; this.speed = 1;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas.parentElement);
    this.intersectionObserver = new IntersectionObserver(entries => {
      if (!entries[0]?.isIntersecting) this.pause();
    }, { threshold: .01 });
    this.intersectionObserver.observe(canvas);
    this.visibilityHandler = () => { if (document.hidden) this.pause(); };
    document.addEventListener("visibilitychange", this.visibilityHandler);
  }
  setVisible(visible) { this.visible = visible; if (!visible) this.pause(); else this.draw(); }
  setView(view) { this.view = view === "camera" ? "camera" : "scene"; this.draw(); }
  setData({ startPresets = {}, endPresets = {}, motion = "none", amplitude = "none", speed = "none", label = "" }) {
    this.start = cameraState(startPresets); const target = cameraState(endPresets);
    const amount = amplitude === "large" ? 2.2 : amplitude === "small" ? .65 : 1.25;
    this.end = motionState(this.start, target, motion, amount);
    this.motion = motion; this.speed = speed === "fast" ? 1.7 : speed === "slow" ? .65 : 1;
    this.label = label; this.progress = 0;
    this.statusElement.textContent = `${label} · ${target.shot.replaceAll("_", " ")} (${target.framingLabel}) · ${motion === "none" ? "static / inherited" : motion.replaceAll("_", " ")}`;
    this.draw();
  }
  togglePlay() { this.playing ? this.pause() : this.play(); return this.playing; }
  setProgress(progress) {
    this.pause();
    this.progress = clamp(Number(progress) || 0, 0, 1);
    this.draw();
  }
  play() { if (!this.visible || this.playing) return; this.playing = true; this.lastTime = performance.now(); this.raf = requestAnimationFrame(time => this.tick(time)); }
  pause() { this.playing = false; cancelAnimationFrame(this.raf); this.raf = 0; }
  reset() { this.pause(); this.progress = 0; this.draw(); }
  tick(time) {
    if (!this.playing || !this.visible) return;
    this.progress = (this.progress + (time - this.lastTime) / 3500 * this.speed) % 1;
    this.lastTime = time; this.draw(); this.raf = requestAnimationFrame(next => this.tick(next));
  }
  draw() {
    if (!this.visible) return;
    const parent = this.canvas.parentElement, dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(280, parent.clientWidth), height = Math.max(260, parent.clientHeight);
    if (this.canvas.width !== Math.round(width*dpr) || this.canvas.height !== Math.round(height*dpr)) {
      this.canvas.width = Math.round(width*dpr); this.canvas.height = Math.round(height*dpr);
      this.canvas.style.width = `${width}px`; this.canvas.style.height = `${height}px`;
    }
    const ctx = this.ctx; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.fillStyle = "#11161d"; ctx.fillRect(0,0,width,height);
    const t = ease(this.progress), current = interpolateState(this.start, this.end, t, this.motion);
    const observer = this.view === "camera" ? current : { position:[7,5.1,8], target:[0,1,0], roll:0, fov:50 };
    const project = projector(observer, width, height); drawGrid(ctx, project); drawMannequin(ctx, project);
    if (this.view === "scene") {
      let previous = this.start.position;
      for (let i=1;i<=24;i++) { const state = interpolateState(this.start,this.end,i/24,this.motion); line(ctx,project,previous,state.position,"rgba(178,156,255,.7)",2,[4,4]); previous=state.position; }
      drawCameraRig(ctx, project, this.start, "#65b9ff"); drawCameraRig(ctx, project, current, "#ffd166");
    }
    ctx.fillStyle = "rgba(8,12,17,.72)"; ctx.fillRect(8,8, this.view === "camera" ? 116 : 190, 25);
    ctx.fillStyle = "#cbd8e5"; ctx.font = "11px Inter,system-ui,sans-serif";
    ctx.fillText(this.view === "camera" ? "CAMERA VIEW" : `SCENE VIEW  ${Math.round(this.progress*100)}%`, 16,25);
  }
  dispose() {
    this.pause(); this.resizeObserver.disconnect(); this.intersectionObserver.disconnect();
    document.removeEventListener("visibilitychange", this.visibilityHandler);
    this.canvas.width = 1; this.canvas.height = 1;
  }
}
