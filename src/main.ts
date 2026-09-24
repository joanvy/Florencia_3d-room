import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh, isMobile } from "@sparkjsdev/spark";
import "./style.css";

type Vec3 = [number, number, number];

interface Viewpoint {
  name: string;
  position: Vec3;
  target: Vec3;
}

interface SceneConfig {
  title: string;
  splat: string;
  /** Euler rotation (degrees, XYZ) applied to the splat, for files that are not already Y-up. */
  rotation?: Vec3;
  position?: Vec3;
  fov?: number;
  /** Axis-aligned box (world space) the camera is kept inside. */
  bounds?: { min: Vec3; max: Vec3 };
  background?: string;
  viewpoints: Viewpoint[];
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const canvas = $<HTMLCanvasElement>("view");
const loader = $("loader");
const barFill = $("bar-fill");
const loaderDetail = $("loader-detail");
const spotsNav = $("spots");

const mobile = isMobile();
// Splat rendering is fill-rate bound: cap the pixel ratio on dense phone screens.
const MAX_PIXEL_RATIO = Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2);
const MIN_PIXEL_RATIO = mobile ? 0.75 : 1;
let pixelRatio = MAX_PIXEL_RATIO;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(pixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight, false);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.02, 100);

// Keep a comfortable *horizontal* field of view on portrait phones, where a
// fixed vertical FOV would feel like looking through a letterbox slot.
let baseFov = 70;
const MIN_HFOV = 80;
function updateFov() {
  const aspect = window.innerWidth / window.innerHeight;
  const vFromH = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(MIN_HFOV / 2)) / aspect));
  camera.fov = Math.min(105, Math.max(baseFov, vFromH));
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
}
updateFov();

let dirty = true;
const requestRender = () => {
  dirty = true;
};

const spark = new SparkRenderer({ renderer, onDirty: requestRender });
scene.add(spark);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.rotateSpeed = mobile ? -0.45 : -0.35; // negative: drag feels like turning your head
controls.zoomSpeed = 0.8;
controls.panSpeed = 0.8;
controls.screenSpacePanning = true;
controls.minDistance = 0.05;
controls.maxDistance = 4;
controls.addEventListener("change", requestRender);

// ---------------------------------------------------------------------------
// Camera transitions between viewpoints

let transition: {
  from: { pos: THREE.Vector3; target: THREE.Vector3 };
  to: { pos: THREE.Vector3; target: THREE.Vector3 };
  start: number;
  duration: number;
} | null = null;

function goTo(vp: Viewpoint, instant = false) {
  const to = { pos: new THREE.Vector3(...vp.position), target: new THREE.Vector3(...vp.target) };
  if (instant) {
    camera.position.copy(to.pos);
    controls.target.copy(to.target);
    controls.update();
    requestRender();
    return;
  }
  transition = {
    from: { pos: camera.position.clone(), target: controls.target.clone() },
    to,
    start: performance.now(),
    duration: 900,
  };
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// ---------------------------------------------------------------------------
// Keep the camera inside the room

let bounds: THREE.Box3 | null = null;
function clampCamera() {
  if (!bounds) return;
  const before = camera.position.clone();
  bounds.clampPoint(camera.position, camera.position);
  // Slide the orbit target with the camera so the view direction is preserved.
  controls.target.add(camera.position.clone().sub(before));
}

// ---------------------------------------------------------------------------
// Adaptive resolution: drop pixel ratio when frames are slow, recover when fast.

let frameTimes: number[] = [];
function adaptResolution(dt: number) {
  frameTimes.push(dt);
  if (frameTimes.length < 30) return;
  const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
  frameTimes = [];
  let next = pixelRatio;
  if (avg > 40) next = Math.max(MIN_PIXEL_RATIO, pixelRatio - 0.25);
  else if (avg < 22) next = Math.min(MAX_PIXEL_RATIO, pixelRatio + 0.25);
  if (next !== pixelRatio) {
    pixelRatio = next;
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  }
}

// ---------------------------------------------------------------------------
// Main loop: render only when something changed.

let last = performance.now();
renderer.setAnimationLoop((now: number) => {
  const dt = now - last;
  last = now;

  if (transition) {
    const t = Math.min(1, (now - transition.start) / transition.duration);
    const k = ease(t);
    camera.position.lerpVectors(transition.from.pos, transition.to.pos, k);
    controls.target.lerpVectors(transition.from.target, transition.to.target, k);
    if (t >= 1) transition = null;
    dirty = true;
  }

  if (controls.update()) dirty = true;
  if (!dirty) return;
  dirty = false;

  clampCamera();
  renderer.render(scene, camera);
  adaptResolution(dt);
});

window.addEventListener("resize", () => {
  updateFov();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  requestRender();
});

// ---------------------------------------------------------------------------
// UI

const helpDialog = $("help");
$("help-btn").addEventListener("click", () => (helpDialog.hidden = false));
$("help-close").addEventListener("click", () => (helpDialog.hidden = true));
helpDialog.addEventListener("click", (e) => {
  if (e.target === helpDialog) helpDialog.hidden = true;
});

function buildViewpointButtons(viewpoints: Viewpoint[]) {
  const buttons: HTMLButtonElement[] = [];
  viewpoints.forEach((vp, i) => {
    const b = document.createElement("button");
    b.className = "pill";
    b.textContent = vp.name;
    b.setAttribute("aria-pressed", String(i === 0));
    b.addEventListener("click", () => {
      buttons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      goTo(vp);
    });
    buttons.push(b);
    spotsNav.appendChild(b);
  });
  // Any manual interaction clears the "current viewpoint" highlight.
  controls.addEventListener("start", () => buttons.forEach((o) => o.setAttribute("aria-pressed", "false")));
}

function showError(message: string) {
  const el = $("error");
  el.innerHTML = "";
  const box = document.createElement("div");
  box.textContent = message;
  el.appendChild(box);
  el.hidden = false;
  loader.classList.add("done");
}

const formatMB = (bytes: number) => (bytes / 1024 / 1024).toFixed(1);

// ---------------------------------------------------------------------------
// Boot

async function main() {
  if (!renderer.capabilities.isWebGL2) {
    showError("Sorry — this device's browser doesn't support WebGL 2, which is needed to show the 3D room.");
    return;
  }

  const config: SceneConfig = await fetch("scene.json").then((r) => r.json());
  document.title = `${config.title} · 3D`;
  $("title").textContent = config.title;
  if (config.background) scene.background = new THREE.Color(config.background);
  if (config.fov) {
    baseFov = config.fov;
    updateFov();
  }
  if (config.bounds) {
    bounds = new THREE.Box3(new THREE.Vector3(...config.bounds.min), new THREE.Vector3(...config.bounds.max));
  }

  buildViewpointButtons(config.viewpoints);
  if (config.viewpoints[0]) goTo(config.viewpoints[0], true);

  const room = new SplatMesh({
    url: config.splat,
    onProgress: (e: ProgressEvent) => {
      if (e.lengthComputable && e.total > 0) {
        barFill.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
        loaderDetail.textContent = `${formatMB(e.loaded)} / ${formatMB(e.total)} MB`;
      } else {
        loaderDetail.textContent = `${formatMB(e.loaded)} MB`;
      }
    },
  });
  if (config.rotation) {
    room.rotation.set(...(config.rotation.map(THREE.MathUtils.degToRad) as Vec3));
  }
  if (config.position) room.position.set(...config.position);
  scene.add(room);

  await room.initialized;
  barFill.style.width = "100%";
  loaderDetail.textContent = "Preparing…";
  requestRender();
  // Give Spark a couple of frames to sort before revealing.
  setTimeout(() => loader.classList.add("done"), 250);
}

main().catch((err) => {
  console.error(err);
  showError("Couldn't load the 3D room. Check your connection and reload the page.");
});
