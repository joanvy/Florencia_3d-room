import * as THREE from "three";
import { SparkRenderer } from "@sparkjsdev/spark";

export interface MirrorConfig {
  /** Centre of the glass, world space. */
  center: [number, number, number];
  /** Unit normal pointing into the room. */
  normal: [number, number, number];
  width: number;
  height: number;
  /** Reflection resolution relative to the canvas (default 0.5, lower on phones). */
  quality?: number;
  tint?: [number, number, number];
}

/**
 * A planar mirror for a splat scene: the room is rendered a second time from
 * the camera reflected in the glass plane, into an offscreen target that is
 * projected onto the mirror quad (the same technique as three's Reflector).
 *
 * Splat reconstructions can't represent real reflections — they build a fake
 * "room behind the glass" instead. The pipeline deletes that phantom room and
 * the glass itself (pipeline/cleanup.py, mirror mode "reflect"), which also
 * guarantees nothing sits between the reflected camera and the glass, so no
 * clip plane is needed; this puts a true, view-dependent reflection back.
 */
export class SplatMirror {
  readonly mesh: THREE.Mesh;
  readonly spark: SparkRenderer;
  private virtualCamera = new THREE.PerspectiveCamera();
  private textureMatrix = new THREE.Matrix4();
  private material: THREE.ShaderMaterial;
  private quality: number;
  private plane = new THREE.Plane();
  private frustum = new THREE.Frustum();
  private box = new THREE.Box3();

  constructor(
    private renderer: THREE.WebGLRenderer,
    cfg: MirrorConfig,
    private onDirty: () => void,
  ) {
    this.quality = cfg.quality ?? 0.5;
    const size = this.targetSize();
    this.spark = new SparkRenderer({ renderer, onDirty, target: { width: size.x, height: size.y } });

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tReflection: { value: null },
        textureMatrix: { value: this.textureMatrix },
        tint: { value: new THREE.Color(...(cfg.tint ?? [0.93, 0.95, 0.97])) },
      },
      vertexShader: /* glsl */ `
        uniform mat4 textureMatrix;
        varying vec4 vUv;
        varying vec2 vLocal;
        void main() {
          vUv = textureMatrix * vec4(position, 1.0);
          vLocal = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D tReflection;
        uniform vec3 tint;
        varying vec4 vUv;
        varying vec2 vLocal;
        void main() {
          // Spark writes linear colour into offscreen targets; encode to sRGB for the canvas.
          vec3 lin = texture2DProj(tReflection, vUv).rgb;
          vec3 srgb = mix(lin * 12.92, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
          vec3 c = srgb * tint;
          // Slight darkening towards the edges sells the glass.
          vec2 e = min(vLocal, 1.0 - vLocal);
          float edge = smoothstep(0.0, 0.03, min(e.x, e.y));
          gl_FragColor = vec4(c * mix(0.75, 1.0, edge), 1.0);
        }`,
    });

    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(cfg.width, cfg.height), this.material);
    const c = new THREE.Vector3(...cfg.center);
    const n = new THREE.Vector3(...cfg.normal).normalize();
    this.mesh.position.copy(c);
    this.mesh.lookAt(c.clone().add(n)); // plane's +Z faces into the room
    this.mesh.updateMatrixWorld();
    this.plane.setFromNormalAndCoplanarPoint(n, c);
    this.mesh.geometry.computeBoundingBox();
    this.box.copy(this.mesh.geometry.boundingBox!).applyMatrix4(this.mesh.matrixWorld);
  }

  private targetSize() {
    const s = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    return new THREE.Vector2(Math.max(64, Math.round(s.x * this.quality)), Math.max(64, Math.round(s.y * this.quality)));
  }

  resize() {
    const size = this.targetSize();
    const t = this.spark.target;
    if (t && (t.width !== size.x || t.height !== size.y)) t.setSize(size.x, size.y);
  }

  /** Render the reflection for `camera`. Call before the main render. */
  update(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    const camPos = camera.getWorldPosition(new THREE.Vector3());
    // Skip when the camera is behind the glass or the mirror is off-screen.
    this.frustum.setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
    );
    if (this.plane.distanceToPoint(camPos) <= 0 || !this.frustum.intersectsBox(this.box)) {
      this.mesh.visible = this.plane.distanceToPoint(camPos) > 0;
      return;
    }
    this.mesh.visible = true;

    const n = this.plane.normal;
    const mirrorPos = this.mesh.position;
    // Reflect camera position, look target and up vector across the plane.
    const view = mirrorPos.clone().sub(camPos).reflect(n).negate().add(mirrorPos);
    const rot = new THREE.Matrix4().extractRotation(camera.matrixWorld);
    const lookAt = new THREE.Vector3(0, 0, -1).applyMatrix4(rot).add(camPos);
    const target = mirrorPos.clone().sub(lookAt).reflect(n).negate().add(mirrorPos);
    const vc = this.virtualCamera;
    vc.position.copy(view);
    vc.up.set(0, 1, 0).applyMatrix4(rot).reflect(n);
    vc.lookAt(target);
    vc.near = camera.near;
    vc.far = camera.far;
    vc.updateMatrixWorld();
    vc.projectionMatrix.copy(camera.projectionMatrix);
    vc.projectionMatrixInverse.copy(camera.projectionMatrixInverse);

    this.textureMatrix.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
    this.textureMatrix.multiply(vc.projectionMatrix).multiply(vc.matrixWorldInverse).multiply(this.mesh.matrixWorld);


    this.mesh.visible = false;
    const rt = this.spark.renderTarget({ scene, camera: vc });
    this.mesh.visible = true;
    this.material.uniforms.tReflection.value = rt.texture;
  }
}
