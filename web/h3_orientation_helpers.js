import * as THREE from "./vendor/three/three.module.min.js";
import { humanFaceBoxes } from "./h3_scene_math.js";

// A world-oriented sphere at infinity: camera translation cannot bring the sky
// into the scene, and camera rotation still moves the latitude/longitude grid.
// Keep the pattern and display colors aligned with Python background_grid().
export function createBackgroundGrid() {
    const material = new THREE.ShaderMaterial({
        side: THREE.BackSide, depthWrite: false, depthTest: false,
        uniforms: { pixelAngle: {value: .002} },
        vertexShader: `varying vec3 skyDirection;
            void main(){skyDirection=position;
            gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
        fragmentShader: `varying vec3 skyDirection; uniform float pixelAngle;
            void main(){
                vec3 d=normalize(skyDirection);
                float longitude=atan(d.x,d.z), latitude=asin(clamp(d.y,-1.0,1.0));
                float horizontal=length(d.xz);
                float distanceToLine=min(abs(sin(12.0*longitude))*horizontal/12.0,
                                         abs(sin(12.0*latitude))/12.0);
                float aa=max(pixelAngle*0.65,0.00005);
                float coverage=1.0-smoothstep(0.0012-aa,0.0012+aa,distanceToLine);
                coverage*=clamp(horizontal/0.03,0.0,1.0);
                // Display-referred colors, identical to CPU output (no lighting).
                gl_FragColor=vec4(vec3(.055,.078,.098)+coverage*vec3(.16,.18,.20),1.0);
            }`,
    });
    const sky = new THREE.Mesh(new THREE.SphereGeometry(100,64,32), material);
    sky.frustumCulled=false;
    sky.renderOrder=-100;
    const size = new THREE.Vector2();
    sky.onBeforeRender=(renderer,scene,camera)=>{
        camera.getWorldPosition(sky.position);
        sky.updateMatrixWorld(true);
        renderer.getDrawingBufferSize(size);
        material.uniforms.pixelAngle.value=2*Math.tan(THREE.MathUtils.degToRad(camera.fov)/2)/Math.max(1,size.y);
    };
    return sky;
}

export function addHumanFaceMarker(group) {
    for (const box of humanFaceBoxes) {
        const material=new THREE.ShaderMaterial({
            vertexShader:`void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
            fragmentShader:`void main(){gl_FragColor=vec4(1.0,1.0,1.0,1.0);}`,
            side:THREE.DoubleSide,
        });
        const marker=new THREE.Mesh(new THREE.BoxGeometry(...box.max.map((v,i)=>v-box.min[i])), material);
        marker.position.fromArray(box.max.map((v,i)=>(v+box.min[i])/2));
        marker.userData.faceMarker=true;
        group.add(marker);
    }
}
