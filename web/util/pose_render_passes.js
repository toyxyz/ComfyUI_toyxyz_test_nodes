import * as T from '../vendor/three/three.module.min.js';

// Data passes use raw RGB, never ACES/sRGB display conversion or scene lights.
export class PoseRenderPasses {
    constructor() {
        this.geometryIndices=new WeakMap();
        this.depthMaterial=new T.ShaderMaterial({
            toneMapped:false,
            uniforms:{depthNear:{value:0},depthFar:{value:1}},
            vertexShader:`
                #include <common>
                #include <skinning_pars_vertex>
                varying float vDepth;
                void main() {
                    #include <skinbase_vertex>
                    #include <begin_vertex>
                    #include <skinning_vertex>
                    #include <project_vertex>
                    vDepth = -mvPosition.z;
                }`,
            fragmentShader:`
                uniform float depthNear;
                uniform float depthFar;
                varying float vDepth;
                void main() {
                    float value = clamp((depthFar - vDepth) / (depthFar - depthNear), 0.0, 1.0);
                    gl_FragColor = vec4(vec3(value), 1.0);
                }`,
        });
        this.normalMaterial=new T.MeshNormalMaterial({toneMapped:false});
    }
    depthRange(root,camera) {
        const roots=Array.isArray(root)?root:[root];roots.forEach(item=>item.updateWorldMatrix(true,true));camera.updateMatrixWorld(true);
        let near=Infinity,far=-Infinity;
        const point=new T.Vector3();
        for(const item of roots)item.traverse(mesh=>{
            if(!mesh.isMesh||!mesh.visible||!mesh.geometry?.attributes.position)return;
            mesh.skeleton?.update();
            let indices=this.geometryIndices.get(mesh.geometry);
            if(!indices){indices=mesh.geometry.index?Array.from(new Set(mesh.geometry.index.array)):
                Array.from({length:mesh.geometry.attributes.position.count},(_,i)=>i);this.geometryIndices.set(mesh.geometry,indices);}
            for(const index of indices){
                mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld).applyMatrix4(camera.matrixWorldInverse);
                const depth=-point.z;if(!Number.isFinite(depth))continue;
                near=Math.min(near,depth);far=Math.max(far,depth);
            }
        });
        if(!Number.isFinite(near)||far<=camera.near||near>=camera.far)return {near:camera.near,far:camera.far};
        const padding=Math.max(.01,(far-near)*.05);
        near=Math.max(camera.near,near-padding);far=Math.min(camera.far,Math.max(near+.001,far+padding));
        return {near,far};
    }
    capture(renderer,scene,camera,bodyRoot,floor) {
        const saved={background:scene.background,override:scene.overrideMaterial,
            colorSpace:renderer.outputColorSpace,toneMapping:renderer.toneMapping,
            shadows:renderer.shadowMap.enabled,floor:floor.visible};
        const range=this.depthRange(bodyRoot,camera);
        // Match opaque MHR surface rendering on folded triangles. Other body
        // templates retain their original front-face-only pass behavior.
        let side=T.FrontSide;
        for(const root of Array.isArray(bodyRoot)?bodyRoot:[bodyRoot])root.traverse(mesh=>{if(mesh.isSkinnedMesh&&mesh.material.side===T.DoubleSide)side=T.DoubleSide;});
        for(const material of [this.depthMaterial,this.normalMaterial])if(material.side!==side){material.side=side;material.needsUpdate=true;}
        try {
            floor.visible=false;scene.background=new T.Color(0x000000);
            renderer.outputColorSpace=T.LinearSRGBColorSpace;renderer.toneMapping=T.NoToneMapping;
            renderer.shadowMap.enabled=false;
            this.depthMaterial.uniforms.depthNear.value=range.near;
            this.depthMaterial.uniforms.depthFar.value=range.far;
            scene.overrideMaterial=this.depthMaterial;renderer.render(scene,camera);
            const depth=renderer.domElement.toDataURL('image/png');
            scene.overrideMaterial=this.normalMaterial;renderer.render(scene,camera);
            const normal=renderer.domElement.toDataURL('image/png');
            return {depth,normal,depth_range:{...range,space:'view',normalized:'posed-body'}};
        } finally {
            floor.visible=saved.floor;scene.background=saved.background;scene.overrideMaterial=saved.override;
            renderer.outputColorSpace=saved.colorSpace;renderer.toneMapping=saved.toneMapping;
            renderer.shadowMap.enabled=saved.shadows;
        }
    }
    dispose(){this.depthMaterial.dispose();this.normalMaterial.dispose();}
}
