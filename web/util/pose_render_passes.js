import * as T from '../vendor/three/three.module.min.js';

// Depth/normal use raw RGB. Color IDs encode the selected sRGB colors exactly.
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
    capture(renderer,scene,camera,bodyRoot) {
        const saved={background:scene.background,override:scene.overrideMaterial,
            colorSpace:renderer.outputColorSpace,toneMapping:renderer.toneMapping,
            shadows:renderer.shadowMap.enabled};
        const range=this.depthRange(bodyRoot,camera);
        const roots=Array.isArray(bodyRoot)?bodyRoot:[bodyRoot];
        const replaced=[],flatMaterials=new Map();
        const flat=original=>{
            if(!flatMaterials.has(original))flatMaterials.set(original,new T.MeshBasicMaterial({
                color:original.color?.clone()??new T.Color(0xffffff),
                side:original.side,toneMapped:false,fog:false,
            }));
            return flatMaterials.get(original);
        };
        const replace=mesh=>{
            if(!mesh.isMesh)return;
            const material=mesh.material;replaced.push([mesh,material]);
            mesh.material=Array.isArray(material)?material.map(flat):flat(material);
        };
        // Match opaque MHR surface rendering on folded triangles. Other body
        // templates retain their original front-face-only pass behavior.
        let side=T.FrontSide;
        for(const root of Array.isArray(bodyRoot)?bodyRoot:[bodyRoot])root.traverse(mesh=>{if(mesh.isSkinnedMesh&&mesh.material.side===T.DoubleSide)side=T.DoubleSide;});
        for(const material of [this.depthMaterial,this.normalMaterial])if(material.side!==side){material.side=side;material.needsUpdate=true;}
        try {
            scene.background=new T.Color(0x000000);
            renderer.outputColorSpace=T.LinearSRGBColorSpace;renderer.toneMapping=T.NoToneMapping;
            renderer.shadowMap.enabled=false;
            this.depthMaterial.uniforms.depthNear.value=range.near;
            this.depthMaterial.uniforms.depthFar.value=range.far;
            scene.overrideMaterial=this.depthMaterial;renderer.render(scene,camera);
            const depth=renderer.domElement.toDataURL('image/png');
            scene.overrideMaterial=this.normalMaterial;renderer.render(scene,camera);
            const normal=renderer.domElement.toDataURL('image/png');
            // Keep skinning, depth testing and each object's own color/side, but
            // no lights, shadows, textures, tone mapping or editor controllers.
            // sRGB encoding reverses Color's hex-to-linear conversion so a
            // #3377cc body remains RGB (51,119,204), not its linear RGB value.
            scene.overrideMaterial=null;renderer.outputColorSpace=T.SRGBColorSpace;
            roots.forEach(root=>root.traverse(replace));renderer.render(scene,camera);
            const color_id=renderer.domElement.toDataURL('image/png');
            return {depth,normal,color_id,depth_range:{...range,space:'view',normalized:'posed-body'}};
        } finally {
            for(const [mesh,material] of replaced)mesh.material=material;
            for(const material of flatMaterials.values())material.dispose();
            scene.background=saved.background;scene.overrideMaterial=saved.override;
            renderer.outputColorSpace=saved.colorSpace;renderer.toneMapping=saved.toneMapping;
            renderer.shadowMap.enabled=saved.shadows;
        }
    }
    captureBodyIds(renderer,scene,camera,bodies,width,height,sourceId) {
        const target=new T.WebGLRenderTarget(width,height,{depthBuffer:true,stencilBuffer:false,samples:0});
        target.texture.colorSpace=T.NoColorSpace;
        const saved={target:renderer.getRenderTarget(),viewport:renderer.getViewport(new T.Vector4()),scissor:renderer.getScissor(new T.Vector4()),scissorTest:renderer.getScissorTest(),autoClear:renderer.autoClear,background:scene.background,override:scene.overrideMaterial,tone:renderer.toneMapping,shadows:renderer.shadowMap.enabled};
        const replaced=[],materials=[];
        const replace=(mesh,label)=>{if(!mesh.isMesh)return;const original=mesh.material;replaced.push([mesh,original]);const flat=material=>{const color=new T.Color().setRGB((label&255)/255,((label>>8)&255)/255,((label>>16)&255)/255);const result=new T.MeshBasicMaterial({color,side:material.side,toneMapped:false,fog:false,dithering:false});materials.push(result);return result;};mesh.material=Array.isArray(original)?original.map(flat):flat(original);};
        try{
            bodies.forEach((body,index)=>body.bodyRoot.traverse(mesh=>replace(mesh,index+1)));
            scene.background=new T.Color(0);scene.overrideMaterial=null;renderer.toneMapping=T.NoToneMapping;renderer.shadowMap.enabled=false;renderer.autoClear=true;renderer.setRenderTarget(target);renderer.setScissorTest(false);renderer.render(scene,camera);
            const bytes=new Uint8Array(width*height*4);renderer.readRenderTargetPixels(target,0,0,width,height,bytes);
            const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(width,height);
            for(let y=0;y<height;y++)pixels.data.set(bytes.subarray(y*width*4,(y+1)*width*4),(height-1-y)*width*4);
            ctx.putImageData(pixels,0,0);
            // No MSAA/color matching: each visible pixel belongs to exactly one
            // stable body even if several mannequins share the same color.
            return {version:1,source_id:sourceId,width,height,id_map:canvas.toDataURL('image/png'),bodies:bodies.map((body,index)=>({id:body.id,label:index+1,color:`#${body.material.color.getHexString()}`,name:`Human mesh ${body.id.replace(/^body-/,'')}`}))};
        }finally{
            for(const [mesh,material] of replaced)mesh.material=material;materials.forEach(material=>material.dispose());target.dispose();
            scene.background=saved.background;scene.overrideMaterial=saved.override;renderer.toneMapping=saved.tone;renderer.shadowMap.enabled=saved.shadows;renderer.autoClear=saved.autoClear;renderer.setRenderTarget(saved.target);renderer.setViewport(saved.viewport);renderer.setScissor(saved.scissor);renderer.setScissorTest(saved.scissorTest);
        }
    }
    dispose(){this.depthMaterial.dispose();this.normalMaterial.dispose();}
}
