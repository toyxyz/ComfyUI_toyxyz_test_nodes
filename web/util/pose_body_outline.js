import * as T from '../vendor/three/three.module.min.js';

export function bodyOutlineColor(color){
    const hsl=color.getHSL({},T.SRGBColorSpace);
    if(hsl.s<.08)return new T.Color(hsl.l>.5?'#ff00cc':'#00ffff');
    return new T.Color().setHSL((hsl.h+.5)%1,1,.5,T.SRGBColorSpace);
}

// Screen-space inverted hull. Clones share geometry and the posed skeleton,
// but exist in the scene ONLY during beauty rendering, never map captures.
export class PoseBodyOutlines {
    constructor(){this.records=new Map();}
    entries(body){
        if(this.records.has(body))return this.records.get(body);
        const entries=[];body.bodyRoot.traverse(source=>{
            if(!source.isMesh||!source.geometry?.attributes.normal)return;
            const material=new T.ShaderMaterial({side:T.BackSide,toneMapped:false,
                uniforms:{outlineColor:{value:new T.Color()},resolution:{value:new T.Vector2()},thickness:{value:4}},
                vertexShader:`
                    #include <common>
                    #include <skinning_pars_vertex>
                    uniform vec2 resolution;
                    uniform float thickness;
                    void main(){
                        #include <skinbase_vertex>
                        #include <beginnormal_vertex>
                        #include <skinnormal_vertex>
                        vec3 outlineNormal=normalize(normalMatrix*objectNormal);
                        #include <begin_vertex>
                        #include <skinning_vertex>
                        #include <project_vertex>
                        vec2 direction=(projectionMatrix*vec4(outlineNormal,0.0)).xy*resolution;
                        direction/=max(length(direction),0.00001);
                        gl_Position.xy+=direction*(2.0*thickness/resolution)*gl_Position.w;
                    }`,
                fragmentShader:`
                    uniform vec3 outlineColor;
                    void main(){
                        gl_FragColor=vec4(outlineColor,1.0);
                        #include <colorspace_fragment>
                    }`,
            });
            const mesh=source.isSkinnedMesh?new T.SkinnedMesh(source.geometry,material):new T.Mesh(source.geometry,material);
            if(source.isSkinnedMesh){mesh.bindMode=source.bindMode;mesh.bind(source.skeleton,source.bindMatrix);mesh.bindMatrixInverse.copy(source.bindMatrixInverse);}
            mesh.name='Body outline (beauty only)';mesh.renderOrder=-1;mesh.frustumCulled=false;entries.push({source,mesh,material});
        });this.records.set(body,entries);return entries;
    }
    render(renderer,scene,camera,bodies,width,height,enabled){
        if(!enabled){renderer.render(scene,camera);return;}
        const attached=[];
        try{
            for(const body of bodies)for(const {source,mesh,material} of this.entries(body)){
                if(!source.parent)continue;
                material.uniforms.outlineColor.value.copy(bodyOutlineColor(body.material.color));material.uniforms.resolution.value.set(width,height);
                mesh.geometry=source.geometry;mesh.position.copy(source.position);mesh.quaternion.copy(source.quaternion);mesh.scale.copy(source.scale);mesh.visible=source.visible;mesh.layers.mask=source.layers.mask;
                source.parent.add(mesh);attached.push(mesh);
            }
            renderer.render(scene,camera);
        }finally{for(const mesh of attached)mesh.removeFromParent();}
    }
    remove(body){for(const entry of this.records.get(body)??[]){entry.mesh.removeFromParent();entry.material.dispose();}this.records.delete(body);}
    dispose(){for(const body of this.records.keys())this.remove(body);}
}
