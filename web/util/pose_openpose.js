import * as T from '../vendor/three/three.module.min.js';

// COCO-18 ordering (anatomical left/right, never screen left/right).
export const OPENPOSE_NAMES = ['nose','neck','right_shoulder','right_elbow','right_wrist',
    'left_shoulder','left_elbow','left_wrist','right_hip','right_knee','right_ankle',
    'left_hip','left_knee','left_ankle','right_eye','left_eye','right_ear','left_ear'];
// Landmark IDs for the standard 6890-vertex SMPL/SMPL-H topology, as documented
// by smplx/vertex_ids.py. No SMPL weights or application code are bundled.
const SMPL_FACE = {nose:[332],right_eye:[6260],left_eye:[2800],right_ear:[4071],left_ear:[583]};
const FACE_NORMALS = {nose:[0,0,1],right_eye:[-.45,0,1],left_eye:[.45,0,1],right_ear:[-1,0,0],left_ear:[1,0,0]};

export class PoseOpenPose {
    constructor(bones,mesh,source,mapping) {
        this.bones=bones;this.mesh=mesh;this.source=source;
        this.byName=new Map(bones.map(bone=>[bone.name,bone]));
        this.mapping=mapping;
        this.head=this.byName.get(mapping?.bones?.head??'head');
        // MHR's head has a non-identity bind orientation. Surface directions are
        // defined in neutral world space, then carried by the posed head.
        const inverseBind=mapping?.neutralFaceNormals?this.head.getWorldQuaternion(new T.Quaternion()).invert():new T.Quaternion();
        this.faceNormals=Object.fromEntries(Object.entries(FACE_NORMALS).map(([name,normal])=>[name,new T.Vector3(...normal).normalize().applyQuaternion(inverseBind)]));
        this.face=mesh?.geometry.attributes.position.count===6890?SMPL_FACE:null;
        // Other user template topologies have no reliable vertex IDs. Provide
        // explicitly approximate, head-attached landmarks rather than detecting.
        const head=this.head?.getWorldPosition(new T.Vector3())??new T.Vector3(0,1.7,0);
        const neck=this.byName.get(mapping?.bones?.neck??'neck')?.getWorldPosition(new T.Vector3());
        const size=neck?T.MathUtils.clamp(head.distanceTo(neck),.08,.2):.115;
        this.proxyScale=size/.115;
    }
    worldPoint(name) {
        // OpenPose's virtual Neck is the shoulder midpoint, not the anatomical
        // neck bone pivot. Derive it from both real shoulder bones.
        if(name==='neck') {
            const left=this.worldPoint('left_shoulder'),right=this.worldPoint('right_shoulder');
            if(left&&right)return left.add(right).multiplyScalar(.5);
        }
        if(FACE_NORMALS[name]) {
            const landmark=this.mapping?.landmarks?.[name];
            if(landmark&&this.mesh){
                const result=new T.Vector3(),point=new T.Vector3();
                for(const [index,weight] of landmark.vertices)result.addScaledVector(this.mesh.getVertexPosition(index,point).applyMatrix4(this.mesh.matrixWorld),weight);
                for(const [index,weight] of landmark.bones)result.addScaledVector(this.bones[index].getWorldPosition(point),weight);
                return result;
            }
            const indices=this.face?.[name];
            if(indices&&this.mesh) {
                const result=new T.Vector3(),point=new T.Vector3();
                for(const index of indices)result.add(this.mesh.getVertexPosition(index,point));
                return result.divideScalar(indices.length).applyMatrix4(this.mesh.matrixWorld);
            }
            if(!this.head)return null;
            const offsets={nose:[0,.012,.12],right_eye:[-.032,.043,.105],left_eye:[.032,.043,.105],
                right_ear:[-.09,.035,-.012],left_ear:[.09,.035,-.012]};
            return this.head.localToWorld(new T.Vector3(...offsets[name]).multiplyScalar(this.proxyScale));
        }
        const bone=this.byName.get(this.mapping?.bones?.[name]??name);
        return bone?.getWorldPosition(new T.Vector3())??null;
    }
    faceVisible(name,point,camera) {
        if(!FACE_NORMALS[name]||!this.head)return true;
        const view=camera.getWorldPosition(new T.Vector3()).sub(this.head.getWorldPosition(new T.Vector3())).normalize();
        const normal=this.faceNormals[name].clone().applyQuaternion(this.head.getWorldQuaternion(new T.Quaternion()));
        // Nose/eyes must face the camera. Ears sit at the silhouette, so both
        // remain visible in exact front/back views; a far-side ear is hidden.
        return normal.dot(view)>(name.endsWith('ear')?-.22:name==='nose'?-.12:.015);
    }
    project(camera,width,height) {
        this.bones[0]?.updateWorldMatrix(true,true);this.mesh?.updateWorldMatrix(true,false);
        // Attached SkinnedMesh updates bindMatrixInverse in updateMatrixWorld,
        // not updateWorldMatrix. Root moves/Undo must project the current skin,
        // even before a renderer frame refreshes that inverse.
        this.mesh?.updateMatrixWorld(true);
        this.mesh?.skeleton.update();camera.updateMatrixWorld(true);
        const keypoints=OPENPOSE_NAMES.map(name=>{
            const world=this.worldPoint(name);if(!world||!this.faceVisible(name,world,camera))return null;
            const ndc=world.project(camera);
            if(![ndc.x,ndc.y,ndc.z].every(Number.isFinite)||ndc.z < -1||ndc.z > 1)return null;
            // Keep offscreen XY endpoints: the rasterizer clips their limbs,
            // rather than silently recentering the model or discarding a limb.
            return [(ndc.x+1)*width/2,(1-ndc.y)*height/2];
        });
        return {version:1,width,height,keypoints};
    }
}
