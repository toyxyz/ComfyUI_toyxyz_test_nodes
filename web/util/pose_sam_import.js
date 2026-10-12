// Export only the connected loader's ancestors; never queue the whole workflow.
export async function buildSAMPosePrompt(node,image,requestId){
    const prompt={pose_image:{class_type:'LoadImage',inputs:{image}},pose_import:{class_type:'ToyxyzPose3DSAMImport',inputs:{image:['pose_image',0],request_id:requestId}}};
    const graph=node.graph,visiting=new Set(),complete=new Set();
    const getLink=id=>graph?.getLink?.(id)??graph?.links?.[id];
    async function reference(linkId){
        const link=getLink(linkId),source=graph?.getNodeById(link?.origin_id);
        if(!source)throw Error('The connected SAM model loader is unavailable.');
        // Reroutes have no Python counterpart. Follow their actual input link.
        if(source.isVirtualNode){
            if(source.inputs?.[0]?.link!=null)return reference(source.inputs[0].link);
            throw Error('Connect a SAM 3D Body loader directly or through a reroute.');
        }
        const id=String(source.id);
        if(visiting.has(id))throw Error('The SAM model connection contains a cycle.');
        if(!complete.has(id)){
            if(source.mode!==undefined&&source.mode!==0)throw Error('The connected SAM model loader must be active.');
            visiting.add(id);const inputs={};
            for(const widget of source.widgets??[]){
                if(widget.options?.serialize===false||widget.serialize===false)continue;
                inputs[widget.name]=widget.serializeValue?await widget.serializeValue(source,source.widgets.indexOf(widget)):widget.value;
            }
            for(const input of source.inputs??[])if(input.link!=null)inputs[input.name]=await reference(input.link);
            prompt[id]={class_type:source.comfyClass??source.type,inputs};visiting.delete(id);complete.add(id);
        }
        return [id,link.origin_slot];
    }
    const socket=node.inputs?.find(input=>input.name==='sam3d_body_model');
    if(socket?.link!=null)prompt.pose_import.inputs.sam3d_body_model=await reference(socket.link);
    else {prompt.pose_model={class_type:'ToyxyzPose3DSAMLoader',inputs:{model_file:'sam_3d_body_dinov3_bf16.safetensors'}};prompt.pose_import.inputs.sam3d_body_model=['pose_model',0];}
    return prompt;
}

export function samBodyFingerprint(data){
    const {openpose,...state}=data;return JSON.stringify(state);
}

// Preflight only the directly selected, known default loader. Never download
// substitute weights for custom filenames or unrelated upstream nodes.
export async function ensureSAMPoseModel(prompt,api,report,isActive=()=>true){
    const reference=prompt.pose_import?.inputs?.sam3d_body_model;
    const loader=Array.isArray(reference)?prompt[reference[0]]:null;
    if(!['ToyxyzPose3DSAMLoader','SAM3DBody_Loader'].includes(loader?.class_type)||loader.inputs?.model_file!=='sam_3d_body_dinov3_bf16.safetensors')return;
    report('Get pose: Checking SAM 3D Body model…');
    let response=await api.fetchApi('/toyxyz/pose3d/sam-model',{method:'POST'});
    let state=await response.json();if(!response.ok)throw Error(state.error??`Model check: HTTP ${response.status}`);
    while(isActive()){
        if(state.state==='ready')return;
        if(state.state==='error')throw Error(state.error??'SAM 3D Body download failed.');
        if(state.state!=='downloading')throw Error('Unexpected SAM model download status.');
        const done=Number(state.downloaded)||0,total=Number(state.total)||0;
        const amount=`${(done/1e9).toFixed(2)} / ${(total/1e9).toFixed(2)} GB`;
        report(`Get pose: Downloading SAM 3D Body — ${total?Math.floor(done/total*100):0}% (${amount})${done===total?' · Verifying…':''}`);
        await new Promise(resolve=>setTimeout(resolve,1000));
        if(!isActive())break;
        response=await api.fetchApi('/toyxyz/pose3d/sam-model');state=await response.json();
        if(!response.ok)throw Error(state.error??`Download status: HTTP ${response.status}`);
    }
    throw Error('Get pose was closed while preparing the model.');
}
