import argparse
import os
import time
import torch
from tqdm import tqdm
from lit_gpt.model_cache import GPTCache, Config
from safetensors.torch import load_file
from sft.datasets.dataset import Sample_Dataset
import os
from tqdm import tqdm
import trimesh
from sft.datasets.serializaiton import deserialize, serialize
from sft.datasets.data_utils import to_mesh, process_mesh_xr

import numpy as np
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pymeshlab

import random

def set_deterministic(seed=2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭非确定性算法&加速
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def set_random():
    seed = np.random.randint(0, 10000) + time.time_ns() % 1000
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_distributed_mode(rank, world_size, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed_mode():
    dist.destroy_process_group()

def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

@ torch.no_grad()
def ar_sample_kvcache(gpt, prompt, pc, global_pc=None, bd_token=None, temperature=0.5, \
                        context_length=25000, window_size=9000,device='cuda',\
                        output_path=None,local_rank=None,i=None,part_idx=None,ablation_sfa=False):
    gpt.eval()
    N        = prompt.shape[0]
    bd = bd_token
    print(f'start sampling with prompt length:{prompt.shape[1]}')
    end_list = [0 for _ in range(N)]
    time_list = []
    with tqdm(total=context_length-1, desc="Processing") as pbar:
        for cur_pos in range(prompt.shape[1], context_length):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if cur_pos >= 9001 and (cur_pos - 9001)%4500 == 0:
                    start = 4500 + ((cur_pos - 9001)//4500)*4500
                else:
                    start = cur_pos-1
                input_pos    = torch.arange(cur_pos, dtype=torch.long, device=device)
                prompt_input = prompt[:, start:cur_pos]
                start_time = time.time()
                logits = gpt(prompt_input, pc=pc, global_pc=global_pc, bd_token=bd_token, start = start, window_size=window_size, input_pos=input_pos)[:, -1]
                end_time = time.time()
                time_list.append(end_time - start_time)

                pc     = None
                global_pc = None
                bd_token = None
            logits_with_noise = add_gumbel_noise(logits, temperature)
            next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)
                        
            if ablation_sfa:
                prompt = torch.cat([prompt, next_token], dim=-1)
            else:
                new_prompt = []
                for u in range(N):
                    if bd.shape[1] > cur_pos:
                        if bd[u,cur_pos] != 4737:
                            new_prompt.append(bd[u,cur_pos])
                        else:
                            new_prompt.append(next_token[u])
                    else:
                        new_prompt.append(next_token[u])
                prompt = torch.cat([prompt, torch.tensor(new_prompt, dtype=torch.long, device=device).unsqueeze(1)], dim=-1)


            pbar.set_description(f"with start:{start},cur_pos:{cur_pos},length:{prompt_input.size(1)},part_idx:{part_idx}")
            pbar.update(1)
            
                
            for u in range(N):
                if end_list[u] == 0:
                    if next_token[u] == torch.tensor([4737], device=device):
                        end_list[u] = 1
            if sum(end_list) == N:
                break
    return prompt, cur_pos, time_list

def first(it):
    return it[0]

def recover_verts(verts, center, scale):
    verts = verts * scale
    verts = verts + center
    verts = verts[:, [2, 0, 1]]
    return verts
def recover_verts2(verts, center, scale):
    verts = verts / scale
    verts = verts + center
    # verts = verts[:, [2, 0, 1]]
    return verts
def process_verts(verts, center, scale):
    # Transpose so that z-axis is vertical.
    verts = verts[:, [2, 0, 1]]
    verts = verts - center
    verts = verts / scale
    return verts

def mesh_filter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """使用PyMeshLab进行网格预处理"""
    ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(ml_mesh, "raw_mesh")
            
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('meshing_merge_close_vertices', 
                    threshold=pymeshlab.PercentageValue(0.5))
    ms.apply_filter('meshing_remove_unreferenced_vertices')
            
    processed = ms.current_mesh()
    mesh.vertices = processed.vertex_matrix()
    mesh.faces = processed.face_matrix()
            
    return mesh

def preprocess_mesh( mesh):
    mesh = mesh_filter(mesh)
        # 归一化处理
    vertices = mesh.vertices
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center, scale = (bbmin + bbmax)*0.5, 2.0 * 0.9 / (bbmax - bbmin).max()
    mesh.vertices = (vertices - center) * scale
        # 确保三角形网格
    if mesh.faces.shape[1] == 4:
        mesh.faces = np.vstack([mesh.faces[:, :3], mesh.faces[:, [0,2,3]]])
        
    return mesh


def custom_collate(data, pad_id):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output

def build_dataloader_func(bs, dataset, local_rank, world_size):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=0,
        drop_last = False,
        collate_fn = partial(custom_collate, pad_id = 4737)
    )
    return dataloader

@torch.inference_mode()
def get_model_answers(
    local_rank,
    world_size
):
    model_path  = args.model_path
    model_id    = args.model_id
    output_path = args.output_path # + f"/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    steps       = args.steps
    temperature = args.temperature
    path        = args.input_path
    # output_path = args.output_path
    point_num   = args.point_num
    uid_list    = args.uid_list.split(",")
    repeat_num  = args.repeat_num
    random_seed = args.random_seed
    ablation_sfa = args.ablation_sfa if hasattr(args, 'ablation_sfa') else False
    ablation_gpc = args.ablation_gpc if hasattr(args, 'ablation_gpc') else False
    ablation_bdt = args.ablation_bdt if hasattr(args, 'ablation_bdt') else False


    if random_seed != -1:
        set_deterministic(random_seed)
    else:
        set_random()

    setup_distributed_mode(local_rank, world_size)
    model_name = f"Diff_LLaMA_{model_id}M"
    config = Config.from_name(model_name)
    config.padded_vocab_size=(2*4**3)+(8**3)+(16**3) +1 +1  #4736+2
    config.block_size = 270000
    model = GPTCache(config).to('cuda')
    
    model.ablation_gpc = ablation_gpc
    model.ablation_bdt = ablation_bdt

    if model_path.split(".")[-1]=="safetensors":
        loaded_state = load_file(model_path)
    elif model_path.split(".")[-1]=="bin":
        loaded_state = torch.load(model_path, map_location='cpu',weights_only=False)
    if 'model_state_dict' in loaded_state:
        model.load_state_dict(loaded_state['model_state_dict'], strict=True)
    else:
        model.load_state_dict(loaded_state, strict=True)
    model       = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
    train_dataset    = Sample_Dataset(point_num = point_num,uid_list = uid_list,path=path)
    train_dataloader = build_dataloader_func(1,train_dataset, local_rank, world_size)
    
    all_time_list = []

    for i, test_batch in tqdm(enumerate(train_dataloader)):
        
        data_len = test_batch['len']
        file_name = test_batch['name'][0]
        args.output_name = file_name

        os.makedirs(f'{output_path}/{file_name}', exist_ok=True)
        
        for part_idx in range(data_len[0]+1):
            all_verts = [[] for _ in range(repeat_num)]
            all_faces = [[] for _ in range(repeat_num)]
            for u in range(repeat_num):
                if part_idx == 0:
                    break
                mesh = trimesh.load(f'{output_path}/{file_name}/{args.output_name}_{local_rank}_{i}_{0}_{u}_mesh_recover.obj')
                all_verts[u]=mesh.vertices
                all_faces[u]=mesh.faces
                for part_idxx in range(part_idx):
                    if part_idxx == 0:
                        continue
                    try:
                        mesh = trimesh.load(f'{output_path}/{file_name}/{args.output_name}_{local_rank}_{i}_{part_idxx}_{u}_mesh_recover.obj')
                    except:
                        continue
                    part_verts = mesh.vertices
                    part_faces = mesh.faces
                    vertex_offset = len(all_verts[u])
                    all_verts[u] = np.concatenate([all_verts[u], part_verts], axis=0)
                    all_faces[u] = np.concatenate([all_faces[u], part_faces + vertex_offset], axis=0)
                mesh = trimesh.Trimesh(vertices=all_verts[u], faces=all_faces[u])
                mesh.export(f'{output_path}/{file_name}/000Final_ALL_{args.output_name}_{local_rank}_{i}_{u}_mesh.obj') 
            
                if part_idx == data_len[0]:
                    # use pymeshlab to post-process the mesh in f'{output_path}/{file_name}/000Final_ALL_{args.output_name}_{local_rank}_{i}_{u}_mesh.obj'
                    try:
                        ms = pymeshlab.MeshSet()
                        ms.load_new_mesh(f'{output_path}/{file_name}/000Final_ALL_{args.output_name}_{local_rank}_{i}_{u}_mesh.obj')
                        ms.meshing_repair_non_manifold_edges(method=1)
                        ms.meshing_repair_non_manifold_edges(method=0)
                        ms.meshing_close_holes(maxholesize = 1000, refinehole = True)
                        ms.meshing_repair_non_manifold_edges(method=1)
                        ms.meshing_repair_non_manifold_edges(method=0)
                        ms.meshing_close_holes(maxholesize = 1000, refinehole = True)
                        ms.meshing_repair_non_manifold_edges(method=1)
                        ms.meshing_repair_non_manifold_edges(method=0)
                        ms.meshing_close_holes(maxholesize = 1000, refinehole = True)
                        os.makedirs(f'{output_path}/Finaloutputs', exist_ok=True)
                        ms.save_current_mesh(f'{output_path}/Finaloutputs/Final_Fix_{args.output_name}_{local_rank}_{i}_{u}_mesh.obj') 
                    except:
                        
                        ms = trimesh.load(f'{output_path}/{file_name}/000Final_ALL_{args.output_name}_{local_rank}_{i}_{u}_mesh.obj')
                        ms.export(f'{output_path}/Finaloutputs/Final_Fix_{args.output_name}_{local_rank}_{i}_{u}_mesh.obj') 
                        continue
            if part_idx == data_len[0]:
                break

            cond_pc = test_batch['pc_local_'+str(part_idx)].to('cuda')
            global_pc = test_batch['pc_global'].to('cuda')
            bd_token_lengths = test_batch['bd_token_length_'+str(part_idx)]
            gt_tokens = test_batch['token_list_'+str(part_idx)].to('cuda')
            center = test_batch['center_'+str(part_idx)]
            scale = test_batch['scale_'+str(part_idx)]
            cluster = test_batch['cluster_'+str(part_idx)]
            
            bd_tokens = torch.tensor([], dtype=torch.long, device=gt_tokens.device)
            bdmesh_ori = [trimesh.Trimesh(vertices=all_verts[u], faces=all_faces[u]) for u in range(repeat_num)]
            for u in range(repeat_num):
                if part_idx == 0 or len(all_faces[u]) <= 20:
                    bd_token = torch.ones(2048, dtype=torch.long, device=gt_tokens.device) * 4737
                    bd_token = bd_token.unsqueeze(0)
                    bd_tokens = torch.cat([bd_tokens, bd_token], dim=0)
                    
                else: 
                    
                    bdmesh = trimesh.Trimesh(vertices=all_verts[u], faces=all_faces[u])
                    bdmesh.vertices = process_verts(bdmesh.vertices, center, scale)
                    bdmesh.vertices = bdmesh.vertices[..., [2, 0, 1]]
                    face_points = np.array(bdmesh.vertices[bdmesh.faces])

                    localpc = cond_pc[0].cpu().numpy()
                    
                    localpc = localpc[np.random.choice(localpc.shape[0], 512, replace=False),:3]
                    meshpc = bdmesh.vertices
                    
                    mesh_distances = np.ones(len(meshpc)) * 999999999.0
                    for idx in range(len(meshpc)):
                        for jdx in range(len(localpc)):
                            dist = np.linalg.norm(meshpc[idx] - localpc[jdx])
                            if dist < mesh_distances[idx]:
                                mesh_distances[idx] = dist
                    facedis = np.min(mesh_distances[bdmesh.faces], axis=1)

                    use_num = 512
                    while True:
                        boundry_faces = np.argsort(facedis)[:use_num]
                        tmpmesh = trimesh.Trimesh(vertices=bdmesh.vertices, faces=bdmesh.faces[boundry_faces])
                        
                        tmpmesh.vertices = tmpmesh.vertices[..., [1, 2, 0]]
                        bdmesh_ori[u] = tmpmesh
                        try:
                            tmpmesh = process_mesh_xr(tmpmesh.vertices, tmpmesh.faces, quan = True, quantization_bits=8, augment=False)
                            tmpmesh = trimesh.Trimesh(vertices=tmpmesh['vertices'], faces=tmpmesh['faces'])
                        except:
                            tmpmesh.export(f'{output_path}/{file_name}/{args.output_name}_{local_rank}_{i}_{part_idx}_{u}_bd_mesh.obj') 
                            bd_token = torch.ones(2048, dtype=torch.long, device=gt_tokens.device) * 4737
                            bd_token = np.array(bd_token.cpu())
                            break
                        bd_token = serialize(tmpmesh)
                        if len(bd_token) > 2000:
                            use_num = use_num - 32
                            continue
                        else:
                            break
                    
                    bd_token = np.array(bd_token)
                    if bd_token.shape[0] == 2048 and bd_token[0] == 4737 and bd_token[-1] == 4737:
                        bd_token = bd_token
                    else:
                        bd_token = np.concatenate([[4736], bd_token, [4737]])
                    bd_token = torch.tensor(bd_token, dtype=torch.long, device=gt_tokens.device)
                    if bd_token.shape[0] < 2048:
                        bd_token = torch.cat([bd_token, torch.ones(2048-bd_token.shape[0], dtype=torch.long, device=bd_token.device) * 4737], dim=0)
                    else:
                        bd_token = bd_token[:2048]
                    bd_tokens = torch.cat([bd_tokens, bd_token.unsqueeze(0)], dim=0)
                    
            print(f"bd_tokens:{bd_tokens.shape}")

            points = cond_pc[0].cpu().numpy()
            point_cloud = trimesh.points.PointCloud(points[..., 0:3])
            point_cloud.export(f'{output_path}/{file_name}/{args.output_name}_{local_rank}_{i}_{part_idx}_local_pc.ply')
            points = global_pc[0].cpu().numpy()
            point_cloud = trimesh.points.PointCloud(points[..., 0:3])
            point_cloud.export(f'{output_path}/{file_name}/{args.output_name}_{local_rank}_{i}_global_pc.ply')

            output_ids, _, time_list = ar_sample_kvcache(model,
                                prompt = torch.tensor([[4736]]).to('cuda').repeat(repeat_num,1),
                                pc = cond_pc.repeat(repeat_num,1,1),
                                global_pc = global_pc.repeat(repeat_num,1,1),
                                bd_token = bd_tokens,
                                window_size=9000,
                                temperature=temperature,
                                context_length=steps,
                                device='cuda',
                                ablation_sfa=ablation_sfa,
                                output_path=output_path,local_rank=local_rank,i=i,part_idx=part_idx)
            all_time_list = all_time_list + time_list
            print(f"all_time_list:{all_time_list}, avg time:{sum(all_time_list)/len(all_time_list)}")
            for u in range(repeat_num):
                code = output_ids[u]
                bd_code = bd_tokens[u]
                start = 1
                
                if ablation_sfa:
                    start = 1
                else:
                    for pos in range(bd_tokens.shape[1]):
                        start = pos
                        if 4737 == bd_tokens[u,pos]:
                            break
                code = code[start:]
                
                index = (code >= 4737).nonzero()
                if index.numel() > 0:
                    code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
                else:
                    code = code.cpu()
                try:
                    vertices = deserialize(code)
                except:
                    print("you got:",len(vertices))
                    continue
                if len(vertices) == 0:
                    print("you got:",len(vertices))
                    continue
                # vertices = vertices[..., [2, 1, 0]]
                try:
                    faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
                    mesh = to_mesh(vertices, faces, transpose=False, post_process=False)
                except:
                    print("you got:",len(vertices))
                    continue

                if(part_idx != 0):
                    index_bd = (bd_code >= 4737).nonzero()
                    if index_bd.numel() > 0:
                        bd_code = bd_code[:index_bd[0, 0].item()].cpu().numpy().astype(np.int64)
                    else:
                        bd_code = bd_code.cpu()
                    try:
                        bd_vertices = deserialize(bd_code)
                    
                        bd_faces = torch.arange(1, len(bd_vertices) + 1).view(-1, 3)
                        bd_mesh = to_mesh(bd_vertices, bd_faces, transpose=False, post_process=False)
                        bdddmesh = trimesh.Trimesh(vertices=bd_mesh.vertices, faces=bd_mesh.faces)

                        v_bdori = bdmesh_ori[u].vertices
                        v_bd = bdddmesh.vertices
                        v_new = mesh.vertices
                        bdpids = []
                        for vid in range(len(v_new)):
                            v_now = v_new[vid]
                        #  计算到 v_bd 的距离
                            dist = np.linalg.norm(v_now - v_bd, axis=1)
                            if min(dist) < 0.00001:
                                bdpids.append(vid)
                        for pid in bdpids:
                            v_now = v_new[pid]
                            dist = np.linalg.norm(v_now - v_bdori, axis=1)
                            v_new[pid] = v_bdori[dist.argmin()]
                        mesh = trimesh.Trimesh(vertices=v_new, faces=mesh.faces)
                    except:
                        print(f"Error in part_idx {part_idx} of i {i}, decodeing buondary mesh error.")
                        # continue

                mesh.vertices = recover_verts(mesh.vertices, center, scale)
                mesh.vertices = mesh.vertices[..., [2, 0, 1]]
                mesh.export(f'{output_path}/{file_name}/{args.output_name}_{local_rank}_{i}_{part_idx}_{u}_mesh_recover.obj') 
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model_id", type=str, default="551", help="A custom name for the model."
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True, 
        help="sampling steps.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./output_pc_aug'
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default='11'
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=4
    )
    parser.add_argument(
        "--point_num",
        type=int,
        default=16384
    )
    parser.add_argument(
        "--uid_list",
        type=str,
        default=''
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--ablation_sfa",
        action="store_true",
        help="Whether to use self attention ablation."
    )
    parser.add_argument(
        "--ablation_bdt",
        action="store_true",
        help="Whether to use boundary detection ablation."
    )
    parser.add_argument(
        "--ablation_gpc",
        action="store_true",
        help="Whether to use global point cloud ablation."
    )
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    get_model_answers(
                local_rank=local_rank,
                world_size=world_size
    )
