import torch
import os
import json
from typing import Dict
from pathlib import Path
import numpy as np
import dataclasses
import trimesh
import open3d as o3d
from sft.datasets.data_utils import load_process_mesh, center_vertices, normalize_vertices_scale, process_mesh, process_mesh_xr
from sft.datasets.serializaiton import serialize
from sft.datasets.serializaiton import deserialize
from sft.datasets.data_utils import to_mesh
from utils.common import init_logger, import_module_or_data
from collections import defaultdict

from datetime import datetime
from tqdm import tqdm
import time
import pymeshlab
import open3d as o3d
import igl

import matplotlib.pyplot as plt
logger = init_logger()

SYNSET_DICT_DIR = Path(__file__).resolve().parent  

class DynamicAttributes:
    def __init__(self):
        self._storage = {}

    def __setattr__(self, name, value):
        if name == '_storage':
            super().__setattr__(name, value)
        else:
            self._storage[name] = value

    def __getattr__(self, name):
        if name not in self._storage:
            new_obj = DynamicAttributes()
            self._storage[name] = new_obj
        return self._storage[name]
    
def sample_pc(verts, faces, pc_num, with_normal=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points
    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points[:,[2,0,1]], normals[:,[2,0,1]]], axis=-1, dtype=np.float16)
    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    return pc_normal



def selection_mesh_from_connected_components(verts, faces, components, fix=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    verts = mesh.vertices
    faces = mesh.faces

    indices  = np.random.choice(50000, 16384, replace=False)
    pc_global_data       = sample_pc(verts, faces, pc_num=50000, with_normal=True)[indices]

    cluster_num = len(components)

    class_idx = -1
    minz = 1000000000
    for i in range(len(components)):
        verts_i = components[i].vertices
        if np.min(verts_i[:, 2]) < minz:
            minz = np.min(verts_i[:, 2])
            class_idx = i


    new_faces = []
    cpverts = verts.copy()  
    data = []
    used_cluster = []


    visited = set()
    queue = [class_idx]

    reconed_meshs = []
    idx = -1
    last_verts = np.zeros((0, 3), dtype=components[0].vertices.dtype)
    last_faces = np.zeros((0, 3), dtype=components[0].faces.dtype)
    while len(visited) < cluster_num:
        
        if len(queue) == 0:
            for j in range(len(components)):
                if j not in visited:
                    queue.append(j)
                    break

        cluster = queue.pop(0)
        if cluster in visited:
            continue 
        visited.add(cluster)
        idx += 1
        
        for j in range(len(components)):
            if j not in visited:
                verts_j = components[j].vertices
                verts_i = components[cluster].vertices
                
                dists = np.linalg.norm(verts_j[:, None, :] - verts_i[None, :, :], axis=-1)
                dist = np.min(dists)
                if dist < 0.01:
                    queue.append(j)
                    
        data_i = {}
        data_i['idx'] = idx
       
        
        selected_verts = components[cluster].vertices
        selected_faces = components[cluster].faces
        if len(selected_faces) + len(last_faces) <= 1000 and len(visited) < cluster_num and False:
            lenvv = len(last_verts)
            last_verts = np.concatenate([last_verts, selected_verts], axis=0)
            last_faces = np.concatenate([last_faces, selected_faces + lenvv], axis=0)
            continue
        else:
            selected_verts = np.concatenate([last_verts, selected_verts], axis=0)
            selected_faces = np.concatenate([last_faces, selected_faces+len(last_verts)], axis=0)
            last_verts = np.zeros((0, 3), dtype=components[0].vertices.dtype)
            last_faces = np.zeros((0, 3), dtype=components[0].faces.dtype)
     
        if len(selected_verts) <= 3 or len(selected_faces) <= 3:
            data_i['skip'] = True
            data_i['bd_token_length'] = 0
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data_i['cluster'] = None
            data.append(data_i)
            continue
        
        if idx == 0:
            bdd = False
        else:
            bdd = True
        if idx == 0:
            data_i['bd_token_length'] = 0
            
            bdd = False
        else:
            boundry_verts = np.zeros((0, 3), dtype=components[cluster].vertices.dtype)
            boundry_faces = np.zeros((0, 3), dtype=components[cluster].faces.dtype)
            for j in visited:
                verts_j = components[j].vertices
                faces_j = components[j].faces
                lenv = len(boundry_verts)
                boundry_verts = np.concatenate([boundry_verts, verts_j], axis=0)
                boundry_faces = np.concatenate([boundry_faces, faces_j + lenv], axis=0)
            bdmesh = trimesh.Trimesh(vertices=boundry_verts, faces=boundry_faces)
            face_points = bdmesh.vertices[bdmesh.faces]
            
            distances = np.ones(bdmesh.faces.shape[0]) * 1000000000
            for iddx in range(bdmesh.faces.shape[0]):
                distances[iddx] = np.min(np.linalg.norm(face_points[iddx][0] - selected_verts, axis=-1))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][1] - selected_verts, axis=-1)))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][2] - selected_verts, axis=-1)))
            
            boundry_faces = np.argsort(distances)[:512]
            
            tmpmesh = trimesh.Trimesh(vertices=bdmesh.vertices, faces=bdmesh.faces[boundry_faces])
            boundry_verts = tmpmesh.vertices
            boundry_faces = tmpmesh.faces

        seleVL = selected_verts.shape[0]
        if bdd:
            all_verts = np.concatenate([selected_verts, boundry_verts], axis=0)
        else:
            all_verts = selected_verts
           
        if len(all_verts) <= 10 or len(selected_faces) <= 10:
            data_i['bd_token_length'] = 0
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data.append(data_i)
            continue
        all_verts, center, scale = rescale_verts(all_verts)
        data_i['center'] = center
        data_i['scale'] = scale
        localmesh = trimesh.Trimesh(vertices=all_verts[:seleVL], faces=selected_faces)
        try:
            localmesh = process_mesh_xr(localmesh.vertices, localmesh.faces, augment=False)
        except:
            print(f"localmesh.vertices.shape: {localmesh.vertices.shape}, localmesh.faces.shape: {localmesh.faces.shape}")
            data_i['bd_token_length'] = 0
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data.append(data_i)
            continue
        selected_verts = localmesh['vertices']
        selected_faces = localmesh['faces']
        if bdd:
            if len(all_verts[seleVL:]) <= 10 or len(boundry_faces) <= 20:
                boundary_verts = []
                boundary_faces = []
                data_i['bd_token_length'] = 0
                bdd = False
            else:
                localbdmesh = trimesh.Trimesh(vertices=all_verts[seleVL:], faces=boundry_faces)
                
                try:
                    localbdmesh = process_mesh_xr(localbdmesh.vertices, localbdmesh.faces, augment=False)
                except:
                    print(f"localbdmesh.vertices.shape: {localbdmesh.vertices.shape}, localbdmesh.faces.shape: {localbdmesh.faces.shape}")
                    data_i['bd_token_length'] = 0
                    boundary_verts = []
                    boundary_faces = []
                    bdd = False
                else:
                    boundary_verts = localbdmesh['vertices']
                    boundary_faces = localbdmesh['faces']
        else:
            boundary_verts = []
            boundary_faces = []
        token_list = serialize(trimesh.Trimesh(vertices=selected_verts, faces=selected_faces))
        if bdd:
            bd_token_list = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces))

            facetodelete = 32
            while len(bd_token_list) > 2000:
                bd_token_list = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces[:-facetodelete]))
                facetodelete += 32
            block_size = 8
            offset_size = 16
            patch_size = 4
            special_block_base = block_size**3 + offset_size**3 + patch_size**3
            token_list[0] += special_block_base
            token_list = np.concatenate([[4736], bd_token_list, token_list, [4737]])
            data_i['bd_token_length'] = len(bd_token_list)
        else:
            token_list = np.concatenate([[4736], token_list, [4737]])
        data_i['token_list'] = torch.tensor(token_list, dtype=torch.long)





        indices = np.random.choice(50000, 16384, replace=False)
        pc_local = sample_pc(selected_verts, selected_faces, pc_num=50000, with_normal=True)[indices]
        data_i['pc_local'] = torch.tensor(pc_local)
        data_i['pc_global'] = torch.tensor(pc_global_data)
        data_i['cluster'] = None
        data_i['skip'] = False
        
        used_cluster.append(cluster)

        data.append(data_i)
    return data


def rescale_verts(verts):
    # Transpose so that z-axis is vertical.
    verts = verts[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    center = (verts.max(0) + verts.min(0)) / 2
    verts = verts - center

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    scale = np.sqrt(np.sum((verts.max(0) - verts.min(0)) ** 2))
    verts = verts / scale

    return verts, center, scale

@dataclasses.dataclass
class MeshMeta:
    model_id: str
    raw_obj: str = None
    face_info: dict = None
    category: str = None
    category_path_en: str = None
    obj_path: str = None
    model_type: str = None
    face_cnt: int = None

class UnionSet:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: 
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

class Sample_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        quant_bit: int     = 9,
        point_num:int      = 16384,
        path:str           = "",
        uid_list:list      = []
    ) -> None:
        super().__init__()
        self.quant_bit    = quant_bit
        self.point_num    = point_num
        self.path         = path
        
        name              = os.listdir(path)
        if uid_list == [] or uid_list == [""]:
            self.uid_list     = [i for i in name if len(i.split("."))>1]
        else:
            self.uid_list    = uid_list
        print("dataset init, dataset length: ", len(self.uid_list))
    
    def __len__(self) -> int:
        return len(self.uid_list)

    def _preprocess_mesh(self, mesh):
        # mesh = self._mesh_filter(mesh)

        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center, scale = (bbmin + bbmax)*0.5, 2.0 * 0.9 / (bbmax - bbmin).max()
        mesh.vertices = (vertices - center) * scale
        
        if mesh.faces.shape[1] == 4:
            mesh.faces = np.vstack([mesh.faces[:, :3], mesh.faces[:, [0,2,3]]])
        
        return mesh

    def _mesh_filter(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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

    def _pairwise_edges(self, face: np.ndarray):
        k = face.shape[0]
        if k < 2:
            return []
        return [(int(face[i]), int(face[(i + 1) % k])) for i in range(k)]

    def split_mesh_into_connected_components_union_set(self, mesh_vertices, mesh_faces):
        if mesh_vertices is None or mesh_faces is None:
            print("faces or vertices is None")
            return None
        if mesh_vertices.size == 0 or mesh_faces.size == 0:
            print(f"faces shape={mesh_faces.shape}, vertices shape={mesh_vertices.shape}")
            return None

        vertices = np.asarray(mesh_vertices)
        faces = np.asarray(mesh_faces)
        if faces.ndim != 2:
            raise ValueError(f"faces should be a 2D integer array, current ndim={faces.ndim}")
        if faces.dtype.kind not in ('i', 'u'):
            faces = faces.astype(np.int64)

        N = vertices.shape[0]
        uf = UnionSet(N)


        used_vertices = np.unique(faces)

        if used_vertices.size and (used_vertices.min() < 0 or used_vertices.max() >= N):
            raise IndexError("faces contains out-of-bound vertex indices")


        for f in faces:
            for u, v in self._pairwise_edges(f):
                uf.union(u, v)


        roots_per_face = np.array([uf.find(int(f[0])) for f in faces], dtype=np.int64)


        root_to_face_idx = defaultdict(list)
        for fi, r in enumerate(roots_per_face):
            root_to_face_idx[int(r)].append(int(fi))


        components = []
        for r, fidx_list in root_to_face_idx.items():
            fidx = np.array(fidx_list, dtype=np.int64)
            sub_faces_global = faces[fidx]  # (m_i, k)


            unique_vids = np.unique(sub_faces_global)
 
            g2l = {int(g): i for i, g in enumerate(unique_vids)}

            remapped = np.vectorize(g2l.__getitem__, otypes=[np.int64])(sub_faces_global)
            remapped = remapped.astype(np.int64)

            if remapped.shape[0] > 5:
                data = DynamicAttributes()
                data.vertices = vertices[unique_vids]            # (n_i, 3)
                data.faces = remapped                            # (m_i, k)
                components.append(data)

        return components if len(components) > 0 else None

    def split_mesh_into_connected_components(self, mesh_vertices, mesh_faces):


        if (mesh_faces.shape[0] == 0 or mesh_vertices.shape[0] == 0):
            print(mesh_faces.shape)
            return None
        

        adjacency_matrix = igl.adjacency_matrix(mesh_faces)
        num_components, components, _ = igl.connected_components(adjacency_matrix)
        component_faces = [[] for _ in range(num_components)]

        if components.shape[0]==0:
            return None
            

        for i in range(mesh_faces.shape[0]):
            face = mesh_faces[i]

            if components[face[0]] == components[face[1]] == components[face[2]]:
                component_faces[components[face[0]]].append(face)
                
        processed_components = []
        for faces in component_faces:
            if len(faces) > 0:
                faces = np.array(faces)
                unique_vertices = np.unique(faces)
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
                new_faces = np.vectorize(vertex_map.get)(faces)
                new_vertices = mesh_vertices[unique_vertices]

                data = DynamicAttributes()
                data.vertices = new_vertices
                data.faces = new_faces
                if len(new_faces) > 5:
                    processed_components.append(data)
                
        return processed_components
    
    def split_mesh_into_connected_components_trimesh(self, mesh_vertices, mesh_faces):

        if (mesh_faces.shape[0] == 0 or mesh_vertices.shape[0] == 0):
            print(f"faces shape={mesh_faces.shape}, vertices shape={mesh_vertices.shape}")
            return None
        
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        
        connected_components = mesh.split()
        print("debug---------------------------------------------")
        print(len(connected_components))
        
        processed_components = []
        for component_mesh in connected_components:
            if component_mesh.faces.shape[0] > 5: 
                data = DynamicAttributes()
                data.vertices = component_mesh.vertices
                data.faces = component_mesh.faces
                processed_components.append(data)
        
        return processed_components
    
    def __getitem__(self, idx: int) -> Dict:

        max_retries = 0
        retry_count = 0
        time_limit = 300  
        start_time = time.time()

        sample_info = self.uid_list[idx]

    
        while retry_count <= max_retries:
            try:

                if time.time() - start_time > time_limit:
                    logger.warning(f"Timeout reached for idx {idx} after {time.time() - start_time:.2f} seconds")
                    return self.__getitem__((idx + 1) % len(self))
                    
                raw_obj = trimesh.load(f"{self.path}/{sample_info}", force='mesh', file_type='obj', process=False)
                
                processed_mesh = self._preprocess_mesh(raw_obj)
                if processed_mesh.faces.shape[0] > 10:
                    break
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.warning(f"Failed to get mesh after {max_retries} retries for idx {idx}, trying next index")
                    return self.__getitem__((idx + 1) % len(self))
                logger.warning(f"Open failed for idx {idx}, attempt {retry_count}/{max_retries}: {str(e)}")
                continue
        
        data = {}
        if True:
            mesh = process_mesh(processed_mesh.vertices, processed_mesh.faces, quantization_bits=8, augment=True)

            verts, faces = mesh['vertices'], mesh['faces']
            
            lendatas = 0
            skip = 0
            processed_components = self.split_mesh_into_connected_components_union_set(verts, faces)
            if len(processed_components) >= 1:
                datas = selection_mesh_from_connected_components(verts, faces, processed_components, fix=False)
                if len(datas) > 0:
                    data['name'] = sample_info.split('.')[0]
                    for i in range(len(datas)):
                        if datas[i]['skip']:
                            skip += 1
                            continue
                        data['pc_local_'+str(i - skip + lendatas)] = datas[i]['pc_local']
                        data['token_list_'+str(i - skip + lendatas)] = datas[i]['token_list']
                        data['bd_token_length_'+str(i - skip + lendatas)] = datas[i]['bd_token_length']
                        data['skip_'+str(i - skip + lendatas)] = datas[i]['skip']
                        data['pc_global'] = datas[i]['pc_global']
                        data['center_'+str(i - skip + lendatas)] = datas[i]['center']
                        data['scale_'+str(i - skip + lendatas)] = datas[i]['scale']
                        data['cluster_'+str(i - skip + lendatas)] = datas[i]['cluster']
                data['len'] = len(datas) - skip + lendatas
            
            
        return data
