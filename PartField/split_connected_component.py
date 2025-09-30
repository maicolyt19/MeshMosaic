import os

import numpy as np
from plyfile import PlyData
from collections import defaultdict, deque

input_dir = '/home/qiujie/PartField/exp_results/clustering/objaverse/ply'
save_dir = '/home/qiujie/PartField/exp_results/clustering/objaverse/split_connected_component'
os.makedirs(save_dir, exist_ok=True)

for f in sorted(os.listdir(input_dir)):
    if not f.endswith('.ply'):
        continue
    input_path = os.path.join(input_dir, f)

    mesh_name = os.path.splitext(f)[0]
    save_path = os.path.join(save_dir, mesh_name + '.obj')


    INPUT_PLY = input_path
    OUTPUT_OBJ = save_path

    ply = PlyData.read(INPUT_PLY)

    v = ply['vertex'].data
    verts = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)

    face_el = ply['face']
    names = face_el.data.dtype.names
    if 'vertex_indices' in names:
        faces_raw = face_el.data['vertex_indices']
    elif 'vertex_index' in names:
        faces_raw = face_el.data['vertex_index']
    else:
        raise ValueError("not find the vertex_indices / vertex_index")

    faces = [np.array(f, dtype=np.int64) for f in faces_raw]
    F = len(faces)


    def pick_color_fields(fields):
        cands = [
            ('red','green','blue','alpha'),
            ('red','green','blue'),
            ('r','g','b','a'),
            ('r','g','b'),
            ('diffuse_red','diffuse_green','diffuse_blue','diffuse_alpha'),
            ('diffuse_red','diffuse_green','diffuse_blue'),
        ]
        for cs in cands:
            if all(c in fields for c in cs[:3]):
                return cs
        return None

    cs = pick_color_fields(names)
    if cs is None:
        raise ValueError("not find red/green/blue")


    fc = np.column_stack([face_el.data[cs[0]], face_el.data[cs[1]], face_el.data[cs[2]]])
    if np.issubdtype(fc.dtype, np.floating):
        face_colors = np.rint(np.clip(fc, 0, 1) * 255.0).astype(np.int32)
    else:
        face_colors = fc.astype(np.int32)


    edge2faces = defaultdict(list)
    for fi, face in enumerate(faces):
        m = len(face)
        for i in range(m):
            a = int(face[i]); b = int(face[(i+1) % m])
            if a == b: 
                continue
            e = (a, b)
            ek = (min(e), max(e)) 
            edge2faces[ek].append(fi)

    adj = [[] for _ in range(F)]
    for flist in edge2faces.values():

        for i in range(len(flist)):
            for j in range(i+1, len(flist)):
                u, v_ = flist[i], flist[j]
                adj[u].append(v_)
                adj[v_].append(u)


    labels = -np.ones(F, dtype=np.int64)
    label = 0


    color_tuples = [tuple(c.tolist()) for c in face_colors]

    for f_id in range(F):
        if labels[f_id] != -1:
            continue
        base_c = color_tuples[f_id]

        q = deque([f_id])
        labels[f_id] = label
        while q:
            u = q.popleft()
            for nb in adj[u]:
                if labels[nb] == -1 and color_tuples[nb] == base_c:
                    labels[nb] = label
                    q.append(nb)
        label += 1

    assert np.all(labels >= 0), "No marked faces"
    print(f"{label} connected branches were detected.")


    with open(OUTPUT_OBJ, "w") as f:
        f.write("# segmented by face-color connected components\n")
        f.write(f"# vertices={len(verts)} faces={F} components={label}\n")
        global_v_offset = 0


        for lb in range(label):
            face_ids = np.where(labels == lb)[0]
            if face_ids.size == 0:
                continue

            c = color_tuples[int(face_ids[0])]


            used_verts = np.unique(np.concatenate([faces[i] for i in face_ids]).ravel())

            remap = {int(old): int(i + 1 + global_v_offset) for i, old in enumerate(used_verts)}

            f.write(f"o part_{lb}_c{c[0]}_{c[1]}_{c[2]}\n")
            f.write("s off\n") 

            for vid in used_verts:
                x, y, z = verts[int(vid)]
                f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")


            for fi in face_ids:
                loop = faces[int(fi)]

                mapped = [remap[int(v)] for v in loop]

                f.write("f " + " ".join(str(x) for x in mapped) + "\n")

            global_v_offset += len(used_verts)

    print(f"Saving：{OUTPUT_OBJ}")


