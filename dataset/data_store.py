import json
import base64
import pickle
import os
import numpy as np

from opengait.evaluation.metric import cuda_dist

def vector_to_b64(vec: np.ndarray) -> str:
    """将 numpy 向量序列化并做 base64 编码，以便存入 json."""
    raw_bytes = pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(raw_bytes).decode('utf-8')

def b64_to_vector(b64_str: str) -> np.ndarray:
    """将 base64 字符串解码并反序列化为 numpy 向量."""
    raw_bytes = base64.b64decode(b64_str.encode('utf-8'))
    return pickle.loads(raw_bytes)

def load_local_db(filepath: str) -> list:
    """
    从本地文件读取 JSON lines，每行一个记录。
    返回记录的列表，每个记录是类似:
    {
      "id": int,
      "name": str,
      "reid_feature": <base64>,
      "gait_feature": <base64 or None>
    }
    若文件不存在，返回空列表。
    """
    if not os.path.isfile(filepath):
        return []
    
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(data)
    return rows

def save_local_db(filepath: str, rows: list):
    """
    将记录列表写回本地文件，每条记录一行 JSON。
    文件将被覆盖写入。
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算余弦相似度
    """
    v2_f = v2.flatten()

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2_f)
    if v1_norm < 1e-12 or v2_norm < 1e-12:
        return 0.0
    return float(np.dot(v1, v2_f) / (v1_norm * v2_norm))

def normalize(vec: np.ndarray) -> np.ndarray:
    """
    简单的归一化
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm

def check_and_record(
    reid_feature: np.ndarray,
    gait_feature: np.ndarray,
    mode: str,
    local_file: str,
    track_id: str = None, 
    threshold: float = 0.65,
) -> int:
    """
    在本地文件中检索或注册新的 ReID/步态特征。

    参数:
    - reid_feature: 传入的目标的 ReID 特征向量 (np.ndarray).
    - gait_feature: 传入的目标的 步态特征向量 (np.ndarray).
    - mode: 'registration' 或 'recognition'.
        'registration': 允许在匹配到同一ID时更新; 未匹配则新建.
        'recognition': 只查询，不更新; 未匹配则返回 -1.
    - local_file: 存放数据的本地JSON行文件 (默认: person_reid_local.jsonl).
    - track_id: 目标的名字或跟踪ID (str), 在 'registration' 模式下存表时会用作 name.
    - threshold: 匹配阈值; 余弦相似度高于此值认为是同一实体.

    返回:
    - matched_id: 若成功匹配则返回对应的 "id"；否则:
        - 在 'registration' 模式下插入新记录后返回新ID
        - 在 'recognition' 模式下未匹配则返回 -1
    """
    # 读取本地所有记录
    rows = load_local_db(local_file)

    # 当前最大ID (自增用)
    current_ids = [r["id"] for r in rows] if rows else []
    max_id = max(current_ids) if current_ids else 0

    max_sim = -1.0
    matched_index = -1   # rows 里的索引
    matched_id = -1      # 自增ID
    matched_name = None

    # 遍历已存特征
    for i, row in enumerate(rows):
        num_id = row["id"]
        db_name = row["name"]          # str
        db_reid_b64 = row.get("reid_feature")
        db_gait_b64 = row.get("gait_feature")

        # 反序列化
        db_reid_vec = b64_to_vector(db_reid_b64) if db_reid_b64 else None
        db_gait_vec = b64_to_vector(db_gait_b64) if db_gait_b64 else None

        # 如果是 'registration' 模式，且 db_name == track_id，则直接更新
        if mode == 'registration':
            if db_name == track_id:
                matched_index = i
                matched_id = num_id
                matched_name = db_name
                alpha = 0.9
                old_reid = b64_to_vector(rows[matched_index]["reid_feature"])
                old_gait = b64_to_vector(rows[matched_index]["gait_feature"])
                updated_reid = alpha * old_reid + (1 - alpha) * reid_feature
                updated_reid = normalize(updated_reid)
                updated_gait = alpha * old_gait + (1 - alpha) * gait_feature
                updated_gait = normalize(updated_gait)
                rows[matched_index]["reid_feature"] = vector_to_b64(updated_reid)
                rows[matched_index]["gait_feature"] = vector_to_b64(updated_gait)
                save_local_db(local_file, rows)
                return matched_id

        # 否则在 'recognition' 模式时，计算相似度
        else:
            if db_reid_vec is not None and db_gait_vec is not None:
                sim_1 = compute_cosine_similarity(reid_feature, db_reid_vec)
                sim_2 = cuda_dist(gait_feature, db_gait_vec)
                sim = sim_1 * 0 + sim_2 * 1
                if sim > max_sim:
                    max_sim = sim
                    matched_index = i
                    matched_id = num_id
                    matched_name = db_name
        
    if mode == 'registration':
        new_id = max_id + 1
        new_reid = normalize(reid_feature)
        new_reid_b64 = vector_to_b64(new_reid)
        new_gait = normalize(gait_feature)
        new_gait_b64 = vector_to_b64(new_gait)
        new_row = {
            "id": new_id,
            "name": track_id, 
            "reid_feature": new_reid_b64,
            "gait_feature": new_gait_b64
        }
        rows.append(new_row)
        save_local_db(local_file, rows)
    else:
        if max_sim >= threshold:
            return matched_id  
        else:
            return -1          
    return new_id

def check_test(
    gait_feature: np.ndarray,
    local_file: str = "person_reid_local.jsonl",
    threshold: float = 0.65,
) -> int:
    """
    在本地文件中，用 gait_feature 去匹配已存条目（只查询，不更新）。
    如果相似度 >= threshold，则返回匹配到的 ID，否则返回 -1。
    """
    rows = load_local_db(local_file)

    max_sim = -1.0
    matched_id = -1

    for row in rows:
        num_id = row["id"]
        db_gait_b64 = row.get("gait_feature")
        if db_gait_b64 is None:
            continue
        db_gait_vec = b64_to_vector(db_gait_b64)

        sim = compute_cosine_similarity(gait_feature, db_gait_vec)
        if sim > max_sim:
            max_sim = sim
            matched_id = num_id

    if max_sim >= threshold:
        return matched_id
    else:
        return -1
    
def load_existing_id_mapping(mapping_file):
    """
    读取已有的id映射表 (如果存在)，返回一个字典和当前最大编号。
    文件格式：每行 => <顺序编号> <track_id>
    """
    id_map = {}   # 用于存储 track_id -> 顺序编号
    max_idx = 0   # 当前已经使用的最大编号
    
    if mapping_file.is_file():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 假设一行是: "1 101" or "2 叶顶强"
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                idx_str, raw_id = parts
                idx = int(idx_str)
                id_map[raw_id] = idx
                if idx > max_idx:
                    max_idx = idx
    return id_map, max_idx

def save_id_mapping(mapping_file, id_map):
    """
    将 id_map (track_id -> 顺序编号) 按照顺序编号排序，写回到TXT中。
    """
    # 这里我们想要按照 value（即顺序编号）从小到大排序
    sorted_items = sorted(id_map.items(), key=lambda x: x[1])
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        for raw_id, idx in sorted_items:
            f.write(f"{idx} {raw_id}\n")

def get_track_number_by_id(mapping_file, track_id, encoding='utf-8'):
    """
    从映射文件中查找给定的 track_id 对应的顺序编号。
    """
    if not os.path.isfile(mapping_file):
        # 如果文件不存在，则直接返回 None(或可以抛异常)
        return None
    
    with open(mapping_file, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设每行格式：“<顺序编号> <track_id>”
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            idx_str, raw_id = parts
            # 若匹配到，就返回对应编号
            if raw_id == track_id:
                return int(idx_str)
    
    # 文件里没有找到对应 track_id
    return None

def get_track_id_by_number(mapping_file, number, encoding='utf-8'):
    """
    从映射文件中查找给定的编号对应的 track_id。
    """

    if not os.path.isfile(mapping_file):
        return None  # 文件不存在，直接返回 None
    
    with open(mapping_file, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设每行格式："<顺序编号> <track_id>"
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            idx_str, raw_id = parts
            try:
                idx = int(idx_str)  # 转换为整数编号
                if idx == number:
                    return raw_id  # 找到匹配的编号，返回对应 track_id
            except ValueError:
                continue  # 遇到格式错误的行，跳过

    return None  # 如果没有找到匹配的编号，返回 None