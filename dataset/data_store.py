import hashlib
import pymysql
import json
import numpy as np
import pickle
import yaml

def serialize_vector(vec: np.ndarray) -> bytes:
    """
    将 numpy 向量序列化为 bytes，便于存储在 BLOB 中。
    也可改用其他方式 (JSON等)，但 pickle 更简单。
    """
    return pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_vector(data: bytes) -> np.ndarray:
    """
    反序列化 bytes 为 numpy 向量
    """
    return pickle.loads(data)

def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算余弦相似度
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-12 or v2_norm < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (v1_norm * v2_norm))

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
    mode: str,
    db_config: dict,
    track_id: str = None, 
    threshold: float = 0.65,
) -> int:
    """
    在数据库中检索或注册新的 ReID 特征。

    参数:
    - reid_feature: 传入的目标的特征向量 (np.ndarray).
    - db_config: 数据库连接参数 (dict).
    - threshold: 匹配阈值; 余弦相似度高于此值认为是同一实体.
    - mode: 'registration' 或 'recognition'.
        'registration': 允许在匹配到同一ID时更新数据库特征；未匹配则新建.
        'recognition': 只查询，不更新已有特征; 未匹配时返回 -1.
    
    返回:
    - matched_id: 如果成功匹配则返回对应的数据库 ID，否则:
        - 在 'registration' 模式下插入新记录后返回新ID
        - 在 'recognition' 模式下未匹配则返回 -1
    """
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # 先把数据库里所有记录都取出来 (仅示例，实际可用更高效的向量检索)
            sql = "SELECT id, name, feature, feature_dim FROM person_reid"
            cursor.execute(sql)
            rows = cursor.fetchall()

        max_sim = -1.0
        matched_id = -1
        matched_num_id = -1
        matched_feature = None  # 记录与其匹配的旧特征，用于做更新

        # 遍历已存特征
        for row in rows:
            num_id = row[0]
            db_id = row[1]
            db_feature_bytes = row[2]
            db_feature_dim = row[3]
            db_feature = deserialize_vector(db_feature_bytes)
            if len(db_feature) != db_feature_dim:
                # 理论上不应发生，如有就需要数据清洗
                continue
            if mode == 'registration':
                if db_id == track_id:
                    matched_id = db_id
                    matched_feature = db_feature
                    alpha = 0.9  # 这个值可以根据需要调整
                    updated_feature = alpha * matched_feature + (1 - alpha) * reid_feature
                    updated_feature = normalize(updated_feature)
                    updated_bytes = serialize_vector(updated_feature)
                    with connection.cursor() as cursor:
                        update_sql = "UPDATE person_reid SET feature=%s, feature_dim=%s WHERE name=%s"
                        cursor.execute(update_sql, (updated_bytes, len(updated_feature), track_id))
                    connection.commit()
                    return matched_id
            else:
                sim = compute_cosine_similarity(reid_feature, db_feature)
                if sim > max_sim:
                    max_sim = sim
                    matched_num_id = num_id
                    matched_feature = db_feature

        # 没有找到目标
        if mode == 'registration':
            # 注册模式: 新建一条记录
            new_feature = normalize(reid_feature)
            new_feature_bytes = serialize_vector(new_feature)
            with connection.cursor() as cursor:
                insert_sql = "INSERT INTO person_reid (name, feature, feature_dim) VALUES (%s, %s, %s)"
                cursor.execute(insert_sql, (track_id, new_feature_bytes, len(new_feature)))
            connection.commit()
            return track_id
        else:
            # 识别模式: 不更新, 返回-1
            if max_sim >= threshold:
                return matched_num_id
            else:
                return -1
    finally:
        connection.close()

def save_id_name_mapping(db_config, output_file="id_name_mapping.txt"):
    """
    连接数据库，查询 `person_reid` 表中的 `id` 和 `name`，并保存到本地 txt 文件中。

    参数:
    - db_config: 数据库连接配置字典
    - output_file: 保存的 txt 文件名 (默认: id_name_mapping.txt)
    """
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            query = "SELECT id, name FROM person_reid"
            cursor.execute(query)
            rows = cursor.fetchall()

        with open(output_file, "w", encoding="utf-8") as f:
            for row in rows:
                person_id, name = row
                name = name if name else "Unknown"  # 处理 name 可能为空的情况
                f.write(f"{person_id}: {name}\n")

        print(f"✅ ID-Name 映射关系已保存至: {output_file}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

    finally:
        connection.close()