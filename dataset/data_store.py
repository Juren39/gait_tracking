import sqlite3

def init_database(db_path="tracking.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建行人信息表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pedestrian_info (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
        reid_feature BLOB
    )
    """)
    
    # 创建跟踪记录表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tracking_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_number INTEGER,
        person_id INTEGER,
        bbox_x REAL,
        bbox_y REAL,
        bbox_width REAL,
        bbox_height REAL,
        confidence REAL,
        FOREIGN KEY (person_id) REFERENCES pedestrian_info(person_id)
    )
    """)
    
    conn.commit()
    return conn

def extract_reid_feature(cropped_person):
    """
    提取裁剪后行人的 ReID 特征。
    """
    # 假设您有一个加载好的 ReID 模型
    reid_model = load_reid_model()
    feature = reid_model(cropped_person)  # 模型前向推理
    return feature.cpu().numpy()