import pymysql

# 数据库配置，请根据实际情况修改
db_config = {
    'host': '10.16.9.79',  # 例如 '192.168.1.100' 或 'your.server.com'
    'port': 3306,                # MySQL 默认端口
    'user': 'root',         # 数据库用户名
    'password': 'qscfthm123', # 数据库密码
    'database': 'track_database',
    'charset': 'utf8mb4'
}

def create_table(db_config):
    """
    （可选）初始化数据库表，如果不存在则创建。
    可以在程序启动时执行一次。
    """
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            create_sql = """
                CREATE TABLE IF NOT EXISTS person_reid (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) DEFAULT NULL,
                    reid_feature LONGBLOB NOT NULL,
                    gait_feature LONGBLOB NOT NULL,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_sql)
        connection.commit()
        print("✅ 表 `person_reid` 创建成功！")   
    except Exception as e:
        print("❌ 创建表时出错:", e)
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_table(db_config)