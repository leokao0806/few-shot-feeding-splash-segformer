import networkx as nx
from neo4j import GraphDatabase
import os

# ================= 設定區域 =================
# 1. GML 檔案路徑
GML_PATH = "vlm_knowledge_construction_v3/knowledge_graph.gml"

# 2. Neo4j 連線設定 (請確認密碼與剛剛設定的一致)
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "66386638")  # 預設帳號是 neo4j


def clear_database(tx):
    """清除資料庫中所有現有資料 (重置用)"""
    tx.run("MATCH (n) DETACH DELETE n")


def create_node(tx, node_id, attributes):
    """
    建立節點
    node_id: NetworkX 中的節點 ID
    attributes: 節點屬性字典 (包含 type, label 等)
    """
    # 取得節點類型 (Label)，預設為 'Thing'
    label = attributes.get('type', 'Thing')

    # 構建 Cypher 語句
    # 我們使用 MERGE 避免重複，並動態設定屬性
    query = f"MERGE (n:`{label}` {{id: $id}}) SET n += $props"

    # 移除 type 屬性，因為它已經變成 Label 了，不需要重複存
    props = {k: v for k, v in attributes.items() if k != 'type'}

    tx.run(query, id=node_id, props=props)


def create_relationship(tx, source, target, relation_type):
    """建立關係"""
    # 這裡假設 source 和 target 的節點都已經透過 create_node 建立好了
    # 我們透過 id 來查找節點
    query = f"""
    MATCH (a {{id: $source}})
    MATCH (b {{id: $target}})
    MERGE (a)-[:`{relation_type}`]->(b)
    """
    tx.run(query, source=source, target=target)


def main():
    # 1. 讀取 GML
    if not os.path.exists(GML_PATH):
        print(f"錯誤：找不到檔案 {GML_PATH}")
        return

    print("正在讀取 GML 檔案...")
    G = nx.read_gml(GML_PATH)
    print(f"讀取成功！節點數: {len(G.nodes)}, 邊數: {len(G.edges)}")

    # 2. 連接 Neo4j
    print("正在連接 Neo4j...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("連線成功！")
    except Exception as e:
        print(f"連線失敗，請檢查 Neo4j 是否已啟動？密碼是否正確？\n錯誤訊息: {e}")
        return

    with driver.session() as session:
        # A. 清空舊資料
        print("正在清空舊資料庫...")
        session.execute_write(clear_database)

        # B. 匯入節點
        print("正在匯入節點...")
        count = 0
        for node_id, attrs in G.nodes(data=True):
            session.execute_write(create_node, node_id, attrs)
            count += 1
            if count % 100 == 0: print(f"  已處理 {count} 個節點...")

        # C. 匯入關係
        print("正在匯入關係...")
        count = 0
        for source, target, attrs in G.edges(data=True):
            # 從 GML 屬性中取得 relation 名稱，若無則預設 RELATED
            rel_type = attrs.get('relation', 'RELATED')
            session.execute_write(create_relationship, source, target, rel_type)
            count += 1
            if count % 100 == 0: print(f"  已處理 {count} 條邊...")

    driver.close()
    print("\n匯入完成！現在請回到 Neo4j Desktop 查看結果。")


if __name__ == "__main__":
    main()