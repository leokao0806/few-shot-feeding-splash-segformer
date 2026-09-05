import json
import requests
from neo4j import GraphDatabase
import random
import re
import math

# ==========================================
# 模組 1：LLM 意圖解析器 (只負責 NLP 轉換)
# ==========================================
import json
import requests

class LLMIntentParser:
    def __init__(self, model_name="llama3:8b", llm_url="http://localhost:11434/api/generate"):
        self.model = model_name
        self.llm_url = llm_url

    def parse(self, query_text, vocab):
        vocab_str = json.dumps(vocab, ensure_ascii=False, indent=2)
        prompt = f"""
        You are an advanced Aquaculture Semantic Router. Extract intent into STRICT JSON.
        Map words to exactly match [DATABASE VOCABULARY ENUMS]. Use null if absent.

        [DATABASE VOCABULARY ENUMS]
        {vocab_str}

        [EXTRACTION RULES]
        1. MICRO INTENSITY:
           - Map words like "strong, huge, large area, intense" -> "high"
           - Map words like "weak, small area, faint, slight" -> "low"
           - If unspecified, use "all"
        2. FIELD:
           - If the user specifies a location, field, Env or Domain (e.g., "Field A", "Env B", "Domain C"), map it to the EXACT string in the "field" vocabulary enum (e.g., "A", "B", "C").
           - If unspecified, use null.

        [EXAMPLES]
        User: "Find strong Cobia splash patterns"
        Output: {{"filters": {{"species": "cobia", "field": null}}, "comparative_logic": {{}}, "micro_intensity_mode": "high"}}

        User: "Find strong splash patterns in Field B"
        Output: {{"filters": {{"species": null, "field": "B"}}, "comparative_logic": {{}}, "micro_intensity_mode": "high"}}


        [JSON SCHEMA TO RETURN]
        {{
            "filters": {{
                "species": "Exact string or null", "field": "Exact string or null",
                "environment": "Exact string or null", "perspective": "Exact string or null",
                "water_color": "Exact string or null", "texture_type": "Exact string or null",
                "splash_shape": "Exact string or null", "surface_cover": "Exact string or null",
                "interference": "Exact string or null", "lighting": "Exact string or null",
                "container_edge": "Exact string or null"
            }},
            "comparative_logic": {{
                "operator": "STRONGER_THAN or WEAKER_THAN or SIMILAR_INTENSITY_TO or null",
                "base_species": "Exact string or null"
            }},
            "micro_intensity_mode": "high or medium or low or all"
        }}

        [User Query]: "{query_text}"
        Return ONLY valid JSON format:
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        try:
            response = requests.post(self.llm_url, json=payload, timeout=60).json()
            result = response.get('response', '{}')
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"[LLM 錯誤] {e}")
            return {"filters": {}, "comparative_logic": {}, "micro_intensity_mode": "all"}


# ==========================================
# 模組 2：Neo4j 圖譜檢索器 (只負責 Cypher)
# ==========================================
class GraphRetriever:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="66386638"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.schema_registry = {
            "species": {"label": "FishSpecies", "rel": "CONTAINS_SPECIES"},
            "field": {"label": "Field", "rel": "ORIGINATES_FROM"},
            "environment": {"label": "Environment", "rel": "HAS_ENVIRONMENT"},
            "perspective": {"label": "Perspective", "rel": "SHOT_FROM"},
            "water_color": {"label": "WaterColor", "rel": "HAS_COLOR"},
            "texture_type": {"label": "Texture", "rel": "HAS_TEXTURE"},
            "splash_shape": {"label": "SplashShape", "rel": "HAS_SHAPE"},
            "surface_cover": {"label": "SurfaceCover", "rel": "COVERED_BY"},
            "interference": {"label": "Interference", "rel": "INTERFERED_BY"},
            "lighting": {"label": "Lighting", "rel": "LIT_BY"},
            "container_edge": {"label": "ContainerEdge", "rel": "BOUNDED_BY"}
        }

    def fetch_vocabulary(self):
        vocab = {}
        with self.driver.session() as session:
            for key, config in self.schema_registry.items():
                res = session.run(f"MATCH (n:{config['label']}) RETURN n.name AS name")
                vocab[key] = [record['name'] for record in res if record['name']]
        return vocab

    def search_clusters(self, intent_json):
        match_clauses = ["MATCH (c:Cluster)"]
        where_clauses = []
        parameters = {}

        filters = intent_json.get('filters', {})
        for key, val in filters.items():
            if val and key in self.schema_registry:
                config = self.schema_registry[key]
                if key == "environment":
                    match_clauses.append("MATCH (c)-[:ORIGINATES_FROM]->(:Field)-[:HAS_ENVIRONMENT]->(e:Environment)")
                    where_clauses.append("e.name = $environment")
                    parameters['environment'] = val
                else:
                    var_name = f"n_{key}"
                    match_clauses.append(f"MATCH (c)-[:{config['rel']}]->({var_name}:{config['label']})")
                    where_clauses.append(f"{var_name}.name = ${key}")
                    parameters[key] = val

        # [簡化顯示，保留了跨物種邏輯的擴充空間]

        if not where_clauses:
            return []

        cypher_query = "\n".join(match_clauses) + "\nWHERE " + " AND ".join(where_clauses)
        # 使用 coalesce 避免 label 屬性遺失問題
        cypher_query += "\nRETURN DISTINCT coalesce(c.label, c.name, c.id) AS cluster"

        with self.driver.session() as session:
            results = session.run(cypher_query, parameters)
            return [record['cluster'] for record in results if record['cluster']]


# ==========================================
# 模組 3：樣本袋打包員 (負責處理原始影像實體檔案)
# =========================================
class PromptBagBuilder:
    def __init__(self, mapping_json_path, knowledge_json_path):
        self.mapping_path = mapping_json_path

        # 新增：圖譜代號與真實場域名稱的映射表
        self.field_code_mapping = {
            "A": "barrel",
            "B": "sea",
            "C": "LNG",
            "D": "seabass_hmh",
            "E": "noon_jsj"
        }

        # 載入映射清單
        try:
            with open(mapping_json_path, 'r', encoding='utf-8') as f:
                self.pt_mappings = json.load(f)
            with open(knowledge_json_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        except Exception as e:
            print(f"❌ [讀取錯誤]: {e}")
            self.pt_mappings = []
            self.knowledge_base = []

    def _parse_distribution(self, dist_str):
        """
        解析 "E(60.5%), D(39.5%)" 格式為 {'E': 0.605, 'D': 0.395}
        """
        dist_dict = {}
        matches = re.findall(r"([A-E])\(([\d.]+)%\)", dist_str)
        for field, percent in matches:
            dist_dict[field] = float(percent) / 100.0
        return dist_dict

    def build_bag(self, target_clusters, intensity_mode="all", total_n=20, target_field=None):
        """
        根據知識圖譜比例進行分層抽樣，或強制指定單一場域
        """
        if not target_clusters: return []

        # 1. 取得目標 Cluster 的知識元數據
        cluster_id_int = int(re.findall(r'\d+', str(target_clusters[0]))[0])
        knowledge_item = next((item for item in self.knowledge_base if item['cluster_id'] == cluster_id_int), None)

        if not knowledge_item:
            return []

        # 2. 解析比例 或 套用強制指定場域，並將「代號」轉換為「真實名稱」
        dist_map = {}
        if target_field:
            target_code = target_field.upper()
            # 查表轉換，若查無則保持原樣（作為防呆）
            real_name = self.field_code_mapping.get(target_code, target_code)
            dist_map[real_name] = 1.0
            print(f"🎯 [強制指定場域]: 代號 {target_code} -> 映射為 {real_name} (權重 100%)")
        else:
            dist_str = knowledge_item['graph_rag']['source_distribution']
            parsed_dist = self._parse_distribution(dist_str)

            # 將圖譜比例中的代號也轉換為真實名稱
            for code, weight in parsed_dist.items():
                real_name = self.field_code_mapping.get(code.upper(), code)
                dist_map[real_name] = weight
            print(f"📊 [比例預算轉換]: 原始={parsed_dist} -> 實際={dist_map}")

        # 3. 預篩選與強度過濾
        all_candidates = [pt for pt in self.pt_mappings if pt['cluster_id'] == cluster_id_int]
        all_candidates.sort(key=lambda x: x['intensity'])
        if intensity_mode == "high":
            all_candidates = all_candidates[int(len(all_candidates) * 0.75):]
        elif intensity_mode == "low":
            all_candidates = all_candidates[:max(1, int(len(all_candidates) * 0.25))]

        # 4. 按場域進行抽樣
        final_bag = []
        for real_field, weight in dist_map.items():
            # 這裡使用轉換後的真實名稱 (如 'sea') 去跟 JSON 裡的欄位比對
            field_candidates = [pt for pt in all_candidates if pt['field'].upper() == real_field.upper()]
            sample_count = max(1, math.floor(total_n * weight))
            selected = random.sample(field_candidates, min(len(field_candidates), sample_count))
            final_bag.extend(selected)

        # 5. 最終微調：數量不足時補齊
        if len(final_bag) < total_n:
            if target_field:
                # 確保從轉換後的真實場域補齊
                real_target = self.field_code_mapping.get(target_field.upper(), target_field).upper()
                remaining = [pt for pt in all_candidates if pt not in final_bag and pt['field'].upper() == real_target]
            else:
                remaining = [pt for pt in all_candidates if pt not in final_bag]

            needed = total_n - len(final_bag)
            final_bag.extend(random.sample(remaining, min(len(remaining), needed)))

        random.shuffle(final_bag)
        return [item['image_absolute_path'] for item in final_bag]



# ==========================================
# 統籌指揮中心：AquacultureGraphRAG (提供給外部呼叫的唯一接口)
# ==========================================
class AquacultureGraphRAG:
    def __init__(self, pt_mapping_path, knowledge_json_path, neo4j_pw="password"):
        print("🔧 系統初始化中...")
        self.parser = LLMIntentParser()
        self.retriever = GraphRetriever(password=neo4j_pw)

        # 實例化改進型樣本袋打包員，注入映射表與圖譜知識檔
        self.bag_builder = PromptBagBuilder(
            mapping_json_path=pt_mapping_path,
            knowledge_json_path=knowledge_json_path
        )

        print("🌐 正在同步 Neo4j 圖譜詞彙...")
        self.vocab = self.retriever.fetch_vocabulary()
        print("✅ 系統準備就緒！")

    def query(self, text_prompt, total_n=20, target_field=None):
        """
        執行檢索增強生成查詢，建構具備場域代表性的提示袋 (Prompt Bag)。
        """
        print(f"\n{'=' * 50}\n🗣️ 專家提問: {text_prompt}")

        # Step 1: LLM 意圖解析 (Text-to-Intent)
        intent = self.parser.parse(text_prompt, self.vocab)
        print(
            f"🧠 [意圖解析]: {json.dumps(intent['filters'], ensure_ascii=False)} | 強度: {intent.get('micro_intensity_mode')}"
        )

        # 必須新增此段落：自動從 LLM 意圖中擷取 target_field
        if not target_field and intent.get('filters', {}).get('field'):
            target_field = intent['filters']['field']

        # Step 2: 圖譜群集檢索 (Graph Retrieval)
        clusters = self.retriever.search_clusters(intent)
        print(f"🎯 [圖譜命中]: 找到群集 {clusters}")

        # Step 3: 分層比例打包樣本袋 (Bag-based Prompt Construction)
        # 必須修改此段落：將 target_field 變數傳入 build_bag
        img_bag = self.bag_builder.build_bag(
            target_clusters=clusters,
            intensity_mode=intent.get('micro_intensity_mode', 'all'),
            total_n=total_n,
            target_field=target_field
        )
        print(f"📦 [正樣本袋]: 成功提取 {len(img_bag)} 個原始影像檔。")

        return {
            "image_list": img_bag,
            "intent": intent
        }