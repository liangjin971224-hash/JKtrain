// ============================================================
// 健康顾问培训系统 · 知识图谱建库脚本
// Database: jiankangshujuku
// 数据来源: 06_正式内容 / 05_待审核样例（样例2 产品资料）
// 编制日期: 2026-10-21
// ============================================================
//
// 节点类型（8 类）
//   Product          产品
//   Ingredient       成分
//   Efficacy         批准保健功能（只能存目录原文）
//   Usage            用法用量
//   TargetGroup      适宜人群
//   Contraindication 不适宜人群
//   Scenario         使用场景
//   Need             客户需求
//
// 关系类型（10 类）
//   HAS_INGREDIENT / CLAIMS / DOSED_AS / SUITABLE_FOR
//   CONTRAINDICATED_FOR / USED_IN / ADDRESSES
//   PAIRS_WITH / INCOMPATIBLE_WITH / SIMILAR_TO
//
// 三条纪律
//   G1  Efficacy 只能存批准功能原文，不可自由填写
//   G2  CONTRAINDICATED_FOR 必须有 reason
//   G3  SIMILAR_TO 只存客观差异，不存优劣
// ============================================================


// ============================================================
// 第 0 步：清理（仅首次执行时使用）
// ============================================================
// MATCH (n) DETACH DELETE n;


// ============================================================
// 第 1 步：约束与索引
// ============================================================

CREATE CONSTRAINT product_id_unique IF NOT EXISTS
FOR (p:Product) REQUIRE p.product_id IS UNIQUE;

CREATE CONSTRAINT ingredient_id_unique IF NOT EXISTS
FOR (i:Ingredient) REQUIRE i.ingredient_id IS UNIQUE;

CREATE CONSTRAINT efficacy_id_unique IF NOT EXISTS
FOR (e:Efficacy) REQUIRE e.efficacy_id IS UNIQUE;

CREATE CONSTRAINT usage_id_unique IF NOT EXISTS
FOR (u:Usage) REQUIRE u.usage_id IS UNIQUE;

CREATE CONSTRAINT target_id_unique IF NOT EXISTS
FOR (t:TargetGroup) REQUIRE t.target_id IS UNIQUE;

CREATE CONSTRAINT contra_id_unique IF NOT EXISTS
FOR (c:Contraindication) REQUIRE c.contra_id IS UNIQUE;

CREATE CONSTRAINT scenario_id_unique IF NOT EXISTS
FOR (s:Scenario) REQUIRE s.scenario_id IS UNIQUE;

CREATE CONSTRAINT need_id_unique IF NOT EXISTS
FOR (n:Need) REQUIRE n.need_id IS UNIQUE;

CREATE INDEX product_name_idx IF NOT EXISTS FOR (p:Product) ON (p.name);
CREATE INDEX product_category_idx IF NOT EXISTS FOR (p:Product) ON (p.category);
CREATE INDEX product_approval_idx IF NOT EXISTS FOR (p:Product) ON (p.approval_no);


// ============================================================
// 第 2 步：产品节点（20 个）
// ============================================================
// category: 保健食品 / 普通食品
// data_origin: synthetic（虚构产品，防误用于真实业务）

MERGE (p:Product {product_id: 'P001'})
SET p.name = '康源牌鱼油软胶囊',
    p.category = '保健食品',
    p.approval_no = '国食健注G20250012',
    p.dosage_form = '软胶囊',
    p.spec = '1.0g/粒×60粒',
    p.retail_price = 298,
    p.status = '在售',
    p.primary_scenario = '血脂管理',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P002'})
SET p.name = '康源牌辅酶Q10软胶囊',
    p.category = '保健食品',
    p.approval_no = '国食健注G20250013',
    p.dosage_form = '软胶囊',
    p.spec = '0.3g/粒×60粒',
    p.retail_price = 368,
    p.status = '在售',
    p.primary_scenario = '抗氧化/心脏关注',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P003'})
SET p.name = '康源牌褪黑素片',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000123',
    p.dosage_form = '片剂',
    p.spec = '0.3g/片×60片',
    p.retail_price = 128,
    p.status = '在售',
    p.primary_scenario = '睡眠',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P004'})
SET p.name = '康源牌乳清蛋白粉',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000124',
    p.dosage_form = '粉剂',
    p.spec = '400g/罐',
    p.retail_price = 428,
    p.status = '在售',
    p.primary_scenario = '蛋白补充',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P005'})
SET p.name = '康源牌大豆分离蛋白粉',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000125',
    p.dosage_form = '粉剂',
    p.spec = '400g/罐',
    p.retail_price = 268,
    p.status = '在售',
    p.primary_scenario = '蛋白补充',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P006'})
SET p.name = '康源牌破壁灵芝孢子粉',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000126',
    p.dosage_form = '粉剂',
    p.spec = '2g/袋×30袋',
    p.retail_price = 598,
    p.status = '在售',
    p.primary_scenario = '免疫',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P007'})
SET p.name = '康源牌螺旋藻片',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000127',
    p.dosage_form = '片剂',
    p.spec = '0.5g/片×120片',
    p.retail_price = 198,
    p.status = '在售',
    p.primary_scenario = '免疫',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P008'})
SET p.name = '康源牌西洋参含片',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000128',
    p.dosage_form = '片剂',
    p.spec = '0.5g/片×60片',
    p.retail_price = 328,
    p.status = '在售',
    p.primary_scenario = '抗疲劳',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P009'})
SET p.name = '康源牌人参胶囊',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000129',
    p.dosage_form = '硬胶囊',
    p.spec = '0.4g/粒×60粒',
    p.retail_price = 458,
    p.status = '在售',
    p.primary_scenario = '抗疲劳',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P010'})
SET p.name = '康源牌灵芝孢子油软胶囊',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000130',
    p.dosage_form = '软胶囊',
    p.spec = '0.5g/粒×60粒',
    p.retail_price = 698,
    p.status = '在售',
    p.primary_scenario = '免疫',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P011'})
SET p.name = '康源牌钙维生素D软胶囊',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000131',
    p.dosage_form = '软胶囊',
    p.spec = '1.0g/粒×100粒',
    p.retail_price = 168,
    p.status = '在售',
    p.primary_scenario = '骨骼',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P012'})
SET p.name = '康源牌多种维生素矿物质片',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000132',
    p.dosage_form = '片剂',
    p.spec = '1.0g/片×60片',
    p.retail_price = 138,
    p.status = '在售',
    p.primary_scenario = '基础营养',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P013'})
SET p.name = '康源牌益生菌粉',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000133',
    p.dosage_form = '粉剂',
    p.spec = '2g/袋×30袋',
    p.retail_price = 228,
    p.status = '在售',
    p.primary_scenario = '肠道',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P014'})
SET p.name = '康源牌膳食纤维粉',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000134',
    p.dosage_form = '粉剂',
    p.spec = '5g/袋×30袋',
    p.retail_price = 158,
    p.status = '在售',
    p.primary_scenario = '肠道',
    p.data_origin = 'synthetic';

MERGE (p:Product {product_id: 'P015'})
SET p.name = '康源牌维生素C咀嚼片',
    p.category = '保健食品',
    p.approval_no = '食健备G202544000135',
    p.dosage_form = '片剂',
    p.spec = '0.8g/片×60片',
    p.retail_price = 98,
    p.status = '在售',
    p.primary_scenario = '基础营养',
    p.data_origin = 'synthetic';

// ---- 普通食品（5 个）----
// 重要：这 5 个产品没有保健功能声称，不能有任何 CLAIMS 关系

MERGE (p:Product {product_id: 'P016'})
SET p.name = '康源牌复合蛋白固体饮料',
    p.category = '普通食品',
    p.approval_no = 'SC11344010100123',
    p.dosage_form = '粉剂',
    p.spec = '360g/罐',
    p.retail_price = 198,
    p.status = '在售',
    p.primary_scenario = '蛋白补充',
    p.data_origin = 'synthetic',
    p.compliance_note = '普通食品，不得作任何保健功能声称';

MERGE (p:Product {product_id: 'P017'})
SET p.name = '康源牌胶原蛋白肽饮品',
    p.category = '普通食品',
    p.approval_no = 'SC11344010100124',
    p.dosage_form = '口服液',
    p.spec = '50ml/瓶×10瓶',
    p.retail_price = 268,
    p.status = '在售',
    p.primary_scenario = '美容',
    p.data_origin = 'synthetic',
    p.compliance_note = '普通食品，不得作任何保健功能声称';

MERGE (p:Product {product_id: 'P018'})
SET p.name = '康源牌枸杞原浆',
    p.category = '普通食品',
    p.approval_no = 'SC11344010100125',
    p.dosage_form = '口服液',
    p.spec = '30ml/袋×30袋',
    p.retail_price = 128,
    p.status = '在售',
    p.primary_scenario = '日常养生',
    p.data_origin = 'synthetic',
    p.compliance_note = '普通食品，不得作任何保健功能声称';

MERGE (p:Product {product_id: 'P019'})
SET p.name = '康源牌黑芝麻丸',
    p.category = '普通食品',
    p.approval_no = 'SC11344010100126',
    p.dosage_form = '丸剂',
    p.spec = '10g/丸×20丸',
    p.retail_price = 88,
    p.status = '在售',
    p.primary_scenario = '日常养生',
    p.data_origin = 'synthetic',
    p.compliance_note = '普通食品，不得作任何保健功能声称';

MERGE (p:Product {product_id: 'P020'})
SET p.name = '康源牌代餐奶昔',
    p.category = '普通食品',
    p.approval_no = 'SC11344010100127',
    p.dosage_form = '粉剂',
    p.spec = '500g/罐',
    p.retail_price = 218,
    p.status = '在售',
    p.primary_scenario = '体重管理',
    p.data_origin = 'synthetic',
    p.compliance_note = '普通食品，不得作任何保健功能声称';


// ============================================================
// 第 3 步：功效节点（只存 24 项目录原文 + 营养素补充）
// ============================================================
// G1 纪律：name 必须是《允许保健食品声称的保健功能目录（2023年版）》原文
// catalog_no 对应目录序号，1–24

MERGE (e:Efficacy {efficacy_id: 'EFF_01'})
SET e.name = '有助于增强免疫力', e.catalog_no = 1, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_02'})
SET e.name = '有助于抗氧化', e.catalog_no = 2, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_03'})
SET e.name = '辅助改善记忆', e.catalog_no = 3, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_06'})
SET e.name = '有助于改善睡眠', e.catalog_no = 6, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_07'})
SET e.name = '缓解体力疲劳', e.catalog_no = 7, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_09'})
SET e.name = '有助于控制体内脂肪', e.catalog_no = 9, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_10'})
SET e.name = '有助于改善骨密度', e.catalog_no = 10, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_15'})
SET e.name = '有助于调节肠道菌群', e.catalog_no = 15, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_17'})
SET e.name = '有助于润肠通便', e.catalog_no = 17, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_19'})
SET e.name = '有助于维持血脂（胆固醇/甘油三酯）健康水平',
    e.catalog_no = 19, e.catalog_version = '2023', e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_20'})
SET e.name = '有助于维持血糖健康水平', e.catalog_no = 20, e.catalog_version = '2023',
    e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_21'})
SET e.name = '有助于维持血压健康水平', e.catalog_no = 21, e.catalog_version = '2023',
    e.data_origin = 'real';

// ---- 营养素补充剂（不属于 24 项功能，单独标记）----
MERGE (e:Efficacy {efficacy_id: 'EFF_NUT_CA'})
SET e.name = '补充钙、维生素D', e.type = '营养素补充剂', e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_NUT_MV'})
SET e.name = '补充多种维生素和矿物质', e.type = '营养素补充剂', e.data_origin = 'real';

MERGE (e:Efficacy {efficacy_id: 'EFF_NUT_VC'})
SET e.name = '补充维生素C', e.type = '营养素补充剂', e.data_origin = 'real';


// ============================================================
// 第 4 步：不适宜人群节点（18 个）
// ============================================================
// G2 纪律：必须有 reason 属性

MERGE (c:Contraindication {contra_id: 'CI_CHILD'})
SET c.name = '少年儿童', c.severity = 'high',
    c.reason = '多数保健食品原料目录明确列为不适宜人群';

MERGE (c:Contraindication {contra_id: 'CI_INFANT'})
SET c.name = '婴幼儿', c.severity = 'high',
    c.reason = '婴幼儿器官发育未成熟，不宜食用';

MERGE (c:Contraindication {contra_id: 'CI_PREG'})
SET c.name = '孕妇', c.severity = 'high',
    c.reason = '孕期营养需专业评估，多数产品不适宜';

MERGE (c:Contraindication {contra_id: 'CI_LACT'})
SET c.name = '乳母', c.severity = 'high',
    c.reason = '哺乳期营养需专业评估，多数产品不适宜';

MERGE (c:Contraindication {contra_id: 'CI_BLEED'})
SET c.name = '出血倾向者和出血性疾病患者', c.severity = 'critical',
    c.reason = '鱼油可能影响凝血功能，与抗凝药同服增加出血风险';

MERGE (c:Contraindication {contra_id: 'CI_LIVER'})
SET c.name = '肝功能不全者', c.severity = 'high',
    c.reason = '肝功能不全时代谢能力下降';

MERGE (c:Contraindication {contra_id: 'CI_ALLERGY'})
SET c.name = '过敏体质人群', c.severity = 'high',
    c.reason = '成分可能引发过敏反应';

MERGE (c:Contraindication {contra_id: 'CI_SEAFOOD'})
SET c.name = '海产品过敏者', c.severity = 'critical',
    c.reason = '鱼油来源于海产品，注意事项明确不推荐食用';

MERGE (c:Contraindication {contra_id: 'CI_DRIVER'})
SET c.name = '驾驶及机械作业者（操作前/中）', c.severity = 'critical',
    c.reason = '褪黑素可能引起嗜睡，操作前或操作中禁用';

MERGE (c:Contraindication {contra_id: 'CI_AUTOIMMUNE'})
SET c.name = '自身免疫症患者（类风湿等）', c.severity = 'high',
    c.reason = '褪黑素慎用，可能影响免疫调节';

MERGE (c:Contraindication {contra_id: 'CI_HYPER'})
SET c.name = '甲亢患者', c.severity = 'high',
    c.reason = '褪黑素慎用';

MERGE (c:Contraindication {contra_id: 'CI_RENAL'})
SET c.name = '肾功能不全者', c.severity = 'critical',
    c.reason = '高蛋白摄入加重肾脏负担，产品注意事项明确慎用';

MERGE (c:Contraindication {contra_id: 'CI_PLANT_ALLERGY'})
SET c.name = '植物蛋白过敏者', c.severity = 'critical',
    c.reason = '大豆分离蛋白为植物来源，过敏者不宜使用';

MERGE (c:Contraindication {contra_id: 'CI_ANIMAL_ALLERGY'})
SET c.name = '动物蛋白过敏者', c.severity = 'critical',
    c.reason = '乳清蛋白为牛奶来源，过敏者不宜使用';

MERGE (c:Contraindication {contra_id: 'CI_DAMPHEAT'})
SET c.name = '湿热体质者', c.severity = 'high',
    c.reason = '人参原料目录明确列为不适宜人群';

MERGE (c:Contraindication {contra_id: 'CI_YINXU'})
SET c.name = '阴虚体质者', c.severity = 'high',
    c.reason = '人参原料目录明确列为不适宜人群';

MERGE (c:Contraindication {contra_id: 'CI_OBSTRUCTION'})
SET c.name = '肠道梗阻者', c.severity = 'critical',
    c.reason = '膳食纤维可能加重梗阻，产品注意事项明确禁用';

MERGE (c:Contraindication {contra_id: 'CI_COLD'})
SET c.name = '感冒期间', c.severity = 'medium',
    c.reason = '人参、西洋参注意事项列明感冒期间不推荐使用';


// ============================================================
// 第 5 步：适宜人群节点
// ============================================================

MERGE (t:TargetGroup {target_id: 'TG_XZPG'})
SET t.name = '血脂偏高者';

MERGE (t:TargetGroup {target_id: 'TG_ADULT'})
SET t.name = '成人';

MERGE (t:TargetGroup {target_id: 'TG_IMMUNO'})
SET t.name = '免疫力低下者';

MERGE (t:TargetGroup {target_id: 'TG_PROTEIN_DEF'})
SET t.name = '因蛋白质摄入不足而致免疫低下的成人';

MERGE (t:TargetGroup {target_id: 'TG_FATIGUE'})
SET t.name = '易疲劳者';

MERGE (t:TargetGroup {target_id: 'TG_NEED_SUPP'})
SET t.name = '需要补充的成人';

MERGE (t:TargetGroup {target_id: 'TG_GUT'})
SET t.name = '肠道菌群失调者';

MERGE (t:TargetGroup {target_id: 'TG_CONSTIPATION'})
SET t.name = '便秘者';


// ============================================================
// 第 6 步：成分节点
// ============================================================

MERGE (i:Ingredient {ingredient_id: 'ING_YUYOU'})
SET i.name = '鱼油', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_Q10'})
SET i.name = '辅酶Q10', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_TIHEISU'})
SET i.name = '褪黑素', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_RUQING'})
SET i.name = '乳清蛋白粉', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_DADOU'})
SET i.name = '大豆分离蛋白', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_LINGZHI_BZ'})
SET i.name = '破壁灵芝孢子粉', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_LUOXUANZAO'})
SET i.name = '螺旋藻', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_XIYANGSHEN'})
SET i.name = '西洋参', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_RENSHEN'})
SET i.name = '人参', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_LINGZHI'})
SET i.name = '灵芝', i.source = '保健食品原料目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_GAI'})
SET i.name = '钙', i.source = '营养素补充剂目录', i.data_origin = 'real';

MERGE (i:Ingredient {ingredient_id: 'ING_VD'})
SET i.name = '维生素D', i.source = '营养素补充剂目录', i.data_origin = 'real';


// ============================================================
// 第 7 步：产品 → 功效 关系（CLAIMS）
// ============================================================
// 严格对照：每个产品的批准功能
// 普通食品（P016–P020）没有任何 CLAIMS 关系

MATCH (p:Product {product_id: 'P001'}), (e:Efficacy {efficacy_id: 'EFF_19'})
MERGE (p)-[r:CLAIMS]->(e)
SET r.approval_no = '国食健注G20250012', r.is_approved = true;

MATCH (p:Product {product_id: 'P002'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '国食健注G20250013', r.is_approved = true;
MATCH (p:Product {product_id: 'P002'}), (e:Efficacy {efficacy_id: 'EFF_02'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '国食健注G20250013', r.is_approved = true;

MATCH (p:Product {product_id: 'P003'}), (e:Efficacy {efficacy_id: 'EFF_06'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000123', r.is_approved = true;

MATCH (p:Product {product_id: 'P004'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000124', r.is_approved = true;

MATCH (p:Product {product_id: 'P005'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000125', r.is_approved = true;

MATCH (p:Product {product_id: 'P006'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000126', r.is_approved = true;

MATCH (p:Product {product_id: 'P007'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000127', r.is_approved = true;

MATCH (p:Product {product_id: 'P008'}), (e:Efficacy {efficacy_id: 'EFF_07'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000128', r.is_approved = true;
MATCH (p:Product {product_id: 'P008'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000128', r.is_approved = true;

MATCH (p:Product {product_id: 'P009'}), (e:Efficacy {efficacy_id: 'EFF_07'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000129', r.is_approved = true;
MATCH (p:Product {product_id: 'P009'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000129', r.is_approved = true;

MATCH (p:Product {product_id: 'P010'}), (e:Efficacy {efficacy_id: 'EFF_01'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000130', r.is_approved = true;

MATCH (p:Product {product_id: 'P011'}), (e:Efficacy {efficacy_id: 'EFF_NUT_CA'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000131', r.is_approved = true;

MATCH (p:Product {product_id: 'P012'}), (e:Efficacy {efficacy_id: 'EFF_NUT_MV'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000132', r.is_approved = true;

MATCH (p:Product {product_id: 'P013'}), (e:Efficacy {efficacy_id: 'EFF_15'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000133', r.is_approved = true;

MATCH (p:Product {product_id: 'P014'}), (e:Efficacy {efficacy_id: 'EFF_17'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000134', r.is_approved = true;

MATCH (p:Product {product_id: 'P015'}), (e:Efficacy {efficacy_id: 'EFF_NUT_VC'})
MERGE (p)-[r:CLAIMS]->(e) SET r.approval_no = '食健备G202544000135', r.is_approved = true;


// ============================================================
// 第 8 步：产品 → 成分 关系（HAS_INGREDIENT）
// ============================================================

MATCH (p:Product {product_id: 'P001'}), (i:Ingredient {ingredient_id: 'ING_YUYOU'})
MERGE (p)-[r:HAS_INGREDIENT]->(i)
SET r.content = '每粒含鱼油1000mg', r.epa_dha = 'EPA+DHA合计300mg';

MATCH (p:Product {product_id: 'P002'}), (i:Ingredient {ingredient_id: 'ING_Q10'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每粒含辅酶Q10 40mg';

MATCH (p:Product {product_id: 'P003'}), (i:Ingredient {ingredient_id: 'ING_TIHEISU'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每片含褪黑素2mg';

MATCH (p:Product {product_id: 'P004'}), (i:Ingredient {ingredient_id: 'ING_RUQING'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每份20g含蛋白质15g';

MATCH (p:Product {product_id: 'P005'}), (i:Ingredient {ingredient_id: 'ING_DADOU'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每份20g含蛋白质16g';

MATCH (p:Product {product_id: 'P006'}), (i:Ingredient {ingredient_id: 'ING_LINGZHI_BZ'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每袋2g';

MATCH (p:Product {product_id: 'P007'}), (i:Ingredient {ingredient_id: 'ING_LUOXUANZAO'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每片0.5g';

MATCH (p:Product {product_id: 'P008'}), (i:Ingredient {ingredient_id: 'ING_XIYANGSHEN'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每片0.5g';

MATCH (p:Product {product_id: 'P009'}), (i:Ingredient {ingredient_id: 'ING_RENSHEN'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每粒0.4g';

MATCH (p:Product {product_id: 'P010'}), (i:Ingredient {ingredient_id: 'ING_LINGZHI'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '灵芝孢子油，每粒0.5g';

MATCH (p:Product {product_id: 'P011'}), (i:Ingredient {ingredient_id: 'ING_GAI'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每粒含钙250mg';
MATCH (p:Product {product_id: 'P011'}), (i:Ingredient {ingredient_id: 'ING_VD'})
MERGE (p)-[r:HAS_INGREDIENT]->(i) SET r.content = '每粒含维生素D3 5μg(200IU)';


// ============================================================
// 第 9 步：产品 → 适宜人群（SUITABLE_FOR）
// ============================================================

MATCH (p:Product {product_id: 'P001'}), (t:TargetGroup {target_id: 'TG_XZPG'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P002'}), (t:TargetGroup {target_id: 'TG_ADULT'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P003'}), (t:TargetGroup {target_id: 'TG_ADULT'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P004'}), (t:TargetGroup {target_id: 'TG_PROTEIN_DEF'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P005'}), (t:TargetGroup {target_id: 'TG_PROTEIN_DEF'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P006'}), (t:TargetGroup {target_id: 'TG_IMMUNO'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P007'}), (t:TargetGroup {target_id: 'TG_IMMUNO'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P008'}), (t:TargetGroup {target_id: 'TG_FATIGUE'})
MERGE (p)-[:SUITABLE_FOR]->(t);
MATCH (p:Product {product_id: 'P008'}), (t:TargetGroup {target_id: 'TG_IMMUNO'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P009'}), (t:TargetGroup {target_id: 'TG_FATIGUE'})
MERGE (p)-[:SUITABLE_FOR]->(t);
MATCH (p:Product {product_id: 'P009'}), (t:TargetGroup {target_id: 'TG_IMMUNO'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P010'}), (t:TargetGroup {target_id: 'TG_IMMUNO'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P011'}), (t:TargetGroup {target_id: 'TG_NEED_SUPP'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P012'}), (t:TargetGroup {target_id: 'TG_NEED_SUPP'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P013'}), (t:TargetGroup {target_id: 'TG_GUT'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P014'}), (t:TargetGroup {target_id: 'TG_CONSTIPATION'})
MERGE (p)-[:SUITABLE_FOR]->(t);

MATCH (p:Product {product_id: 'P015'}), (t:TargetGroup {target_id: 'TG_NEED_SUPP'})
MERGE (p)-[:SUITABLE_FOR]->(t);


// ============================================================
// 第 10 步：产品 → 不适宜人群（CONTRAINDICATED_FOR）
// ============================================================
// G2 纪律：必须有 reason（已在节点上定义，关系上可覆盖更具体的原因）

// ---- P001 鱼油 ----
MATCH (p:Product {product_id: 'P001'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P001'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P001'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P001'}), (c:Contraindication {contra_id: 'CI_BLEED'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '出血风险，与抗凝药同服尤需注意';
MATCH (p:Product {product_id: 'P001'}), (c:Contraindication {contra_id: 'CI_LIVER'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P001'}), (c:Contraindication {contra_id: 'CI_SEAFOOD'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '注意事项明确不推荐海产品过敏者食用';

// ---- P002 辅酶Q10 ----
MATCH (p:Product {product_id: 'P002'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P002'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P002'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P002'}), (c:Contraindication {contra_id: 'CI_ALLERGY'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P003 褪黑素 ----
MATCH (p:Product {product_id: 'P003'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P003'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P003'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P003'}), (c:Contraindication {contra_id: 'CI_DRIVER'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '嗜睡风险，操作前或操作中禁用';
MATCH (p:Product {product_id: 'P003'}), (c:Contraindication {contra_id: 'CI_AUTOIMMUNE'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P003'}), (c:Contraindication {contra_id: 'CI_HYPER'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P004 乳清蛋白 ----
MATCH (p:Product {product_id: 'P004'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P004'}), (c:Contraindication {contra_id: 'CI_RENAL'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '产品注意事项：肾功能不全者慎用';
MATCH (p:Product {product_id: 'P004'}), (c:Contraindication {contra_id: 'CI_ANIMAL_ALLERGY'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '动物蛋白过敏者不宜使用';

// ---- P005 大豆蛋白 ----
MATCH (p:Product {product_id: 'P005'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P005'}), (c:Contraindication {contra_id: 'CI_RENAL'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '产品注意事项：肾功能不全者慎用';
MATCH (p:Product {product_id: 'P005'}), (c:Contraindication {contra_id: 'CI_PLANT_ALLERGY'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '植物蛋白过敏者不宜使用';

// ---- P006 灵芝孢子粉 ----
MATCH (p:Product {product_id: 'P006'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P006'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P006'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P007 螺旋藻 ----
MATCH (p:Product {product_id: 'P007'}), (c:Contraindication {contra_id: 'CI_INFANT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P007'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P007'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P007'}), (c:Contraindication {contra_id: 'CI_ALLERGY'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P008 西洋参 ----
MATCH (p:Product {product_id: 'P008'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P008'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P008'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P008'}), (c:Contraindication {contra_id: 'CI_COLD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P009 人参 ----
MATCH (p:Product {product_id: 'P009'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P009'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P009'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P009'}), (c:Contraindication {contra_id: 'CI_DAMPHEAT'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '湿热体质不适宜，原料目录明确规定';
MATCH (p:Product {product_id: 'P009'}), (c:Contraindication {contra_id: 'CI_YINXU'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '阴虚体质不适宜，原料目录明确规定';
MATCH (p:Product {product_id: 'P009'}), (c:Contraindication {contra_id: 'CI_COLD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P010 灵芝孢子油 ----
MATCH (p:Product {product_id: 'P010'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P010'}), (c:Contraindication {contra_id: 'CI_PREG'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);
MATCH (p:Product {product_id: 'P010'}), (c:Contraindication {contra_id: 'CI_LACT'})
MERGE (p)-[:CONTRAINDICATED_FOR]->(c);

// ---- P011 钙维D ----
MATCH (p:Product {product_id: 'P011'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '按年龄段区分，需查具体批件';

// ---- P014 膳食纤维 ----
MATCH (p:Product {product_id: 'P014'}), (c:Contraindication {contra_id: 'CI_OBSTRUCTION'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c) SET r.reason = '肠道梗阻者禁用，可能加重梗阻';


// ============================================================
// 第 11 步：配伍禁忌与相似产品
// ============================================================
// G3 纪律：SIMILAR_TO 只存客观差异，不存优劣

// ---- 中药配伍禁忌 ----
MERGE (h1:Ingredient {ingredient_id: 'HERB_LILU'})
SET h1.name = '藜芦（中草药）', h1.type = '中药', h1.data_origin = 'real';

MERGE (h2:Ingredient {ingredient_id: 'HERB_WULINGZHI'})
SET h2.name = '五灵脂（中草药）', h2.type = '中药', h2.data_origin = 'real';

MATCH (p:Product {product_id: 'P009'}), (h:Ingredient {ingredient_id: 'HERB_LILU'})
MERGE (p)-[r:INCOMPATIBLE_WITH]->(h)
SET r.reason = '人参不宜与含藜芦的中草药同用', r.severity = 'critical';

MATCH (p:Product {product_id: 'P009'}), (h:Ingredient {ingredient_id: 'HERB_WULINGZHI'})
MERGE (p)-[r:INCOMPATIBLE_WITH]->(h)
SET r.reason = '人参不宜与含五灵脂的中草药同用', r.severity = 'critical';

MATCH (p:Product {product_id: 'P008'}), (h:Ingredient {ingredient_id: 'HERB_LILU'})
MERGE (p)-[r:INCOMPATIBLE_WITH]->(h)
SET r.reason = '西洋参不宜与含藜芦的中草药同用', r.severity = 'critical';

// ---- 相似产品（只存客观差异）----
MATCH (a:Product {product_id: 'P004'}), (b:Product {product_id: 'P016'})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.difference = 'P004为保健食品（食健备G202544000124），批准功能「有助于增强免疫力」，每份蛋白15g；P016为普通食品（SC11344010100123），无任何功能声称，每份蛋白12g',
    r.lesson_point = '外观相似成分相似，类别不同，这是最容易出错的地方';

MATCH (a:Product {product_id: 'P005'}), (b:Product {product_id: 'P016'})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.difference = 'P005为保健食品，有批准功能；P016为普通食品，无功能声称',
    r.lesson_point = '同上';

MATCH (a:Product {product_id: 'P006'}), (b:Product {product_id: 'P010'})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.difference = 'P006为粉剂需冲服（598元），P010为软胶囊携带方便（698元）；两者批准功能相同，均为「有助于增强免疫力」',
    r.lesson_point = '不做优劣评价，只陈述形态与价格差异';

MATCH (a:Product {product_id: 'P001'}), (b:Product {product_id: 'P002'})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.difference = 'P001批准功能为「有助于维持血脂健康水平」；P002为「有助于增强免疫力」+「有助于抗氧化」。两者均无与心脏相关的功能声称',
    r.lesson_point = '客户常误以为辅酶Q10能"护心"，实际批准功能里没有';


// ============================================================
// 第 12 步：使用场景与需求
// ============================================================

MERGE (s:Scenario {scenario_id: 'SCN_LIPID'}) SET s.name = '血脂管理';
MERGE (s:Scenario {scenario_id: 'SCN_IMMUNE'}) SET s.name = '免疫调节';
MERGE (s:Scenario {scenario_id: 'SCN_SLEEP'}) SET s.name = '睡眠改善';
MERGE (s:Scenario {scenario_id: 'SCN_PROTEIN'}) SET s.name = '蛋白补充';
MERGE (s:Scenario {scenario_id: 'SCN_FATIGUE'}) SET s.name = '抗疲劳';
MERGE (s:Scenario {scenario_id: 'SCN_BONE'}) SET s.name = '骨骼健康';
MERGE (s:Scenario {scenario_id: 'SCN_GUT'}) SET s.name = '肠道调理';
MERGE (s:Scenario {scenario_id: 'SCN_DAILY'}) SET s.name = '日常养生';

MATCH (p:Product {product_id: 'P001'}), (s:Scenario {scenario_id: 'SCN_LIPID'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P002'}), (s:Scenario {scenario_id: 'SCN_IMMUNE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P003'}), (s:Scenario {scenario_id: 'SCN_SLEEP'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P004'}), (s:Scenario {scenario_id: 'SCN_PROTEIN'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P005'}), (s:Scenario {scenario_id: 'SCN_PROTEIN'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P006'}), (s:Scenario {scenario_id: 'SCN_IMMUNE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P007'}), (s:Scenario {scenario_id: 'SCN_IMMUNE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P008'}), (s:Scenario {scenario_id: 'SCN_FATIGUE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P009'}), (s:Scenario {scenario_id: 'SCN_FATIGUE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P010'}), (s:Scenario {scenario_id: 'SCN_IMMUNE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P011'}), (s:Scenario {scenario_id: 'SCN_BONE'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P013'}), (s:Scenario {scenario_id: 'SCN_GUT'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P014'}), (s:Scenario {scenario_id: 'SCN_GUT'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P018'}), (s:Scenario {scenario_id: 'SCN_DAILY'})
MERGE (p)-[:USED_IN]->(s);
MATCH (p:Product {product_id: 'P019'}), (s:Scenario {scenario_id: 'SCN_DAILY'})
MERGE (p)-[:USED_IN]->(s);

MERGE (n:Need {need_id: 'NEED_LIPID'}) SET n.name = '血脂健康关注';
MERGE (n:Need {need_id: 'NEED_IMMUNE'}) SET n.name = '免疫力关注';
MERGE (n:Need {need_id: 'NEED_SLEEP'}) SET n.name = '睡眠关注';
MERGE (n:Need {need_id: 'NEED_PROTEIN'}) SET n.name = '蛋白质摄入不足';
MERGE (n:Need {need_id: 'NEED_FATIGUE'}) SET n.name = '易疲劳';
MERGE (n:Need {need_id: 'NEED_BONE'}) SET n.name = '骨骼健康关注';
MERGE (n:Need {need_id: 'NEED_GUT'}) SET n.name = '肠道健康关注';

MATCH (e:Efficacy {efficacy_id: 'EFF_19'}), (n:Need {need_id: 'NEED_LIPID'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_01'}), (n:Need {need_id: 'NEED_IMMUNE'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_06'}), (n:Need {need_id: 'NEED_SLEEP'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_07'}), (n:Need {need_id: 'NEED_FATIGUE'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_10'}), (n:Need {need_id: 'NEED_BONE'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_15'}), (n:Need {need_id: 'NEED_GUT'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_17'}), (n:Need {need_id: 'NEED_GUT'})
MERGE (e)-[:ADDRESSES]->(n);
MATCH (e:Efficacy {efficacy_id: 'EFF_NUT_CA'}), (n:Need {need_id: 'NEED_BONE'})
MERGE (e)-[:ADDRESSES]->(n);


// ============================================================
// 第 13 步：校验查询
// ============================================================

// 13.1 节点统计
// MATCH (n) RETURN labels(n)[0] AS 节点类型, count(n) AS 数量 ORDER BY 数量 DESC;

// 13.2 关系统计
// MATCH ()-[r]->() RETURN type(r) AS 关系类型, count(r) AS 数量 ORDER BY 数量 DESC;

// 13.3 数据完整性检查：普通食品不应有 CLAIMS 关系（应为 0）
// MATCH (p:Product {category:'普通食品'})-[r:CLAIMS]->(e)
// RETURN p.name, e.name;

// 13.4 数据完整性检查：所有 CONTRAINDICATED_FOR 关系都应有 reason
// MATCH (p)-[r:CONTRAINDICATED_FOR]->(c)
// WHERE r.reason IS NULL AND c.reason IS NULL
// RETURN p.name, c.name;

// 13.5 数据完整性检查：所有 Efficacy 都应在目录内（catalog_no ≤ 24 或 type=营养素补充剂）
// MATCH (e:Efficacy)
// WHERE e.catalog_no IS NULL AND e.type IS NULL
// RETURN e.name;

// 13.6 典型查询：客户血脂偏高，有哪些产品
// MATCH (p:Product)-[:CLAIMS]->(e:Efficacy {efficacy_id:'EFF_19'})
// RETURN p.name, p.category, p.approval_no, e.name;

// 13.7 典型查询：客户在服抗凝药，哪些产品要排除（图谱最核心的能力）
// MATCH (p:Product)-[r:CONTRAINDICATED_FOR]->(c:Contraindication {contra_id:'CI_BLEED'})
// RETURN p.name, c.name, r.reason, c.severity;

// 13.8 典型查询：某产品的完整画像
// MATCH (p:Product {product_id:'P001'})
// OPTIONAL MATCH (p)-[:CLAIMS]->(e:Efficacy)
// OPTIONAL MATCH (p)-[:HAS_INGREDIENT]->(i:Ingredient)
// OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(t:TargetGroup)
// OPTIONAL MATCH (p)-[:CONTRAINDICATED_FOR]->(c:Contraindication)
// RETURN p.name, collect(DISTINCT e.name) AS 批准功能,
//        collect(DISTINCT i.name) AS 成分,
//        collect(DISTINCT t.name) AS 适宜人群,
//        collect(DISTINCT c.name) AS 不适宜人群;

// 13.9 典型查询：容易混淆的产品对（教学重点）
// MATCH (a:Product)-[r:SIMILAR_TO]->(b:Product)
// RETURN a.name, b.name, r.difference, r.lesson_point;

// 13.10 典型查询：从需求反查产品（排除不适宜人群）
// MATCH (n:Need {need_id:'NEED_LIPID'})<-[:ADDRESSES]-(e:Efficacy)<-[:CLAIMS]-(p:Product)
// OPTIONAL MATCH (p)-[:CONTRAINDICATED_FOR]->(c:Contraindication)
// RETURN p.name, p.category, collect(c.name) AS 禁忌人群;
