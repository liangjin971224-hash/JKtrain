// ============================================================
// 补充脚本 01：补齐遗漏的适宜/不适宜人群 + 用法节点
// Database: jiankangshujuku
// 说明：首轮入库后发现 P012/P013/P015 缺不适宜人群，
//       且全部产品缺 Usage 节点（用法用量），此处补齐
// ============================================================


// ============================================================
// 一、补齐 P012 / P013 / P015 的不适宜人群
// ============================================================

// P012 多种维生素矿物质片（营养素补充剂）
// 依据：营养素补充剂类产品需避免与其他同类补充剂叠加超量
MATCH (p:Product {product_id: 'P012'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c)
SET r.reason = '营养素补充剂，儿童用量需按年龄段确定，建议咨询医生';

// P013 益生菌粉
MATCH (p:Product {product_id: 'P013'}), (c:Contraindication {contra_id: 'CI_INFANT'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c)
SET r.reason = '婴幼儿慎用，需查具体批件';

// P015 维生素C咀嚼片
MATCH (p:Product {product_id: 'P015'}), (c:Contraindication {contra_id: 'CI_CHILD'})
MERGE (p)-[r:CONTRAINDICATED_FOR]->(c)
SET r.reason = '营养素补充剂，儿童用量需按年龄段确定';


// ============================================================
// 二、普通食品的说明性节点（不设适宜/不适宜人群）
// ============================================================
// 五个普通食品没有法定的"适宜人群/不适宜人群"标签，
// 但为便于检索，补充合规说明字段（已在入库时设置 compliance_note）

// 说明：普通食品不需要 SUITABLE_FOR / CONTRAINDICATED_FOR 关系。
// 这不是数据缺失，而是类别属性决定的。


// ============================================================
// 三、用法用量节点（Usage）
// ============================================================

MERGE (u:Usage {usage_id: 'USE_P001'})
SET u.dosage = '每日2次，每次2粒，随餐服用',
    u.daily_limit = '不超过4.0g（EPA+DHA不低于1.0g）',
    u.note = '随餐服用有利于脂溶性成分吸收';

MERGE (u:Usage {usage_id: 'USE_P002'})
SET u.dosage = '每日1次，每次1粒，餐后服用',
    u.daily_limit = '30–50mg';

MERGE (u:Usage {usage_id: 'USE_P003'})
SET u.dosage = '每日1次，每次1片，睡前30分钟服用',
    u.daily_limit = '1–3mg',
    u.note = '操作前或操作中不得食用';

MERGE (u:Usage {usage_id: 'USE_P004'})
SET u.dosage = '每日1–2次，每次1份（20g），温水冲调',
    u.daily_limit = '蛋白质6–25g（成人每日总量建议60g）';

MERGE (u:Usage {usage_id: 'USE_P005'})
SET u.dosage = '每日1–2次，每次1份（20g），温水冲调',
    u.daily_limit = '蛋白质6–25g';

MERGE (u:Usage {usage_id: 'USE_P006'})
SET u.dosage = '每日1–2次，每次1袋（2g），温水冲服',
    u.daily_limit = '1–4g';

MERGE (u:Usage {usage_id: 'USE_P007'})
SET u.dosage = '每日2–3次，每次2片',
    u.daily_limit = '3–4g';

MERGE (u:Usage {usage_id: 'USE_P008'})
SET u.dosage = '每日2次，每次2片，含服',
    u.daily_limit = '1.5–3g';

MERGE (u:Usage {usage_id: 'USE_P009'})
SET u.dosage = '每日1–2次，每次1粒',
    u.daily_limit = '1–3g',
    u.note = '长期食用建议根据个人体质酌情减量';

MERGE (u:Usage {usage_id: 'USE_P010'})
SET u.dosage = '每日2次，每次1粒',
    u.daily_limit = '灵芝4–6g（折算）';

MERGE (u:Usage {usage_id: 'USE_P011'})
SET u.dosage = '每日1次，每次1–2粒',
    u.daily_limit = '钙不超过2000mg',
    u.note = '不宜与同类营养素补充剂同时食用';

MERGE (u:Usage {usage_id: 'USE_P012'})
SET u.dosage = '每日1次，每次1片',
    u.daily_limit = '按说明，不宜叠加同类补充剂';

MERGE (u:Usage {usage_id: 'USE_P013'})
SET u.dosage = '每日1次，每次1袋',
    u.daily_limit = '按说明',
    u.note = '冲调水温不超过40℃；与抗生素同服应间隔2小时以上';

MERGE (u:Usage {usage_id: 'USE_P014'})
SET u.dosage = '每日1–2次，每次1袋，冲服',
    u.daily_limit = '按说明',
    u.note = '服用后应多饮水；不宜与药物同时服用（间隔1–2小时）';

MERGE (u:Usage {usage_id: 'USE_P015'})
SET u.dosage = '每日1次，每次1片，嚼服',
    u.daily_limit = '不超过2000mg';


// ---- 建立产品 → 用法 关系 ----

MATCH (p:Product {product_id:'P001'}), (u:Usage {usage_id:'USE_P001'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P002'}), (u:Usage {usage_id:'USE_P002'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P003'}), (u:Usage {usage_id:'USE_P003'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P004'}), (u:Usage {usage_id:'USE_P004'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P005'}), (u:Usage {usage_id:'USE_P005'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P006'}), (u:Usage {usage_id:'USE_P006'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P007'}), (u:Usage {usage_id:'USE_P007'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P008'}), (u:Usage {usage_id:'USE_P008'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P009'}), (u:Usage {usage_id:'USE_P009'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P010'}), (u:Usage {usage_id:'USE_P010'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P011'}), (u:Usage {usage_id:'USE_P011'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P012'}), (u:Usage {usage_id:'USE_P012'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P013'}), (u:Usage {usage_id:'USE_P013'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P014'}), (u:Usage {usage_id:'USE_P014'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P015'}), (u:Usage {usage_id:'USE_P015'}) MERGE (p)-[:DOSED_AS]->(u);


// ============================================================
// 四、产品别名节点（提升检索命中率）
// ============================================================
// 客户常说的口语化名称，需要能映射到标准产品

MERGE (a:Alias {alias_id: 'AL_P001_1'}) SET a.name = '鱼油', a.product_id = 'P001';
MERGE (a:Alias {alias_id: 'AL_P001_2'}) SET a.name = '深海鱼油', a.product_id = 'P001';
MERGE (a:Alias {alias_id: 'AL_P001_3'}) SET a.name = 'omega3', a.product_id = 'P001';
MERGE (a:Alias {alias_id: 'AL_P002_1'}) SET a.name = '辅酶Q10', a.product_id = 'P002';
MERGE (a:Alias {alias_id: 'AL_P002_2'}) SET a.name = 'Q10', a.product_id = 'P002';
MERGE (a:Alias {alias_id: 'AL_P003_1'}) SET a.name = '褪黑素', a.product_id = 'P003';
MERGE (a:Alias {alias_id: 'AL_P003_2'}) SET a.name = '助眠片', a.product_id = 'P003';
MERGE (a:Alias {alias_id: 'AL_P004_1'}) SET a.name = '蛋白粉', a.product_id = 'P004';
MERGE (a:Alias {alias_id: 'AL_P004_2'}) SET a.name = '乳清蛋白', a.product_id = 'P004';
MERGE (a:Alias {alias_id: 'AL_P005_1'}) SET a.name = '大豆蛋白', a.product_id = 'P005';
MERGE (a:Alias {alias_id: 'AL_P005_2'}) SET a.name = '植物蛋白', a.product_id = 'P005';
MERGE (a:Alias {alias_id: 'AL_P006_1'}) SET a.name = '灵芝孢子粉', a.product_id = 'P006';
MERGE (a:Alias {alias_id: 'AL_P006_2'}) SET a.name = '孢子粉', a.product_id = 'P006';
MERGE (a:Alias {alias_id: 'AL_P007_1'}) SET a.name = '螺旋藻', a.product_id = 'P007';
MERGE (a:Alias {alias_id: 'AL_P008_1'}) SET a.name = '西洋参', a.product_id = 'P008';
MERGE (a:Alias {alias_id: 'AL_P009_1'}) SET a.name = '人参', a.product_id = 'P009';
MERGE (a:Alias {alias_id: 'AL_P010_1'}) SET a.name = '灵芝孢子油', a.product_id = 'P010';
MERGE (a:Alias {alias_id: 'AL_P011_1'}) SET a.name = '钙片', a.product_id = 'P011';
MERGE (a:Alias {alias_id: 'AL_P011_2'}) SET a.name = '钙维D', a.product_id = 'P011';
MERGE (a:Alias {alias_id: 'AL_P012_1'}) SET a.name = '复合维生素', a.product_id = 'P012';
MERGE (a:Alias {alias_id: 'AL_P013_1'}) SET a.name = '益生菌', a.product_id = 'P013';
MERGE (a:Alias {alias_id: 'AL_P014_1'}) SET a.name = '膳食纤维', a.product_id = 'P014';
MERGE (a:Alias {alias_id: 'AL_P014_2'}) SET a.name = '纤维粉', a.product_id = 'P014';
MERGE (a:Alias {alias_id: 'AL_P015_1'}) SET a.name = '维生素C', a.product_id = 'P015';
MERGE (a:Alias {alias_id: 'AL_P015_2'}) SET a.name = 'VC', a.product_id = 'P015';

// ---- 别名 → 产品 ----
MATCH (p:Product {product_id:'P001'}), (a:Alias {product_id:'P001'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P002'}), (a:Alias {product_id:'P002'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P003'}), (a:Alias {product_id:'P003'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P004'}), (a:Alias {product_id:'P004'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P005'}), (a:Alias {product_id:'P005'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P006'}), (a:Alias {product_id:'P006'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P007'}), (a:Alias {product_id:'P007'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P008'}), (a:Alias {product_id:'P008'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P009'}), (a:Alias {product_id:'P009'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P010'}), (a:Alias {product_id:'P010'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P011'}), (a:Alias {product_id:'P011'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P012'}), (a:Alias {product_id:'P012'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P013'}), (a:Alias {product_id:'P013'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P014'}), (a:Alias {product_id:'P014'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P015'}), (a:Alias {product_id:'P015'}) MERGE (a)-[:ALIAS_OF]->(p);


// ============================================================
// 五、合规规则节点（把 D005 的禁用词清单结构化）
// ============================================================
// 说明：这些规则供 FR-026 违规宣称拦截使用

MERGE (r:Rule {rule_id: 'RULE_FORBID_TREAT'})
SET r.name = '疾病治疗类禁用表述',
    r.level = 'critical',
    r.category = 'forbidden_claim',
    r.patterns = ['治疗', '治愈', '根治', '疗效', '对症', '抗炎', '消炎',
                  '降血糖', '降血压', '降血脂', '防癌', '抗癌'],
    r.message = '保健食品不得宣称疾病治疗功能',
    r.source = 'D005 第二章 + 《食品安全法》《广告法》';

MERGE (r:Rule {rule_id: 'RULE_FORBID_DRUG_REPLACE'})
SET r.name = '药物替代类禁用表述',
    r.level = 'critical',
    r.category = 'forbidden_claim',
    r.patterns = ['替代药物', '不用吃药', '可以停药', '比药管用',
                  '药食同源无副作用', '天然无害可长期大量服用'],
    r.message = '不得暗示可替代药物治疗',
    r.source = 'D005 第二章 + B2 补充规范';

MERGE (r:Rule {rule_id: 'RULE_FORBID_ABSOLUTE'})
SET r.name = '绝对化禁用表述',
    r.level = 'critical',
    r.category = 'forbidden_claim',
    r.patterns = ['最有效', '第一', '唯一', '彻底解决', '100%有效',
                  '包治', '根治', '国家认证', '专家推荐', '医院同款'],
    r.message = '不得使用绝对化用语或无依据的权威暗示',
    r.source = 'D005 第二章 + 《广告法》';

MERGE (r:Rule {rule_id: 'RULE_FORBID_PROMISE'})
SET r.name = '承诺类禁用表述',
    r.level = 'critical',
    r.category = 'forbidden_claim',
    r.patterns = ['保证有效', '无效退款', '三天见效', '一个疗程见效',
                  '一定有效', '肯定有效'],
    r.message = '不得作出效果或时间承诺',
    r.source = 'D005 第二章';

MERGE (r:Rule {rule_id: 'RULE_FORBID_SOCIAL_PROOF'})
SET r.name = '他人经验背书禁用表述',
    r.level = 'high',
    r.category = 'forbidden_claim',
    r.patterns = ['很多人吃了都说好', '很多人吃了都', '别人都说', '我朋友吃了就',
                  '客户反馈都说', '老客户都在回购', '卖得最好'],
    r.message = '不得以他人经验作为效果背书',
    r.source = 'D005 第二章';

MERGE (r:Rule {rule_id: 'RULE_FORBID_PUSH'})
SET r.name = '主动推销禁用表述（B2 模式）',
    r.level = 'critical',
    r.category = 'forbidden_behavior',
    r.patterns = ['要不要买', '带两盒', '带一盒', '现在有活动', '买二送一',
                  '限时', '赶紧买', '今天最后一天', '先买'],
    r.message = 'B2 模式下不得主动推销，客户问才答',
    r.source = 'B2 模式定稿方案 + B2 补充规范';

MERGE (r:Rule {rule_id: 'RULE_FORBID_EVASION'})
SET r.name = '回避推诿禁用表述（B2 模式）',
    r.level = 'high',
    r.category = 'forbidden_behavior',
    r.patterns = ['这个我不清楚', '我不太了解', '这个不好说',
                  '我们不聊产品', '我们只做健康管理', '产品的事我不方便说',
                  '您问医生吧'],
    r.message = '客户询问产品信息必须如实回答；回避是失职',
    r.source = 'B2 补充规范-必须如实回答';

MERGE (r:Rule {rule_id: 'RULE_REQUIRE_WARNING'})
SET r.name = '强制警示语',
    r.level = 'required',
    r.category = 'required_text',
    r.text = '保健食品不是药物，不能代替药物治疗疾病',
    r.scene = '涉及任何保健食品时',
    r.message = '必须原文输出，不可省略不可改写',
    r.source = '《食品安全法》/ 保健食品标签管理规定';

MERGE (r:Rule {rule_id: 'RULE_REQUIRE_GROUP_NOTICE'})
SET r.name = '不适宜人群主动告知',
    r.level = 'required',
    r.category = 'required_text',
    r.scene = '客户询问某类人群能否食用时；客户提及自身情况涉及不适宜人群时',
    r.message = '必须完整列出不适宜人群，不得省略',
    r.source = 'D005 第三章';

MERGE (r:Rule {rule_id: 'RULE_REQUIRE_DOSAGE_LIMIT'})
SET r.name = '每日用量上限告知',
    r.level = 'required',
    r.category = 'required_text',
    r.scene = '涉及食用方法、用量询问时',
    r.message = '必须说明每日用量上限，不得建议超量',
    r.source = 'D005 第三章';


// ============================================================
// 六、校验
// ============================================================
// 6.1 应无产品缺用法
// MATCH (p:Product) WHERE NOT EXISTS((p)-[:DOSED_AS]->()) RETURN p.product_id, p.name;

// 6.2 保健食品都应有不适宜人群（普通食品不要求）
// MATCH (p:Product {category:'保健食品'})
// WHERE NOT EXISTS((p)-[:CONTRAINDICATED_FOR]->())
// RETURN p.product_id, p.name;

// 6.3 规则统计
// MATCH (r:Rule) RETURN r.category AS 类别, r.level AS 级别, count(r) AS 数量;
