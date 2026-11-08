// ============================================================
// 补充脚本 02：补齐 5 个普通食品的用法与别名
// Database: jiankangshujuku
// 说明：首轮补充脚本只覆盖了 P001–P015（保健食品），
//       普通食品 P016–P020 缺 DOSED_AS 与 ALIAS_OF
// ============================================================


// ============================================================
// 一、普通食品的用法用量节点
// ============================================================

MERGE (u:Usage {usage_id: 'USE_P016'})
SET u.dosage = '每日1–2次，每次1份（20g），温水冲调',
    u.daily_limit = '按说明，作为日常饮食的一部分',
    u.note = '普通食品，无保健功能声称；不能替代正常饮食';

MERGE (u:Usage {usage_id: 'USE_P017'})
SET u.dosage = '每日1瓶（50ml），可直接饮用或冷藏后饮用',
    u.daily_limit = '按说明',
    u.note = '普通食品，无保健功能声称';

MERGE (u:Usage {usage_id: 'USE_P018'})
SET u.dosage = '每日1–2袋（30ml），可直接饮用或兑水',
    u.daily_limit = '按说明',
    u.note = '普通食品，无保健功能声称';

MERGE (u:Usage {usage_id: 'USE_P019'})
SET u.dosage = '每日1–2丸（10g/丸），作为日常零食',
    u.daily_limit = '按说明',
    u.note = '普通食品，无保健功能声称';

MERGE (u:Usage {usage_id: 'USE_P020'})
SET u.dosage = '每日1次，每次1份（约40g），温水冲调',
    u.daily_limit = '按说明',
    u.note = '普通食品，无保健功能声称；不建议完全替代正餐，具体请咨询营养专业人员';

// ---- 建立产品 → 用法 关系 ----
MATCH (p:Product {product_id:'P016'}), (u:Usage {usage_id:'USE_P016'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P017'}), (u:Usage {usage_id:'USE_P017'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P018'}), (u:Usage {usage_id:'USE_P018'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P019'}), (u:Usage {usage_id:'USE_P019'}) MERGE (p)-[:DOSED_AS]->(u);
MATCH (p:Product {product_id:'P020'}), (u:Usage {usage_id:'USE_P020'}) MERGE (p)-[:DOSED_AS]->(u);


// ============================================================
// 二、普通食品的别名
// ============================================================

MERGE (a:Alias {alias_id: 'AL_P016_1'}) SET a.name = '复合蛋白粉', a.product_id = 'P016';
MERGE (a:Alias {alias_id: 'AL_P016_2'}) SET a.name = '固体饮料', a.product_id = 'P016';

MERGE (a:Alias {alias_id: 'AL_P017_1'}) SET a.name = '胶原蛋白', a.product_id = 'P017';
MERGE (a:Alias {alias_id: 'AL_P017_2'}) SET a.name = '胶原蛋白肽', a.product_id = 'P017';

MERGE (a:Alias {alias_id: 'AL_P018_1'}) SET a.name = '枸杞原浆', a.product_id = 'P018';
MERGE (a:Alias {alias_id: 'AL_P018_2'}) SET a.name = '枸杞汁', a.product_id = 'P018';

MERGE (a:Alias {alias_id: 'AL_P019_1'}) SET a.name = '黑芝麻丸', a.product_id = 'P019';
MERGE (a:Alias {alias_id: 'AL_P019_2'}) SET a.name = '芝麻丸', a.product_id = 'P019';

MERGE (a:Alias {alias_id: 'AL_P020_1'}) SET a.name = '代餐', a.product_id = 'P020';
MERGE (a:Alias {alias_id: 'AL_P020_2'}) SET a.name = '奶昔', a.product_id = 'P020';

// ---- 别名 → 产品 ----
MATCH (p:Product {product_id:'P016'}), (a:Alias {product_id:'P016'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P017'}), (a:Alias {product_id:'P017'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P018'}), (a:Alias {product_id:'P018'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P019'}), (a:Alias {product_id:'P019'}) MERGE (a)-[:ALIAS_OF]->(p);
MATCH (p:Product {product_id:'P020'}), (a:Alias {product_id:'P020'}) MERGE (a)-[:ALIAS_OF]->(p);


// ============================================================
// 三、普通食品的"禁止功能声称"标记关系
// ============================================================
// 说明：为了让检索层能快速判断"这个产品不能谈功效"，
//       给普通食品挂一个特殊标记节点

MERGE (w:WarningMark {mark_id: 'MARK_NO_CLAIM'})
SET w.name = '禁止功能声称',
    w.description = '普通食品不得作任何保健功能声称，连"有助于"也不可以',
    w.level = 'critical',
    w.source = '《食品安全法》《广告法》';

MATCH (p:Product {category:'普通食品'}), (w:WarningMark {mark_id:'MARK_NO_CLAIM'})
MERGE (p)-[:HAS_WARNING]->(w);
