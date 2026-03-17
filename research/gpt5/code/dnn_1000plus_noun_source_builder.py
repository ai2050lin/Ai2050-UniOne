from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv


CATEGORY_BANK: Dict[str, List[Tuple[str, str]]] = {
    "fruit": [
        ("apple", "苹果"), ("banana", "香蕉"), ("orange", "橙子"), ("grape", "葡萄"), ("pear", "梨"),
        ("peach", "桃子"), ("mango", "芒果"), ("lemon", "柠檬"), ("strawberry", "草莓"), ("watermelon", "西瓜"),
        ("pineapple", "菠萝"), ("cherry", "樱桃"), ("plum", "李子"), ("kiwi", "猕猴桃"), ("coconut", "椰子"),
        ("papaya", "木瓜"), ("guava", "番石榴"), ("fig", "无花果"), ("date", "枣"), ("lychee", "荔枝"),
        ("longan", "龙眼"), ("pomegranate", "石榴"), ("blueberry", "蓝莓"), ("raspberry", "覆盆子"), ("blackberry", "黑莓"),
        ("apricot", "杏"), ("persimmon", "柿子"), ("melon", "甜瓜"), ("lime", "青柠"), ("tangerine", "橘子"),
        ("pomelo", "柚子"), ("durian", "榴莲"), ("jackfruit", "菠萝蜜"), ("avocado", "牛油果"), ("cranberry", "蔓越莓"),
        ("mulberry", "桑葚"), ("dragonfruit", "火龙果"), ("passionfruit", "百香果"), ("starfruit", "杨桃"), ("nectarine", "油桃"),
        ("currant", "醋栗"), ("grapefruit", "葡萄柚"), ("mandarin", "蜜橘"), ("olive", "橄榄"), ("quince", "榅桲"),
        ("soursop", "刺果番荔枝"), ("loquat", "枇杷"), ("jabuticaba", "嘉宝果"), ("rambutan", "红毛丹"), ("salak", "蛇皮果"),
        ("breadfruit", "面包果"), ("plantain", "大蕉"),
    ],
    "animal": [
        ("cat", "猫"), ("dog", "狗"), ("rabbit", "兔子"), ("horse", "马"), ("tiger", "老虎"),
        ("lion", "狮子"), ("wolf", "狼"), ("bear", "熊"), ("deer", "鹿"), ("monkey", "猴子"),
        ("elephant", "大象"), ("bird", "鸟"), ("fish", "鱼"), ("goat", "山羊"), ("zebra", "斑马"),
        ("giraffe", "长颈鹿"), ("camel", "骆驼"), ("cow", "牛"), ("sheep", "绵羊"), ("pig", "猪"),
        ("mouse", "老鼠"), ("rat", "鼠"), ("fox", "狐狸"), ("panda", "熊猫"), ("kangaroo", "袋鼠"),
        ("dolphin", "海豚"), ("whale", "鲸鱼"), ("shark", "鲨鱼"), ("eagle", "鹰"), ("sparrow", "麻雀"),
        ("owl", "猫头鹰"), ("parrot", "鹦鹉"), ("penguin", "企鹅"), ("frog", "青蛙"), ("snake", "蛇"),
        ("lizard", "蜥蜴"), ("crocodile", "鳄鱼"), ("hippopotamus", "河马"), ("rhinoceros", "犀牛"), ("ant", "蚂蚁"),
        ("bee", "蜜蜂"), ("butterfly", "蝴蝶"), ("spider", "蜘蛛"), ("crab", "螃蟹"), ("lobster", "龙虾"),
        ("octopus", "章鱼"), ("seal", "海豹"), ("otter", "水獭"), ("yak", "牦牛"), ("donkey", "驴"),
        ("swan", "天鹅"), ("peacock", "孔雀"),
    ],
    "celestial": [
        ("sun", "太阳"), ("moon", "月亮"), ("star", "恒星"), ("planet", "行星"), ("comet", "彗星"),
        ("galaxy", "星系"), ("asteroid", "小行星"), ("meteor", "流星"), ("satellite", "卫星"), ("nebula", "星云"),
        ("mercury", "水星"), ("venus", "金星"), ("earth", "地球"), ("mars", "火星"), ("jupiter", "木星"),
        ("saturn", "土星"), ("uranus", "天王星"), ("neptune", "海王星"), ("pluto", "冥王星"), ("eclipse", "食"),
        ("orbit", "轨道"), ("constellation", "星座"), ("supernova", "超新星"), ("blackhole", "黑洞"), ("quasar", "类星体"),
        ("pulsar", "脉冲星"), ("cosmos", "宇宙"), ("milkyway", "银河"), ("cluster", "星团"), ("sunspot", "太阳黑子"),
        ("moonlight", "月光"), ("starlight", "星光"), ("daybreak", "破晓"), ("dusk", "黄昏"), ("solstice", "至日"),
        ("equinox", "分点"), ("tide", "潮汐"), ("aurora", "极光"), ("zenith", "天顶"), ("nadir", "天底"),
        ("lunarphase", "月相"), ("crescent", "新月"), ("fullmoon", "满月"), ("equinoxline", "分界线"), ("firmament", "苍穹"),
        ("skylight", "天光"), ("horizon", "地平线"), ("asterism", "星群"), ("heliosphere", "日球层"), ("magnetosphere", "磁层"),
        ("exoplanet", "系外行星"), ("interstellar", "星际空间"),
    ],
    "vehicle": [
        ("car", "汽车"), ("bus", "公交车"), ("train", "火车"), ("bicycle", "自行车"), ("airplane", "飞机"),
        ("ship", "轮船"), ("truck", "卡车"), ("motorcycle", "摩托车"), ("subway", "地铁"), ("boat", "小船"),
        ("tram", "电车"), ("helicopter", "直升机"), ("scooter", "踏板车"), ("van", "面包车"), ("taxi", "出租车"),
        ("rocket", "火箭"), ("spaceship", "飞船"), ("canoe", "独木舟"), ("kayak", "皮划艇"), ("ferry", "渡船"),
        ("yacht", "游艇"), ("submarine", "潜艇"), ("glider", "滑翔机"), ("tractor", "拖拉机"), ("bulldozer", "推土机"),
        ("ambulance", "救护车"), ("firetruck", "消防车"), ("policecar", "警车"), ("tank", "坦克"), ("drone", "无人机"),
        ("wagon", "马车"), ("cart", "手推车"), ("skateboard", "滑板"), ("sled", "雪橇"), ("gondola", "贡多拉"),
        ("raft", "木筏"), ("minibus", "中巴"), ("monorail", "单轨列车"), ("jet", "喷气机"), ("airship", "飞艇"),
        ("hovercraft", "气垫船"), ("cruiser", "巡洋舰"), ("destroyer", "驱逐舰"), ("carrier", "航母"), ("locomotive", "机车"),
        ("rickshaw", "人力车"), ("segway", "平衡车"), ("snowmobile", "雪地摩托"), ("forklift", "叉车"), ("pickup", "皮卡"),
        ("camper", "房车"), ("rowboat", "划艇"),
    ],
    "object": [
        ("chair", "椅子"), ("table", "桌子"), ("bed", "床"), ("lamp", "台灯"), ("door", "门"),
        ("window", "窗户"), ("bottle", "瓶子"), ("cup", "杯子"), ("spoon", "勺子"), ("knife", "刀"),
        ("phone", "手机"), ("computer", "电脑"), ("keyboard", "键盘"), ("clock", "时钟"), ("mirror", "镜子"),
        ("book", "书"), ("pen", "笔"), ("pencil", "铅笔"), ("notebook", "笔记本"), ("bag", "包"),
        ("box", "盒子"), ("plate", "盘子"), ("bowl", "碗"), ("fork", "叉子"), ("umbrella", "雨伞"),
        ("glasses", "眼镜"), ("wallet", "钱包"), ("camera", "相机"), ("television", "电视"), ("remote", "遥控器"),
        ("blanket", "毯子"), ("pillow", "枕头"), ("sofa", "沙发"), ("cabinet", "柜子"), ("shelf", "架子"),
        ("brush", "刷子"), ("comb", "梳子"), ("towel", "毛巾"), ("soap", "肥皂"), ("bucket", "水桶"),
        ("rope", "绳子"), ("candle", "蜡烛"), ("vase", "花瓶"), ("helmet", "头盔"), ("ring", "戒指"),
        ("necklace", "项链"), ("watch", "手表"), ("scissors", "剪刀"), ("stapler", "订书机"), ("hammer", "锤子"),
        ("screwdriver", "螺丝刀"), ("needle", "针"),
    ],
    "food": [
        ("bread", "面包"), ("rice", "米饭"), ("meat", "肉"), ("soup", "汤"), ("pizza", "披萨"),
        ("cake", "蛋糕"), ("coffee", "咖啡"), ("tea", "茶"), ("milk", "牛奶"), ("cheese", "奶酪"),
        ("noodle", "面条"), ("egg", "鸡蛋"), ("salad", "沙拉"), ("butter", "黄油"), ("chocolate", "巧克力"),
        ("applepie", "苹果派"), ("dumpling", "饺子"), ("sandwich", "三明治"), ("burger", "汉堡"), ("sausage", "香肠"),
        ("steak", "牛排"), ("pancake", "煎饼"), ("porridge", "粥"), ("yogurt", "酸奶"), ("icecream", "冰淇淋"),
        ("cookie", "饼干"), ("jam", "果酱"), ("honey", "蜂蜜"), ("tofu", "豆腐"), ("bean", "豆子"),
        ("corn", "玉米"), ("potato", "土豆"), ("carrot", "胡萝卜"), ("cabbage", "卷心菜"), ("spinach", "菠菜"),
        ("onion", "洋葱"), ("garlic", "大蒜"), ("ginger", "姜"), ("pepper", "辣椒"), ("salt", "盐"),
        ("sugar", "糖"), ("vinegar", "醋"), ("soy", "酱油"), ("fishcake", "鱼饼"), ("sausagebun", "香肠面包"),
        ("friedrice", "炒饭"), ("sushi", "寿司"), ("ramen", "拉面"), ("hotpot", "火锅"), ("pudding", "布丁"),
        ("omelet", "煎蛋卷"), ("waffle", "华夫饼"),
    ],
    "nature": [
        ("tree", "树"), ("flower", "花"), ("grass", "草"), ("forest", "森林"), ("river", "河流"),
        ("mountain", "山"), ("ocean", "海洋"), ("desert", "沙漠"), ("leaf", "叶子"), ("seed", "种子"),
        ("cloud", "云"), ("rain", "雨"), ("snow", "雪"), ("wind", "风"), ("storm", "风暴"),
        ("stone", "石头"), ("rock", "岩石"), ("soil", "土壤"), ("sand", "沙子"), ("waterfall", "瀑布"),
        ("lake", "湖"), ("island", "岛"), ("valley", "山谷"), ("cave", "洞穴"), ("volcano", "火山"),
        ("glacier", "冰川"), ("treebark", "树皮"), ("branch", "树枝"), ("root", "树根"), ("blossom", "花朵"),
        ("petal", "花瓣"), ("thorn", "刺"), ("moss", "苔藓"), ("fern", "蕨类"), ("prairie", "草原"),
        ("jungle", "丛林"), ("swamp", "沼泽"), ("reef", "珊瑚礁"), ("wave", "海浪"), ("breeze", "微风"),
        ("fog", "雾"), ("hail", "冰雹"), ("lightning", "闪电"), ("thunder", "雷"), ("cliff", "悬崖"),
        ("pebble", "卵石"), ("meadow", "草地"), ("sapling", "树苗"), ("flowerbed", "花坛"), ("wood", "木材"),
        ("sunshine", "阳光"), ("moonbeam", "月光束"),
    ],
    "human": [
        ("teacher", "老师"), ("doctor", "医生"), ("student", "学生"), ("parent", "父母"), ("friend", "朋友"),
        ("king", "国王"), ("queen", "女王"), ("artist", "艺术家"), ("worker", "工人"), ("engineer", "工程师"),
        ("lawyer", "律师"), ("pilot", "飞行员"), ("farmer", "农民"), ("nurse", "护士"), ("child", "孩子"),
        ("writer", "作家"), ("poet", "诗人"), ("singer", "歌手"), ("dancer", "舞者"), ("chef", "厨师"),
        ("driver", "司机"), ("judge", "法官"), ("soldier", "士兵"), ("scientist", "科学家"), ("teacheraid", "助教"),
        ("manager", "经理"), ("clerk", "店员"), ("guard", "警卫"), ("reporter", "记者"), ("actor", "演员"),
        ("director", "导演"), ("architect", "建筑师"), ("designer", "设计师"), ("programmer", "程序员"), ("captain", "船长"),
        ("sailor", "水手"), ("miner", "矿工"), ("builder", "建筑工"), ("dentist", "牙医"), ("therapist", "治疗师"),
        ("librarian", "图书管理员"), ("cashier", "收银员"), ("bartender", "调酒师"), ("gardener", "园丁"), ("tailor", "裁缝"),
        ("barber", "理发师"), ("plumber", "水管工"), ("electrician", "电工"), ("athlete", "运动员"), ("coach", "教练"),
        ("diplomat", "外交官"), ("monk", "僧人"),
    ],
    "tech": [
        ("algorithm", "算法"), ("data", "数据"), ("number", "数字"), ("equation", "方程"), ("database", "数据库"),
        ("network", "网络"), ("software", "软件"), ("hardware", "硬件"), ("robot", "机器人"), ("chip", "芯片"),
        ("transformer", "变换器"), ("token", "词元"), ("embedding", "嵌入"), ("gradient", "梯度"), ("parameter", "参数"),
        ("server", "服务器"), ("client", "客户端"), ("browser", "浏览器"), ("compiler", "编译器"), ("kernel", "内核"),
        ("memory", "内存"), ("cache", "缓存"), ("sensor", "传感器"), ("signal", "信号"), ("circuit", "电路"),
        ("firmware", "固件"), ("protocol", "协议"), ("packet", "数据包"), ("module", "模块"), ("library", "库"),
        ("runtime", "运行时"), ("thread", "线程"), ("process", "进程"), ("cluster", "集群"), ("router", "路由器"),
        ("switch", "交换机"), ("screen", "屏幕"), ("battery", "电池"), ("transistor", "晶体管"), ("microphone", "麦克风"),
        ("speaker", "扬声器"), ("antenna", "天线"), ("satnav", "导航系统"), ("laser", "激光"), ("quantum", "量子比特"),
        ("dataset", "数据集"), ("optimizer", "优化器"), ("checkpoint", "检查点"), ("backbone", "主干"), ("tokenizer", "分词器"),
        ("decoder", "解码器"), ("encoder", "编码器"),
    ],
    "abstract": [
        ("justice", "正义"), ("truth", "真理"), ("freedom", "自由"), ("beauty", "美"), ("love", "爱"),
        ("hope", "希望"), ("fear", "恐惧"), ("wisdom", "智慧"), ("peace", "和平"), ("chaos", "混乱"),
        ("order", "秩序"), ("time", "时间"), ("space", "空间"), ("infinity", "无限"), ("memory", "记忆"),
        ("dream", "梦想"), ("idea", "思想"), ("belief", "信念"), ("value", "价值"), ("honor", "荣誉"),
        ("faith", "信仰"), ("reason", "理性"), ("logic", "逻辑"), ("emotion", "情感"), ("purpose", "目的"),
        ("identity", "身份"), ("meaning", "意义"), ("destiny", "命运"), ("luck", "运气"), ("history", "历史"),
        ("future", "未来"), ("past", "过去"), ("silence", "沉默"), ("language", "语言"), ("culture", "文化"),
        ("law", "法律"), ("power", "权力"), ("knowledge", "知识"), ("curiosity", "好奇"), ("patience", "耐心"),
        ("courage", "勇气"), ("kindness", "善良"), ("mercy", "仁慈"), ("glory", "荣耀"), ("harmony", "和谐"),
        ("conflict", "冲突"), ("balance", "平衡"), ("symmetry", "对称"), ("duality", "二元性"), ("signalness", "信号性"),
        ("ambiguity", "歧义"), ("certainty", "确定性"),
    ],
}


@dataclass
class Dnn1000PlusNounSourceBuilder:
    root: Path

    def build_rows(self) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        for category, pairs in CATEGORY_BANK.items():
            for en, zh in pairs:
                rows.append((en, category))
                rows.append((zh, category))
        return rows

    def write_csv(self, rel_path: str) -> Path:
        out_path = self.root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.build_rows()
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# noun", "category"])
            writer.writerows(rows)
        return out_path
