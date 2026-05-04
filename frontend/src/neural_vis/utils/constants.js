/**
 * 常量定义
 */
export const LAYER_GAP = 3.5;
export const PLANE_SIZE = 18;
export const SPHERE_BASE_SIZE = 0.2;
export const TRAJECTORY_LINE_WIDTH = 3;

// 类别颜色方案
export const CATEGORY_COLORS = {
  fruit: '#ff6b6b', animal: '#4ecdc4', vehicle: '#ffe66d', tool: '#a855f7',
  nature: '#34d399', food: '#f97316', person: '#ec4899', abstract: '#6366f1',
};

// 层功能颜色
export const LAYER_FUNC_COLORS = {
  lexical: '#ff6b6b', semantic: '#4ecdc4', syntactic: '#ffe66d', decision: '#a855f7',
};

// 子空间颜色
export const SUBSPACE_COLORS = {
  w_u: '#4ecdc4',         // W_U可见 - 青绿
  w_u_perp: '#ff6b6b',    // W_U⊥ - 红色
  grammar: '#ffe66d',      // 语法 - 黄色
  semantic: '#4ecdc4',     // 语义 - 青绿
  logic: '#a855f7',        // 逻辑 - 紫色
  dark_matter: '#f97316',  // 暗物质 - 橙色
};

// 语法角色颜色
export const GRAMMAR_ROLE_COLORS = {
  nsubj: '#ff6b6b', dobj: '#4ecdc4', amod: '#ffe66d', aux: '#a855f7',
  iobj: '#34d399', ccomp: '#f97316', xcomp: '#ec4899', mark: '#6366f1',
};

// 因果链颜色
export const CAUSAL_COLORS = {
  intervention: '#ff6b6b',  // 干预点 - 红
  propagation: '#4ecdc4',   // 传播 - 青
  decay: '#64748b',         // 衰减 - 灰
  flip: '#ffe66d',          // 翻转 - 黄
};

/**
 * delta_cos → 颜色映射: 1.0=红, 0.5=橙, 0.0=蓝
 */
export function deltaCosToColor(deltaCos) {
  const r = Math.max(0, Math.min(1, deltaCos));
  let red, green, blue;
  if (r > 0.5) {
    const t = (r - 0.5) * 2;
    red = Math.round(239 * t + 245 * (1 - t));
    green = Math.round(68 * t + 158 * (1 - t));
    blue = Math.round(68 * t + 11 * (1 - t));
  } else {
    const t = r * 2;
    red = Math.round(245 * t + 59 * (1 - t));
    green = Math.round(158 * t + 130 * (1 - t));
    blue = Math.round(11 * t + 246 * (1 - t));
  }
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}

/**
 * cos_with_wu → 颜色映射: 高=青绿(W_U对齐), 低=红色(W_U⊥)
 */
export function cosWuToColor(cosWu) {
  const r = Math.max(0, Math.min(1, cosWu));
  // 红(0) → 橙(0.3) → 黄(0.5) → 青(0.8) → 绿(1.0)
  let red, green, blue;
  if (r > 0.7) {
    const t = (r - 0.7) / 0.3;
    red = Math.round(78 * (1 - t) + 34 * t);
    green = Math.round(205 * (1 - t) + 211 * t);
    blue = Math.round(196 * (1 - t) + 153 * t);
  } else if (r > 0.4) {
    const t = (r - 0.4) / 0.3;
    red = Math.round(234 * (1 - t) + 78 * t);
    green = Math.round(179 * (1 - t) + 205 * t);
    blue = Math.round(8 * (1 - t) + 196 * t);
  } else {
    const t = r / 0.4;
    red = Math.round(239 * (1 - t) + 234 * t);
    green = Math.round(68 * (1 - t) + 179 * t);
    blue = Math.round(68 * (1 - t) + 8 * t);
  }
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}

/**
 * 比例值 → 颜色映射 (0=暗, 1=亮)
 */
export function ratioToColor(ratio, baseColor = [78, 205, 196]) {
  const r = Math.max(0, Math.min(1, ratio));
  const intensity = 0.2 + r * 0.8;
  const red = Math.round(baseColor[0] * intensity);
  const green = Math.round(baseColor[1] * intensity);
  const blue = Math.round(baseColor[2] * intensity);
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}
