# Stage56 ICSPB 方程化块

## 方程
- anchor_fiber_relation_equation: H(term,ctx)=A_anchor(term)+F_modifier(term)+R_relation(term,ctx)+G_control(ctx)
- closure_equation: C = +logic_P + style_I + style_SB - logic_FB + syntax_CX - syntax_MD
- relation_projection_equation: Pi_relation = axis_specificity * local_linearity - hierarchy_gain * bundle_load
- layer_head_mlp_equation: L* = argmax_l [Spine_model(l) + syntax_CX*X_l + logic_P*P_l - logic_FB*FB_l]

## 检验
- closure_positive_terms_present: True
- closure_negative_terms_present: True
- relation_axis_is_local_not_global: True
