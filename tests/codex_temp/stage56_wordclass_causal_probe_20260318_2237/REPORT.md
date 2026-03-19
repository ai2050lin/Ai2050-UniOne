# Stage56 词类因果探针块

- probe_count: 4
- strong_classes: ['adjective', 'verb', 'abstract_noun', 'adverb']

- adjective: mechanism=modifier_fiber, judgement=adjective_as_fiber, probe_strength=direct
- verb: mechanism=transport_operator_with_anchor_residue, judgement=verb_transport, probe_strength=direct
- abstract_noun: mechanism=relation_bundle, judgement=abstract_bundle, probe_strength=direct
- adverb: mechanism=control_axis_modifier, judgement=adverb_control_modifier, probe_strength=semi_direct
