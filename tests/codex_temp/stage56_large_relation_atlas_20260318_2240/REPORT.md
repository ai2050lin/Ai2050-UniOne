# Stage56 大规模关系图谱块

- group_count: 241
- counts_by_interpretation: {'hybrid': 4, 'local_linear': 15, 'path_bundle': 222}
- counts_by_word_class: {'abstract_noun': 21, 'adjective': 5, 'adverb': 2, 'concept': 3, 'noun': 188, 'verb': 22}

## 最强局部线性样例
- gender_role_swap / noun / ['king', 'man', 'woman', 'queen']: linear=0.6517, bundle=0.3041
- gender_role_swap / noun / ['prince', 'man', 'woman', 'princess']: linear=0.6517, bundle=0.3041
- gender_role_swap / noun / ['actor', 'man', 'woman', 'actress']: linear=0.6517, bundle=0.3041
- gender_role_swap / noun / ['waiter', 'man', 'woman', 'waitress']: linear=0.6517, bundle=0.3041
- gender_role_swap / noun / ['hero', 'man', 'woman', 'heroine']: linear=0.6517, bundle=0.3041
- adjective_polarity / adjective / ['hot', 'cold', 'warm']: linear=0.5932, bundle=0.3868
- profession_role_swap / noun / ['teacher', 'student', 'mentor', 'apprentice']: linear=0.5926, bundle=0.3933
- profession_role_swap / noun / ['doctor', 'patient', 'lawyer', 'client']: linear=0.5926, bundle=0.3933
- profession_role_swap / noun / ['captain', 'crew', 'teacher', 'class']: linear=0.5926, bundle=0.3933
- adjective_degree / adjective / ['small', 'smaller', 'smallest']: linear=0.5898, bundle=0.4112
- verb_antonym / verb / ['open', 'close', 'lock']: linear=0.5812, bundle=0.4299
- verb_antonym / verb / ['create', 'destroy', 'build']: linear=0.5812, bundle=0.4299

## 最强路径束样例
- category_instance_quadruplet / abstract_noun / ['abstract', 'balance', 'beauty', 'chaos']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / abstract_noun / ['abstract', 'curiosity', 'emotion', 'glory']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / abstract_noun / ['abstract', 'harmony', 'history', 'honor']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / abstract_noun / ['abstract', 'idea', 'identity', 'justice']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / abstract_noun / ['abstract', 'language', 'love', 'luck']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / abstract_noun / ['abstract', 'meaning', 'order', 'past']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / abstract_noun / ['abstract', 'patience', 'power', 'purpose']: linear=0.2821, bundle=0.6117
- category_instance_quadruplet / verb / ['action', 'catch', 'climb', 'compare']: linear=0.3351, bundle=0.6079
- category_instance_quadruplet / verb / ['action', 'create', 'destroy', 'drive']: linear=0.3351, bundle=0.6079
- category_instance_quadruplet / verb / ['action', 'drop', 'explore', 'fly']: linear=0.3351, bundle=0.6079
- category_instance_quadruplet / verb / ['action', 'help', 'imagine', 'learn']: linear=0.3351, bundle=0.6079
- category_instance_quadruplet / verb / ['action', 'lift', 'move', 'open']: linear=0.3351, bundle=0.6079
