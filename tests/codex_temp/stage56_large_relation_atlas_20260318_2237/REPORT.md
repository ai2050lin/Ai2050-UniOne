# Stage56 大规模关系图谱块

- group_count: 25
- counts_by_interpretation: {'hybrid': 4, 'local_linear': 15, 'path_bundle': 6}
- counts_by_word_class: {'abstract_noun': 3, 'adjective': 5, 'adverb': 2, 'concept': 3, 'noun': 8, 'verb': 4}

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
- protocol_role / concept / ['protocol', 'client', 'thread', 'algorithm']: linear=0.2690, bundle=0.5897
- protocol_role / concept / ['teacher', 'engineer', 'captain', 'miner']: linear=0.2630, bundle=0.5897
- protocol_role / concept / ['create', 'help', 'explore', 'watch']: linear=0.3582, bundle=0.5897
- verb_process_chain / verb / ['plan', 'start', 'move', 'finish']: linear=0.3933, bundle=0.5785
- verb_process_chain / verb / ['think', 'reason', 'compare', 'solve']: linear=0.3933, bundle=0.5785
- adverb_manner / adverb / ['formally', 'casually', 'logically']: linear=0.4218, bundle=0.5340
