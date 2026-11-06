| variant | seed | backends | done / 40 | missing |
|---|---|---|---|---|
| combined_d4       | 0    | optimised  | 40 / 40 |  0 |
| original          | 0    | primitives | 40 / 40 |  0 |
| original          | 1    | primitives |  7 / 40 | 33 |
| original          | 2    | optimised  | 22 / 40 | 18 |
| original          | 2    | primitives | 18 / 40 | 22 |
| projected_conv    | 0    | primitives | 40 / 40 |  0 |
| projected_conv    | 1    | primitives |  6 / 40 | 34 |
| projected_conv    | 2    | primitives | 18 / 40 | 22 |
| projected_conv_d4 | 0    | primitives | 40 / 40 |  0 |
| projected_conv_d4 | 2    | optimised  | 40 / 40 |  0 |
| projected_d4      | 0    | primitives | 40 / 40 |  0 |
| projected_d4      | 1    | primitives |  6 / 40 | 34 |
| projected_d4      | 2    | optimised  | 23 / 40 | 17 |
| projected_d4      | 2    | primitives | 17 / 40 | 23 |
| projected_lse     | 0    | primitives | 40 / 40 |  0 |
| projected_lse     | 1    | primitives |  3 / 40 | 37 |
| projected_lse     | 2    | optimised  |  3 / 40 | 37 |
| projected_lse     | 2    | primitives | 17 / 40 | 23 |

## Missing task IDs

- original seed 1 (primitives): 33 missing -> 08ed6ac7 1190e5a7 1c786137 25ff71a9 36fdfd69 3906de3d 3aa6fb7a 694f12f3 6cdd2623 6d75e8bb 6f8cd79b 74dd1130 7f4411dc 868de0fa 8731374e 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d406998b d43fd935 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d f8c80d96
- original seed 2 (optimised): 18 missing -> 0dfd9992 1bfc4729 1c786137 36fdfd69 3aa6fb7a 3bdb4ada 3eda0437 445eab21 623ea044 6cdd2623 6d75e8bb 6f8cd79b 74dd1130 7f4411dc 95990924 d406998b d43fd935 f8c80d96
- original seed 2 (primitives): 22 missing -> 08ed6ac7 1190e5a7 25ff71a9 3906de3d 694f12f3 868de0fa 8731374e 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d
- projected_conv seed 1 (primitives): 34 missing -> 08ed6ac7 1190e5a7 1c786137 25ff71a9 36fdfd69 3906de3d 3aa6fb7a 694f12f3 6cdd2623 6d75e8bb 6f8cd79b 74dd1130 7f4411dc 868de0fa 8731374e 95990924 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d406998b d43fd935 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d f8c80d96
- projected_conv seed 2 (primitives): 22 missing -> 08ed6ac7 1190e5a7 25ff71a9 3906de3d 694f12f3 868de0fa 8731374e 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d
- projected_d4 seed 1 (primitives): 34 missing -> 08ed6ac7 1190e5a7 1c786137 25ff71a9 36fdfd69 3906de3d 3aa6fb7a 694f12f3 6cdd2623 6d75e8bb 6f8cd79b 74dd1130 7f4411dc 868de0fa 8731374e 95990924 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d406998b d43fd935 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d f8c80d96
- projected_d4 seed 2 (optimised): 17 missing -> 0dfd9992 1bfc4729 1c786137 36fdfd69 3aa6fb7a 3bdb4ada 3eda0437 445eab21 623ea044 6cdd2623 6f8cd79b 74dd1130 7f4411dc 95990924 d406998b d43fd935 f8c80d96
- projected_d4 seed 2 (primitives): 23 missing -> 08ed6ac7 1190e5a7 25ff71a9 3906de3d 694f12f3 6d75e8bb 868de0fa 8731374e 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d
- projected_lse seed 1 (primitives): 37 missing -> 08ed6ac7 0dfd9992 1190e5a7 1c786137 25ff71a9 36fdfd69 3906de3d 3aa6fb7a 3bdb4ada 623ea044 694f12f3 6cdd2623 6d75e8bb 6f8cd79b 74dd1130 7f4411dc 868de0fa 8731374e 95990924 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d406998b d43fd935 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d f8c80d96
- projected_lse seed 2 (optimised): 37 missing -> 08ed6ac7 0dfd9992 1190e5a7 1bfc4729 1c786137 25ff71a9 36fdfd69 3906de3d 3aa6fb7a 3bdb4ada 3eda0437 445eab21 623ea044 694f12f3 6cdd2623 6f8cd79b 74dd1130 7f4411dc 868de0fa 8731374e 95990924 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d406998b d43fd935 d5d6de2d e5062a87 eb281b96 eb5a1d5d f8c80d96
- projected_lse seed 2 (primitives): 23 missing -> 08ed6ac7 1190e5a7 25ff71a9 3906de3d 694f12f3 6d75e8bb 868de0fa 8731374e 97999447 9aec4887 a416b8f3 a61ba2ce aedd82e4 b527c5c6 bd4472b8 c9f8e694 d06dbe63 d22278a0 d5d6de2d d9fac9be e5062a87 eb281b96 eb5a1d5d
