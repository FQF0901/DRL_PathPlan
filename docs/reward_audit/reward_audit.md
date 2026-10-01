# P2 奖励审计报告（v5 剖面 C）

- 生成：2026-10-01 16:00:32（工具 `tools/reward_audit.py`）
- pre-v6 快照：`/tmp/opencode/v6_pre` @ `031cc1c`（reuse）
- ckpt：`/workspace/01_Proj/DRL_PathPlan/runs/_refs_rlbase/e_beta_prime/final.pt`（sha256 `sha256:c9e2d31e4b9d4f784335ffe693b70ddecd4b0ff0e2d4fb062c82aa947f13ea87`）
- 池：`/workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_val.json`，排除 `None`；采集 episode 1000 条
- 分类计数：{'arrive_dest': 441, 'out_of_road': 490, 'collision': 34, 'max_step': 35}
- 审计样本 501 条 / 反解样本 499 条（互斥，见 §样本清单）
- 重放口径：HEAD `RewardAggregator`（完整 v5 项集含 `low_speed`；`ttc`/`lane_boundary`/`lane_center` 默认关）；γ=0.99；反解复算容差 ≤ 0.5，剖面判定容差 ≤ 5.0

## 0. 结论

- 目标剖面达标（审计样本，未折扣均值 |Δ| ≤ 5.0）：**是**
- 每类 n ≥ 10：**是**
- 总样本 ≥ 50（审计/反解各自）：**是**
- rc=1 档：剖面判定 **通过**（|Δ|max=+3.656，每类 n≥10：是）
- rc=3 档：剖面判定 **通过**（|Δ|max=+3.568，每类 n≥10：是）
- rc=10 档：剖面判定 **通过**（|Δ|max=+2.759，每类 n≥10：是）
- rc=30 档：剖面判定 **通过**（|Δ|max=+1.877，每类 n≥10：是）
- 基准档（rc=1）arrive_dest：n=221 mean=+49.120（目标 +50，Δ=-0.880）
- 基准档（rc=1）collision：n=17 mean=-23.656（目标 -20，Δ=-3.656）
- 基准档（rc=1）out_of_road：n=245 mean=-15.668（目标 -15，Δ=-0.668）
- 基准档（rc=1）max_step：n=18 mean=-11.337（目标 -10，Δ=-1.337）

## 1. 样本清单（id + hash）

### audit（501 条）

| 类 | id | seed | ctx_sha256 | 文件 sha256 | total |
|---|---|---|---|---|---|
| arrive_dest | 0 | 5000000 | `sha256:03308cd8f05…` | `sha256:b549913b2b7…` | +47.864 |
| arrive_dest | 5 | 5000005 | `sha256:083899da36c…` | `sha256:bbfe5246a3d…` | +51.565 |
| arrive_dest | 12 | 5000012 | `sha256:b6d35c99c7f…` | `sha256:ba926c07590…` | +55.169 |
| arrive_dest | 17 | 5000017 | `sha256:2b82011cfb3…` | `sha256:2fedb9bd51b…` | +55.069 |
| arrive_dest | 29 | 5000029 | `sha256:afde162ba1a…` | `sha256:e5cd2219831…` | +41.806 |
| arrive_dest | 33 | 5000033 | `sha256:52ed52ec07c…` | `sha256:50230f2f5ac…` | +54.794 |
| arrive_dest | 37 | 5000037 | `sha256:5541dfa0306…` | `sha256:cf0b2c8a249…` | +46.493 |
| arrive_dest | 41 | 5000041 | `sha256:aaf5470ed64…` | `sha256:7691b19950f…` | +39.112 |
| arrive_dest | 45 | 5000045 | `sha256:d22c2989e42…` | `sha256:3b34feaf587…` | +47.529 |
| arrive_dest | 50 | 5000050 | `sha256:cd25ef0f42a…` | `sha256:9da9161dd30…` | +47.241 |
| arrive_dest | 54 | 5000054 | `sha256:e6e3acfb403…` | `sha256:65a2afd5ea3…` | +51.237 |
| arrive_dest | 60 | 5000060 | `sha256:080c56f8ea5…` | `sha256:d56ffd2a8a7…` | +50.251 |
| arrive_dest | 62 | 5000062 | `sha256:1984792394a…` | `sha256:57d052d8c79…` | +60.102 |
| arrive_dest | 64 | 5000064 | `sha256:67d3fddf44e…` | `sha256:bf5bd4b0974…` | +44.192 |
| arrive_dest | 69 | 5000069 | `sha256:0d247886834…` | `sha256:d340ed2a937…` | +54.672 |
| arrive_dest | 75 | 5000075 | `sha256:80ac74acb96…` | `sha256:937d1c6d856…` | +50.905 |
| arrive_dest | 80 | 5000080 | `sha256:2c72c843618…` | `sha256:548d6cbbee4…` | +50.905 |
| arrive_dest | 82 | 5000082 | `sha256:50ac1811b1c…` | `sha256:fb086fde786…` | +46.298 |
| arrive_dest | 90 | 5000090 | `sha256:09f0ae804cb…` | `sha256:14edfd04c78…` | +49.592 |
| arrive_dest | 95 | 5000095 | `sha256:4616e48340a…` | `sha256:982d702f093…` | +41.945 |
| arrive_dest | 99 | 5000099 | `sha256:df229e27f9d…` | `sha256:22aebdabe9e…` | +62.226 |
| arrive_dest | 107 | 5000107 | `sha256:934efec2447…` | `sha256:1ef1fe43ef4…` | +41.957 |
| arrive_dest | 111 | 5000111 | `sha256:71401a86f06…` | `sha256:6cde161a0fb…` | +53.360 |
| arrive_dest | 113 | 5000113 | `sha256:a63570dde07…` | `sha256:5ba86fc9f48…` | +53.119 |
| arrive_dest | 116 | 5000116 | `sha256:b3195e9900b…` | `sha256:b1ad205054d…` | +59.704 |
| arrive_dest | 118 | 5000118 | `sha256:98f697c4477…` | `sha256:bf2f49fb8cd…` | +58.899 |
| arrive_dest | 126 | 5000126 | `sha256:c8b38545c2a…` | `sha256:5e66b686e73…` | +46.111 |
| arrive_dest | 129 | 5000129 | `sha256:27c78b47ac2…` | `sha256:c348cf0ddea…` | +55.331 |
| arrive_dest | 138 | 5000138 | `sha256:0d0fb96e579…` | `sha256:a7ce3d86366…` | +43.169 |
| arrive_dest | 141 | 5000141 | `sha256:59fc2b5fb96…` | `sha256:0b0a620c4c4…` | +55.031 |
| arrive_dest | 144 | 5000144 | `sha256:c33a15a2bd5…` | `sha256:84eb0aaf6a6…` | +55.139 |
| arrive_dest | 149 | 5000149 | `sha256:fd75d3ee12b…` | `sha256:20ac725826f…` | +52.295 |
| arrive_dest | 153 | 5000153 | `sha256:41e1519dae2…` | `sha256:0bb85fe5af6…` | +47.482 |
| arrive_dest | 156 | 5000156 | `sha256:f3a8b66842f…` | `sha256:fa2452aa92b…` | +44.568 |
| arrive_dest | 159 | 5000159 | `sha256:e5a66f9cbf9…` | `sha256:9f805cb354a…` | +53.337 |
| arrive_dest | 161 | 5000161 | `sha256:d3c8bd50597…` | `sha256:d7ae898cf6c…` | +55.803 |
| arrive_dest | 168 | 5000168 | `sha256:3525dcad977…` | `sha256:f30685ddd3c…` | +42.390 |
| arrive_dest | 174 | 5000174 | `sha256:6e12308be4b…` | `sha256:9cb8204740c…` | +48.364 |
| arrive_dest | 177 | 5000177 | `sha256:125615c3ac7…` | `sha256:5ffb25aaa7d…` | +46.659 |
| arrive_dest | 185 | 5000185 | `sha256:a2f5987641b…` | `sha256:fc737f54216…` | +53.510 |
| arrive_dest | 188 | 5000188 | `sha256:b69d2853160…` | `sha256:292dfeb66b6…` | +45.170 |
| arrive_dest | 190 | 5000190 | `sha256:468ce9a3cd0…` | `sha256:3d231f97f7c…` | +45.195 |
| arrive_dest | 195 | 5000195 | `sha256:e20904979d7…` | `sha256:78900ec9c0c…` | +51.483 |
| arrive_dest | 198 | 5000198 | `sha256:59433e2903e…` | `sha256:68f8cac8b3c…` | +54.641 |
| arrive_dest | 201 | 5000201 | `sha256:6e107003225…` | `sha256:1551ed5e251…` | +48.307 |
| arrive_dest | 204 | 5000204 | `sha256:5052bc83b35…` | `sha256:9cc3ec63f5e…` | +49.382 |
| arrive_dest | 207 | 5000207 | `sha256:b2abc8a2d9d…` | `sha256:7ced1b29a35…` | +41.603 |
| arrive_dest | 210 | 5000210 | `sha256:10209b96674…` | `sha256:b8946f4a064…` | +55.238 |
| arrive_dest | 213 | 5000213 | `sha256:a749280c130…` | `sha256:793877174af…` | +56.309 |
| arrive_dest | 216 | 5000216 | `sha256:8814de78353…` | `sha256:324c9656325…` | +52.834 |
| arrive_dest | 221 | 5000221 | `sha256:f3a8746f9e0…` | `sha256:48c2915c44a…` | +43.482 |
| arrive_dest | 228 | 5000228 | `sha256:af0d6072eff…` | `sha256:3dadf4584ea…` | +53.620 |
| arrive_dest | 230 | 5000230 | `sha256:b5b01b9a75c…` | `sha256:5503a77b3cc…` | +52.796 |
| arrive_dest | 233 | 5000233 | `sha256:591dfab8a5e…` | `sha256:e54a47fc7ce…` | +46.518 |
| arrive_dest | 237 | 5000237 | `sha256:1dab7ce838f…` | `sha256:c7f6fce2372…` | +48.450 |
| arrive_dest | 244 | 5000244 | `sha256:cbde34bff30…` | `sha256:64c8b0fc503…` | +57.181 |
| arrive_dest | 246 | 5000246 | `sha256:0b6f25355da…` | `sha256:3ae665c08ce…` | +47.714 |
| arrive_dest | 252 | 5000252 | `sha256:e1e21983a7a…` | `sha256:02206e75d41…` | +42.723 |
| arrive_dest | 255 | 5000255 | `sha256:8ca772abb58…` | `sha256:1c3b11029e0…` | +44.145 |
| arrive_dest | 261 | 5000261 | `sha256:7b033042a09…` | `sha256:980ff838008…` | +57.051 |
| arrive_dest | 267 | 5000267 | `sha256:4dc495e65e1…` | `sha256:5e1548a240b…` | +46.969 |
| arrive_dest | 272 | 5000272 | `sha256:5308b40357e…` | `sha256:856e566b897…` | +41.229 |
| arrive_dest | 282 | 5000282 | `sha256:f17dc16e645…` | `sha256:97ce4746961…` | +50.502 |
| arrive_dest | 287 | 5000287 | `sha256:df4904f0ade…` | `sha256:807bfe970ab…` | +47.928 |
| arrive_dest | 290 | 5000290 | `sha256:eaf9fb924cd…` | `sha256:7bb370f9a82…` | +56.503 |
| arrive_dest | 298 | 5000298 | `sha256:028d02a5664…` | `sha256:ac026722589…` | +51.850 |
| arrive_dest | 300 | 5000300 | `sha256:0008537ed6a…` | `sha256:715d2ad7457…` | +50.468 |
| arrive_dest | 303 | 5000303 | `sha256:de585ad9f3c…` | `sha256:52b7751c2c0…` | +43.869 |
| arrive_dest | 306 | 5000306 | `sha256:27ba2df2714…` | `sha256:1e19ed961fc…` | +57.267 |
| arrive_dest | 310 | 5000310 | `sha256:185af60fdef…` | `sha256:94e34448b75…` | +43.040 |
| arrive_dest | 318 | 5000318 | `sha256:c8770b9d91b…` | `sha256:200aa2c5f33…` | +44.383 |
| arrive_dest | 330 | 5000330 | `sha256:d0ce74aba15…` | `sha256:cde71211907…` | +44.982 |
| arrive_dest | 335 | 5000335 | `sha256:731b57aa9d6…` | `sha256:3602934d3dd…` | +53.473 |
| arrive_dest | 341 | 5000341 | `sha256:146f5375c8a…` | `sha256:6d6d01dfb23…` | +54.726 |
| arrive_dest | 345 | 5000345 | `sha256:71c5983d8ab…` | `sha256:ae5b33d2f79…` | +56.086 |
| arrive_dest | 348 | 5000348 | `sha256:6daef9420b8…` | `sha256:3ef76d4f089…` | +43.700 |
| arrive_dest | 354 | 5000354 | `sha256:11dce6f62fc…` | `sha256:c4cd06b6a62…` | +42.069 |
| arrive_dest | 357 | 5000357 | `sha256:a21cf747cd8…` | `sha256:06b68212af5…` | +45.689 |
| arrive_dest | 366 | 5000366 | `sha256:4ec47609771…` | `sha256:8063376f1df…` | +43.392 |
| arrive_dest | 368 | 5000368 | `sha256:41b1e0deab0…` | `sha256:aaebe13932e…` | +52.778 |
| arrive_dest | 375 | 5000375 | `sha256:254f671cc21…` | `sha256:199c5bbc033…` | +54.080 |
| arrive_dest | 383 | 5000383 | `sha256:e17c6899ee5…` | `sha256:57f282b2ac8…` | +45.674 |
| arrive_dest | 387 | 5000387 | `sha256:b4dbff36d63…` | `sha256:de714e0f0f1…` | +43.624 |
| arrive_dest | 392 | 5000392 | `sha256:9d07d0df455…` | `sha256:1e395659f7b…` | +45.923 |
| arrive_dest | 396 | 5000396 | `sha256:d2fd6255ece…` | `sha256:1434f89dc0c…` | +44.517 |
| arrive_dest | 398 | 5000398 | `sha256:3cd28fbad4e…` | `sha256:30c11fb091c…` | +51.239 |
| arrive_dest | 403 | 5000403 | `sha256:2cba218dbea…` | `sha256:b956948c56e…` | +39.276 |
| arrive_dest | 407 | 5000407 | `sha256:43a3f255d8f…` | `sha256:5d5bf9b6359…` | +44.431 |
| arrive_dest | 411 | 5000411 | `sha256:4c812622b9d…` | `sha256:8bdbd93465e…` | +51.988 |
| arrive_dest | 414 | 5000414 | `sha256:e7162d60ae6…` | `sha256:3f468e5e71f…` | +49.043 |
| arrive_dest | 420 | 5000420 | `sha256:68600ded218…` | `sha256:b0baf768b43…` | +54.075 |
| arrive_dest | 423 | 5000423 | `sha256:37273b3e8d5…` | `sha256:b88d9ba9e6d…` | +45.595 |
| arrive_dest | 427 | 5000427 | `sha256:6ba6860bd6d…` | `sha256:f64820e8f9b…` | +60.442 |
| arrive_dest | 432 | 5000432 | `sha256:ea6492ac1e8…` | `sha256:6e91b07bc2f…` | +56.562 |
| arrive_dest | 440 | 5000440 | `sha256:e47a5f8e2e4…` | `sha256:3d929689dd3…` | +45.753 |
| arrive_dest | 443 | 5000443 | `sha256:9d51fb1ba39…` | `sha256:4df6432a78c…` | +51.064 |
| arrive_dest | 453 | 5000453 | `sha256:954967f7cf3…` | `sha256:6cef12de88f…` | +45.886 |
| arrive_dest | 458 | 5000458 | `sha256:5ca53dda1d6…` | `sha256:366725dd436…` | +34.322 |
| arrive_dest | 462 | 5000462 | `sha256:0a7a69d91f0…` | `sha256:8c6cdf653e1…` | +43.525 |
| arrive_dest | 464 | 5000464 | `sha256:12cb5bb40d7…` | `sha256:2b4da748a2c…` | +55.470 |
| arrive_dest | 466 | 5000466 | `sha256:74e725510d5…` | `sha256:85c55e147f9…` | +55.691 |
| arrive_dest | 468 | 5000468 | `sha256:49ed5924743…` | `sha256:d6181e18994…` | +44.199 |
| arrive_dest | 471 | 5000471 | `sha256:d3037d7bbe4…` | `sha256:9ab092cbf5e…` | +51.976 |
| arrive_dest | 479 | 5000479 | `sha256:faf5e4c5403…` | `sha256:daa331a48df…` | +54.263 |
| arrive_dest | 482 | 5000482 | `sha256:fee750de2c8…` | `sha256:f79fa38782c…` | +44.080 |
| arrive_dest | 485 | 5000485 | `sha256:d7fd4f0678b…` | `sha256:70d68630560…` | +52.863 |
| arrive_dest | 491 | 5000491 | `sha256:83b33959407…` | `sha256:b45a8613f9d…` | +41.802 |
| arrive_dest | 495 | 5000495 | `sha256:ce68e7d1784…` | `sha256:016d23206e4…` | +54.406 |
| arrive_dest | 499 | 5000499 | `sha256:d67681d0dd2…` | `sha256:f8de367603c…` | +50.731 |
| arrive_dest | 504 | 5000504 | `sha256:05418cd00c6…` | `sha256:2de2cf51d66…` | +43.872 |
| arrive_dest | 507 | 5000507 | `sha256:65c45084671…` | `sha256:558963893de…` | +44.028 |
| arrive_dest | 516 | 5000516 | `sha256:84a16a3df31…` | `sha256:d0372dfebf2…` | +59.595 |
| arrive_dest | 519 | 5000519 | `sha256:2d3b16ad451…` | `sha256:2d6be0786dc…` | +43.538 |
| arrive_dest | 528 | 5000528 | `sha256:df545619e33…` | `sha256:81c29e18392…` | +44.069 |
| arrive_dest | 537 | 5000537 | `sha256:e263eba81bb…` | `sha256:e9cd10c61e3…` | +55.717 |
| arrive_dest | 543 | 5000543 | `sha256:ae730d4a9f8…` | `sha256:cc79b3fc067…` | +53.072 |
| arrive_dest | 546 | 5000546 | `sha256:3dc7ef5458a…` | `sha256:5e1412fe3a0…` | +55.965 |
| arrive_dest | 552 | 5000552 | `sha256:a3a82e3caab…` | `sha256:d00ab2a8c9c…` | +45.290 |
| arrive_dest | 554 | 5000554 | `sha256:2b324fe0779…` | `sha256:956a6973cda…` | +45.152 |
| arrive_dest | 561 | 5000561 | `sha256:17a51459e21…` | `sha256:f39e13519c9…` | +44.281 |
| arrive_dest | 567 | 5000567 | `sha256:57a04b864fd…` | `sha256:a2867591c0f…` | +43.306 |
| arrive_dest | 570 | 5000570 | `sha256:a095de75ac7…` | `sha256:a4538690612…` | +50.489 |
| arrive_dest | 578 | 5000578 | `sha256:bb1b969804f…` | `sha256:98783046307…` | +42.111 |
| arrive_dest | 587 | 5000587 | `sha256:ea62c0f363f…` | `sha256:7fcc60d1f69…` | +51.722 |
| arrive_dest | 590 | 5000590 | `sha256:e21bc7f9038…` | `sha256:65980c7d65b…` | +45.318 |
| arrive_dest | 594 | 5000594 | `sha256:e12146d3cb7…` | `sha256:3de0e1e363d…` | +48.894 |
| arrive_dest | 597 | 5000597 | `sha256:203edae60d8…` | `sha256:04ae438777a…` | +46.958 |
| arrive_dest | 603 | 5000603 | `sha256:52db6f1dda7…` | `sha256:fe677aa06fd…` | +50.163 |
| arrive_dest | 608 | 5000608 | `sha256:26e4a73011b…` | `sha256:9c6108674d2…` | +30.242 |
| arrive_dest | 610 | 5000610 | `sha256:0ca4485f6b2…` | `sha256:e01479240ae…` | +57.763 |
| arrive_dest | 613 | 5000613 | `sha256:64a732a2c28…` | `sha256:94f7167846f…` | +47.712 |
| arrive_dest | 616 | 5000616 | `sha256:067587453a6…` | `sha256:c392674421c…` | +49.123 |
| arrive_dest | 622 | 5000622 | `sha256:0fe79bea1c9…` | `sha256:3e24689e6c5…` | +50.881 |
| arrive_dest | 628 | 5000628 | `sha256:8a83d4579e4…` | `sha256:af7cc8af158…` | +45.357 |
| arrive_dest | 633 | 5000633 | `sha256:c643eddfce0…` | `sha256:769ad386dc1…` | +42.656 |
| arrive_dest | 642 | 5000642 | `sha256:d306b353604…` | `sha256:dded6b90bd1…` | +43.779 |
| arrive_dest | 647 | 5000647 | `sha256:46abccad467…` | `sha256:670b950ee5b…` | +59.181 |
| arrive_dest | 649 | 5000649 | `sha256:bb652f09a93…` | `sha256:1638cbfe003…` | +53.544 |
| arrive_dest | 654 | 5000654 | `sha256:8d428cfb9e9…` | `sha256:4605c93e0d4…` | +49.410 |
| arrive_dest | 657 | 5000657 | `sha256:f6a5473eacb…` | `sha256:f1bf0e2ba27…` | +44.205 |
| arrive_dest | 662 | 5000662 | `sha256:7177b4f8f9a…` | `sha256:70ea51cb93c…` | +57.531 |
| arrive_dest | 668 | 5000668 | `sha256:97456a9b266…` | `sha256:5bf0f883870…` | +44.455 |
| arrive_dest | 672 | 5000672 | `sha256:a652a59c3e1…` | `sha256:d91e754b6e1…` | +53.681 |
| arrive_dest | 675 | 5000675 | `sha256:f5eabc34a5c…` | `sha256:21781750e7e…` | +45.275 |
| arrive_dest | 681 | 5000681 | `sha256:a31b2a15175…` | `sha256:ae7d4c1954f…` | +45.544 |
| arrive_dest | 684 | 5000684 | `sha256:dfe20c12f97…` | `sha256:b0fd5c05328…` | +51.901 |
| arrive_dest | 689 | 5000689 | `sha256:11ff0f275d7…` | `sha256:df21dd0abca…` | +45.610 |
| arrive_dest | 692 | 5000692 | `sha256:e15a232ae4c…` | `sha256:11249df1996…` | +52.820 |
| arrive_dest | 695 | 5000695 | `sha256:09bc82615e1…` | `sha256:bb29c8a4855…` | +46.969 |
| arrive_dest | 698 | 5000698 | `sha256:faa1a38afce…` | `sha256:0dadb18c35a…` | +48.252 |
| arrive_dest | 701 | 5000701 | `sha256:d9b042b6367…` | `sha256:081ad0b6c2b…` | +50.270 |
| arrive_dest | 703 | 5000703 | `sha256:2d47fce739e…` | `sha256:b200a5c9cc2…` | +50.002 |
| arrive_dest | 705 | 5000705 | `sha256:06a7f1f44b8…` | `sha256:2309a8ec7a6…` | +53.892 |
| arrive_dest | 711 | 5000711 | `sha256:8c4b83c7c02…` | `sha256:230d2a033c6…` | +44.199 |
| arrive_dest | 714 | 5000714 | `sha256:99ffb3c3076…` | `sha256:1dc0f62e0c7…` | +56.381 |
| arrive_dest | 717 | 5000717 | `sha256:a4e93c678bb…` | `sha256:e8255afa532…` | +48.412 |
| arrive_dest | 719 | 5000719 | `sha256:6c779838cfa…` | `sha256:9d9b407ad78…` | +41.818 |
| arrive_dest | 723 | 5000723 | `sha256:38bd099109d…` | `sha256:72dd4008d3c…` | +49.092 |
| arrive_dest | 728 | 5000728 | `sha256:7651c8b809f…` | `sha256:7e33ee819ea…` | +47.402 |
| arrive_dest | 730 | 5000730 | `sha256:32aafa83ac5…` | `sha256:b42ca8d9e7a…` | +57.658 |
| arrive_dest | 737 | 5000737 | `sha256:bf4914c49d6…` | `sha256:828a5e8bf49…` | +52.230 |
| arrive_dest | 739 | 5000739 | `sha256:fce4f7ce979…` | `sha256:ea477d86a95…` | +44.916 |
| arrive_dest | 744 | 5000744 | `sha256:ffb9d213452…` | `sha256:489b5be2cad…` | +42.290 |
| arrive_dest | 747 | 5000747 | `sha256:b902630b76e…` | `sha256:073d0fb70c6…` | +47.786 |
| arrive_dest | 749 | 5000749 | `sha256:b5f749a158a…` | `sha256:1feb684b888…` | +49.993 |
| arrive_dest | 756 | 5000756 | `sha256:662fcdda17e…` | `sha256:62fa9a5e5bf…` | +42.775 |
| arrive_dest | 760 | 5000760 | `sha256:cb0df766ff7…` | `sha256:143e0597e29…` | +44.418 |
| arrive_dest | 766 | 5000766 | `sha256:a3a91d12d80…` | `sha256:723bd15b199…` | +44.653 |
| arrive_dest | 773 | 5000773 | `sha256:8d75485d751…` | `sha256:7319c3e10d5…` | +58.210 |
| arrive_dest | 778 | 5000778 | `sha256:2bb3770a514…` | `sha256:eaa014e4a8b…` | +55.603 |
| arrive_dest | 786 | 5000786 | `sha256:83804c19304…` | `sha256:e36a25ed606…` | +60.897 |
| arrive_dest | 791 | 5000791 | `sha256:fac5cc8e75f…` | `sha256:6f4a1224818…` | +47.288 |
| arrive_dest | 793 | 5000793 | `sha256:6550f5f042e…` | `sha256:59a7b577d1a…` | +50.706 |
| arrive_dest | 795 | 5000795 | `sha256:478981387f2…` | `sha256:9ce6227e62d…` | +44.171 |
| arrive_dest | 797 | 5000797 | `sha256:dd44ed482c8…` | `sha256:dbaeb49a960…` | +62.360 |
| arrive_dest | 800 | 5000800 | `sha256:52fd9f29338…` | `sha256:d2b31038860…` | +45.804 |
| arrive_dest | 804 | 5000804 | `sha256:8010f525837…` | `sha256:beb4eca387c…` | +48.982 |
| arrive_dest | 809 | 5000809 | `sha256:079526eb6cd…` | `sha256:5a27f913b8e…` | +46.720 |
| arrive_dest | 814 | 5000814 | `sha256:c5e3d949603…` | `sha256:78f9ffae8d9…` | +43.500 |
| arrive_dest | 818 | 5000818 | `sha256:4061cad7bf8…` | `sha256:e5c6f9d2c5e…` | +52.781 |
| arrive_dest | 820 | 5000820 | `sha256:6b331c5ed9a…` | `sha256:4617f2575a4…` | +56.544 |
| arrive_dest | 822 | 5000822 | `sha256:b3b31740461…` | `sha256:ca1500a537c…` | +45.852 |
| arrive_dest | 825 | 5000825 | `sha256:9947a14c642…` | `sha256:8076e002389…` | +49.538 |
| arrive_dest | 832 | 5000832 | `sha256:e67fa58e6c0…` | `sha256:4f764dd2d38…` | +52.922 |
| arrive_dest | 837 | 5000837 | `sha256:76b9d002ca7…` | `sha256:96f55628f69…` | +41.274 |
| arrive_dest | 839 | 5000839 | `sha256:ff60b51ed93…` | `sha256:92bbdb9b941…` | +45.275 |
| arrive_dest | 842 | 5000842 | `sha256:f47ceda3c50…` | `sha256:c37f24f9128…` | +49.428 |
| arrive_dest | 846 | 5000846 | `sha256:b84ce86c9d5…` | `sha256:bad22fb7324…` | +44.413 |
| arrive_dest | 855 | 5000855 | `sha256:3f916f4378a…` | `sha256:2120865bea3…` | +44.796 |
| arrive_dest | 861 | 5000861 | `sha256:4e8e4ec7d2b…` | `sha256:d78536af3bf…` | +55.092 |
| arrive_dest | 864 | 5000864 | `sha256:063d66e1e2e…` | `sha256:6e1434c44b7…` | +48.121 |
| arrive_dest | 867 | 5000867 | `sha256:a53c4d1f599…` | `sha256:df2870724bb…` | +48.330 |
| arrive_dest | 870 | 5000870 | `sha256:5279f72eb0e…` | `sha256:f95d5653aed…` | +44.467 |
| arrive_dest | 873 | 5000873 | `sha256:4858f22bf03…` | `sha256:c74c923fbfe…` | +47.507 |
| arrive_dest | 876 | 5000876 | `sha256:9d990928983…` | `sha256:7e5b012de5c…` | +52.632 |
| arrive_dest | 882 | 5000882 | `sha256:8f575f82c38…` | `sha256:8d0ed67b18c…` | +44.036 |
| arrive_dest | 884 | 5000884 | `sha256:e20237d1d6f…` | `sha256:a69ffaec54a…` | +52.481 |
| arrive_dest | 887 | 5000887 | `sha256:1afd52d2ec7…` | `sha256:7c7e5ecf5e8…` | +51.293 |
| arrive_dest | 893 | 5000893 | `sha256:b057e950c20…` | `sha256:52f94b19021…` | +52.588 |
| arrive_dest | 897 | 5000897 | `sha256:c9f2298590f…` | `sha256:050faa194ec…` | +45.068 |
| arrive_dest | 903 | 5000903 | `sha256:8c011ba61f0…` | `sha256:b6d68871747…` | +44.474 |
| arrive_dest | 913 | 5000913 | `sha256:3fbd04c3503…` | `sha256:b94abb8a774…` | +61.480 |
| arrive_dest | 916 | 5000916 | `sha256:89a474a6c1e…` | `sha256:4b29a91e23d…` | +43.017 |
| arrive_dest | 921 | 5000921 | `sha256:dde4beed6b7…` | `sha256:16246b4b27b…` | +44.522 |
| arrive_dest | 927 | 5000927 | `sha256:4850f444cf2…` | `sha256:48dd8b77079…` | +52.119 |
| arrive_dest | 929 | 5000929 | `sha256:34a60c73daf…` | `sha256:9ef9f628dde…` | +49.651 |
| arrive_dest | 933 | 5000933 | `sha256:910588cfc6e…` | `sha256:58d1d7b447c…` | +43.174 |
| arrive_dest | 937 | 5000937 | `sha256:cfa6a576f09…` | `sha256:87feb0c3cd9…` | +50.045 |
| arrive_dest | 945 | 5000945 | `sha256:39bf0df2b22…` | `sha256:d627f7f5594…` | +47.834 |
| arrive_dest | 951 | 5000951 | `sha256:0590ad169f9…` | `sha256:9322fdec44d…` | +47.806 |
| arrive_dest | 955 | 5000955 | `sha256:ecb0cecd7e3…` | `sha256:019c080414c…` | +53.098 |
| arrive_dest | 958 | 5000958 | `sha256:e628f8fc9ad…` | `sha256:630c51057e0…` | +32.821 |
| arrive_dest | 960 | 5000960 | `sha256:a59d8cf8fe8…` | `sha256:1dd79bab21e…` | +43.744 |
| arrive_dest | 963 | 5000963 | `sha256:de197f09eee…` | `sha256:a28a18c3844…` | +44.454 |
| arrive_dest | 966 | 5000966 | `sha256:6ed2afd1344…` | `sha256:bb9aaac385b…` | +57.074 |
| arrive_dest | 974 | 5000974 | `sha256:23a1ea0e617…` | `sha256:f3297f5ecc1…` | +56.498 |
| arrive_dest | 976 | 5000976 | `sha256:684601fe255…` | `sha256:ecc90435742…` | +45.646 |
| arrive_dest | 985 | 5000985 | `sha256:5c71dc8cd15…` | `sha256:2b7abbcf156…` | +54.827 |
| arrive_dest | 992 | 5000992 | `sha256:95853839df8…` | `sha256:09ca984d43c…` | +63.157 |
| arrive_dest | 995 | 5000995 | `sha256:262c1004d1c…` | `sha256:41cdc26032c…` | +46.816 |
| arrive_dest | 999 | 5000999 | `sha256:8c794950f08…` | `sha256:23b57998892…` | +44.632 |
| collision | 20 | 5000020 | `sha256:e8532d50dd4…` | `sha256:aef55ba57e9…` | -13.490 |
| collision | 101 | 5000101 | `sha256:f6ef26f5120…` | `sha256:60c01688a98…` | -33.756 |
| collision | 227 | 5000227 | `sha256:0da80789d65…` | `sha256:846571c85e7…` | -17.202 |
| collision | 293 | 5000293 | `sha256:46ff25788e0…` | `sha256:d2ea0b07fe0…` | -31.589 |
| collision | 320 | 5000320 | `sha256:b22749679f5…` | `sha256:67d790f5be5…` | -26.094 |
| collision | 347 | 5000347 | `sha256:546816caa67…` | `sha256:5a59c1f061b…` | -26.438 |
| collision | 461 | 5000461 | `sha256:01a2f1c573d…` | `sha256:08f5b80aba7…` | -24.673 |
| collision | 503 | 5000503 | `sha256:926f4d2fcca…` | `sha256:940fc0660c9…` | -15.032 |
| collision | 542 | 5000542 | `sha256:c6fbd46b399…` | `sha256:511334a66c2…` | -10.825 |
| collision | 638 | 5000638 | `sha256:f37532dd853…` | `sha256:a6074d1e85a…` | -21.328 |
| collision | 680 | 5000680 | `sha256:6b8d9dcfea5…` | `sha256:543ff491846…` | -26.495 |
| collision | 734 | 5000734 | `sha256:c2cf2e866a6…` | `sha256:08541387813…` | -26.964 |
| collision | 833 | 5000833 | `sha256:f030e1a9f6e…` | `sha256:59a6cea6bf2…` | -17.925 |
| collision | 848 | 5000848 | `sha256:423170f5ba3…` | `sha256:3ff0e85b64c…` | -27.049 |
| collision | 863 | 5000863 | `sha256:60e77a69aac…` | `sha256:9f18634707c…` | -34.324 |
| collision | 911 | 5000911 | `sha256:28fc7280d3d…` | `sha256:c69fb75f826…` | -24.844 |
| collision | 977 | 5000977 | `sha256:65b88fb6fd1…` | `sha256:624304f72d2…` | -24.131 |
| out_of_road | 1 | 5000001 | `sha256:9ece3b619f5…` | `sha256:635ac7deb8e…` | -17.648 |
| out_of_road | 3 | 5000003 | `sha256:3a5774380a9…` | `sha256:cfec949cb11…` | -16.147 |
| out_of_road | 8 | 5000008 | `sha256:3517c46d050…` | `sha256:1d9ea65288e…` | -16.664 |
| out_of_road | 10 | 5000010 | `sha256:603dae7cf0a…` | `sha256:405ab13fe6c…` | -23.323 |
| out_of_road | 13 | 5000013 | `sha256:b95df299fff…` | `sha256:6506b471dab…` | -15.847 |
| out_of_road | 15 | 5000015 | `sha256:f40b6f2f16c…` | `sha256:561703ccc76…` | -12.298 |
| out_of_road | 19 | 5000019 | `sha256:e7018d84f0d…` | `sha256:4f99b7a39a7…` | -23.042 |
| out_of_road | 22 | 5000022 | `sha256:ed02d5538c3…` | `sha256:49448102f99…` | -14.191 |
| out_of_road | 25 | 5000025 | `sha256:d527aa43259…` | `sha256:a4ae85dea07…` | -16.687 |
| out_of_road | 28 | 5000028 | `sha256:585ddece34e…` | `sha256:191b3610ba4…` | -16.405 |
| out_of_road | 34 | 5000034 | `sha256:37594c650f0…` | `sha256:5405f207591…` | -12.621 |
| out_of_road | 38 | 5000038 | `sha256:9141d6aeb31…` | `sha256:1f8f4184144…` | -10.190 |
| out_of_road | 43 | 5000043 | `sha256:dc4ad3664e2…` | `sha256:9ea79d8451f…` | -8.527 |
| out_of_road | 47 | 5000047 | `sha256:e25957ef6e5…` | `sha256:d978a45ffe0…` | -16.306 |
| out_of_road | 52 | 5000052 | `sha256:db1f38890bb…` | `sha256:b0b596b2ba7…` | -16.805 |
| out_of_road | 55 | 5000055 | `sha256:1e4ba10e4fa…` | `sha256:64ebe4155c7…` | -15.141 |
| out_of_road | 58 | 5000058 | `sha256:9354c8c2fec…` | `sha256:fce1e46d24d…` | -16.079 |
| out_of_road | 65 | 5000065 | `sha256:91fdac1cb9e…` | `sha256:7c4532d1b99…` | -16.726 |
| out_of_road | 68 | 5000068 | `sha256:a9cf2340024…` | `sha256:db0a7fa606d…` | -17.317 |
| out_of_road | 71 | 5000071 | `sha256:6b3fd4b50dd…` | `sha256:7dc0ff6c91e…` | -17.311 |
| out_of_road | 73 | 5000073 | `sha256:0f69bc8058d…` | `sha256:0737dab2fbd…` | -12.619 |
| out_of_road | 79 | 5000079 | `sha256:e563e0fa8fe…` | `sha256:42e1648781e…` | -5.506 |
| out_of_road | 84 | 5000084 | `sha256:b45ed0c4337…` | `sha256:00bd2afb170…` | -17.015 |
| out_of_road | 87 | 5000087 | `sha256:60c1d578673…` | `sha256:0fbf3060a62…` | -15.558 |
| out_of_road | 91 | 5000091 | `sha256:d60aa8a4f04…` | `sha256:1f70358f1f5…` | -7.970 |
| out_of_road | 94 | 5000094 | `sha256:957fd9cc72d…` | `sha256:613bd1bc6e8…` | -17.460 |
| out_of_road | 98 | 5000098 | `sha256:657a6f53e99…` | `sha256:b4339c02780…` | -19.722 |
| out_of_road | 102 | 5000102 | `sha256:be08a2ae928…` | `sha256:88be81699fd…` | -15.295 |
| out_of_road | 104 | 5000104 | `sha256:93a11ed60fa…` | `sha256:092c9486300…` | -4.864 |
| out_of_road | 108 | 5000108 | `sha256:317173028f2…` | `sha256:83e1c63ba54…` | -12.213 |
| out_of_road | 114 | 5000114 | `sha256:1c7a931a82d…` | `sha256:f6a082c2a92…` | -12.404 |
| out_of_road | 120 | 5000120 | `sha256:f1d15bc5095…` | `sha256:4469a8bd1bd…` | -12.662 |
| out_of_road | 123 | 5000123 | `sha256:dfa93ed3959…` | `sha256:c4229931dab…` | -11.681 |
| out_of_road | 125 | 5000125 | `sha256:198bcac3ef1…` | `sha256:c34d370911c…` | -4.660 |
| out_of_road | 130 | 5000130 | `sha256:d913c3403e3…` | `sha256:715a93661a7…` | -18.084 |
| out_of_road | 133 | 5000133 | `sha256:688059a6314…` | `sha256:9b703e2ed33…` | -7.733 |
| out_of_road | 136 | 5000136 | `sha256:f432df0a3a8…` | `sha256:ecb8762f6c3…` | -15.192 |
| out_of_road | 142 | 5000142 | `sha256:f4792548192…` | `sha256:242509c5d65…` | -23.740 |
| out_of_road | 147 | 5000147 | `sha256:657348f0d70…` | `sha256:636134d90de…` | -11.192 |
| out_of_road | 151 | 5000151 | `sha256:86d13e175e8…` | `sha256:3e665956ee4…` | -16.869 |
| out_of_road | 154 | 5000154 | `sha256:2fe310a85db…` | `sha256:e55bbd79056…` | -16.174 |
| out_of_road | 162 | 5000162 | `sha256:7b09fc311d6…` | `sha256:bec8b216d66…` | -11.196 |
| out_of_road | 164 | 5000164 | `sha256:b3117978568…` | `sha256:d4f7cda3a8c…` | -28.267 |
| out_of_road | 167 | 5000167 | `sha256:7445ef4683d…` | `sha256:1ca6f37e74a…` | -19.326 |
| out_of_road | 170 | 5000170 | `sha256:d9ff77e2ed3…` | `sha256:fbb3ae655ff…` | -20.622 |
| out_of_road | 173 | 5000173 | `sha256:ae60444628a…` | `sha256:b183ffbbb5b…` | -20.215 |
| out_of_road | 178 | 5000178 | `sha256:4126f3fb115…` | `sha256:0d8d3913bef…` | -16.845 |
| out_of_road | 181 | 5000181 | `sha256:078a0cef073…` | `sha256:cb90ff13b82…` | -10.092 |
| out_of_road | 183 | 5000183 | `sha256:22fb5d9a2e4…` | `sha256:ce51d0859c1…` | -17.950 |
| out_of_road | 187 | 5000187 | `sha256:1ab94ea80fd…` | `sha256:54692c78473…` | -16.960 |
| out_of_road | 193 | 5000193 | `sha256:de33c5d98e3…` | `sha256:af99645204b…` | -5.591 |
| out_of_road | 196 | 5000196 | `sha256:fd8cf8627fc…` | `sha256:74a3292e16a…` | -14.120 |
| out_of_road | 202 | 5000202 | `sha256:e4e768897a7…` | `sha256:f32acf7ca34…` | -11.294 |
| out_of_road | 208 | 5000208 | `sha256:b125bef5015…` | `sha256:6de2a92ca7a…` | -18.761 |
| out_of_road | 214 | 5000214 | `sha256:0eb4b10c065…` | `sha256:7ad66173bcb…` | -18.093 |
| out_of_road | 218 | 5000218 | `sha256:fff5435b100…` | `sha256:63f232f6ebd…` | -19.131 |
| out_of_road | 223 | 5000223 | `sha256:c76fb805b18…` | `sha256:a03cce4009d…` | -15.197 |
| out_of_road | 226 | 5000226 | `sha256:7edc58efda7…` | `sha256:57394243be0…` | -19.311 |
| out_of_road | 235 | 5000235 | `sha256:1b98337d3ea…` | `sha256:c0c41805563…` | -15.889 |
| out_of_road | 239 | 5000239 | `sha256:46cb93ee6fb…` | `sha256:259f61eefa1…` | -30.760 |
| out_of_road | 243 | 5000243 | `sha256:a595b5ededd…` | `sha256:27c69deac72…` | -10.086 |
| out_of_road | 248 | 5000248 | `sha256:27fe4a9461b…` | `sha256:e2186b80830…` | -24.662 |
| out_of_road | 251 | 5000251 | `sha256:585e116b6e9…` | `sha256:2d2a6efd2a5…` | -21.041 |
| out_of_road | 256 | 5000256 | `sha256:2ad0a667dbc…` | `sha256:8b08a769ba6…` | -18.257 |
| out_of_road | 259 | 5000259 | `sha256:f982db5ded9…` | `sha256:b706eadb16d…` | -17.948 |
| out_of_road | 262 | 5000262 | `sha256:bf9548d66ea…` | `sha256:49dd1f04883…` | -17.579 |
| out_of_road | 265 | 5000265 | `sha256:3121f89835e…` | `sha256:9bb851a535d…` | -23.667 |
| out_of_road | 268 | 5000268 | `sha256:920f9c9d893…` | `sha256:d023b5c9f6f…` | -16.474 |
| out_of_road | 270 | 5000270 | `sha256:9f25f641d3c…` | `sha256:7cfd485ac22…` | -20.055 |
| out_of_road | 275 | 5000275 | `sha256:8abdf01fefe…` | `sha256:16d7da3077d…` | -3.489 |
| out_of_road | 277 | 5000277 | `sha256:0ba4e2c2db7…` | `sha256:e048edb3376…` | -17.177 |
| out_of_road | 279 | 5000279 | `sha256:8f382e5fc5d…` | `sha256:034337f760b…` | -10.402 |
| out_of_road | 281 | 5000281 | `sha256:ba84d5c83e4…` | `sha256:4c09858f47b…` | -19.635 |
| out_of_road | 285 | 5000285 | `sha256:15ebc466d94…` | `sha256:e7eafc8b2a1…` | -16.660 |
| out_of_road | 291 | 5000291 | `sha256:fd284d36933…` | `sha256:67492264da0…` | -17.919 |
| out_of_road | 295 | 5000295 | `sha256:b5fc29a9acd…` | `sha256:042880e9ad4…` | -16.073 |
| out_of_road | 297 | 5000297 | `sha256:7fa894884a1…` | `sha256:dab507b3cfa…` | -11.289 |
| out_of_road | 304 | 5000304 | `sha256:a5702fddafa…` | `sha256:4ef9fa28530…` | -7.454 |
| out_of_road | 309 | 5000309 | `sha256:97c8c1ab81c…` | `sha256:bd1ebdfb39a…` | -17.122 |
| out_of_road | 313 | 5000313 | `sha256:de46900e3f0…` | `sha256:b111c8c3bc0…` | -15.311 |
| out_of_road | 316 | 5000316 | `sha256:e9dc3b6612c…` | `sha256:59e6456b968…` | -16.087 |
| out_of_road | 319 | 5000319 | `sha256:c1348dd270f…` | `sha256:779dcd0ee5b…` | -13.841 |
| out_of_road | 323 | 5000323 | `sha256:5bc50f68eb6…` | `sha256:305e91d0d9b…` | -30.385 |
| out_of_road | 325 | 5000325 | `sha256:73d8441cee1…` | `sha256:f9191ae037f…` | -13.746 |
| out_of_road | 327 | 5000327 | `sha256:0624330493e…` | `sha256:55dbbe31a81…` | -15.259 |
| out_of_road | 329 | 5000329 | `sha256:c51dcf14138…` | `sha256:32140f3cc2c…` | -21.287 |
| out_of_road | 332 | 5000332 | `sha256:85c0cd498b3…` | `sha256:43fe5808ab7…` | -20.887 |
| out_of_road | 336 | 5000336 | `sha256:9e518ee151b…` | `sha256:143487a6cca…` | -15.621 |
| out_of_road | 340 | 5000340 | `sha256:b27fd80b2ae…` | `sha256:a41fb1ba76a…` | -16.431 |
| out_of_road | 349 | 5000349 | `sha256:fe13b81449a…` | `sha256:96f16930de0…` | -15.152 |
| out_of_road | 352 | 5000352 | `sha256:aa3e8d85dda…` | `sha256:fe709313b17…` | -6.675 |
| out_of_road | 355 | 5000355 | `sha256:317d6a123fa…` | `sha256:52377ce6818…` | -22.998 |
| out_of_road | 360 | 5000360 | `sha256:894bc972aa9…` | `sha256:3cbe9dd1fa5…` | -12.868 |
| out_of_road | 363 | 5000363 | `sha256:f7b63a40ff1…` | `sha256:2ba49891dfb…` | -15.894 |
| out_of_road | 365 | 5000365 | `sha256:1f7ece102f4…` | `sha256:74580d09241…` | -5.428 |
| out_of_road | 371 | 5000371 | `sha256:5d13befad46…` | `sha256:8ecabc9a294…` | -16.620 |
| out_of_road | 373 | 5000373 | `sha256:540a4d327d5…` | `sha256:bf7abdc1d57…` | -17.465 |
| out_of_road | 376 | 5000376 | `sha256:dedffc7e228…` | `sha256:778c9d7e654…` | -17.244 |
| out_of_road | 379 | 5000379 | `sha256:6ffda1e525b…` | `sha256:796d529e832…` | -16.289 |
| out_of_road | 381 | 5000381 | `sha256:1cbf291d441…` | `sha256:6663d5c1d31…` | -15.149 |
| out_of_road | 385 | 5000385 | `sha256:93d72cb361c…` | `sha256:514763b8e3e…` | -16.714 |
| out_of_road | 388 | 5000388 | `sha256:fe28eb0814c…` | `sha256:65fc29e19f5…` | -16.632 |
| out_of_road | 391 | 5000391 | `sha256:c5214dfd68b…` | `sha256:97ddfc47290…` | -19.319 |
| out_of_road | 395 | 5000395 | `sha256:fde9513c871…` | `sha256:0367c47ad0b…` | -6.709 |
| out_of_road | 401 | 5000401 | `sha256:b6e7b9ace54…` | `sha256:3b732dc37e6…` | -17.718 |
| out_of_road | 404 | 5000404 | `sha256:78285c47a98…` | `sha256:d9cede13ac5…` | -15.600 |
| out_of_road | 409 | 5000409 | `sha256:50b6890d227…` | `sha256:0a2c61b1b28…` | -10.938 |
| out_of_road | 415 | 5000415 | `sha256:88073797a08…` | `sha256:2fa2f9a98b1…` | -17.852 |
| out_of_road | 419 | 5000419 | `sha256:293862b9712…` | `sha256:6566e85366c…` | -17.990 |
| out_of_road | 424 | 5000424 | `sha256:7ae0f1e6165…` | `sha256:2de649f4a18…` | -25.409 |
| out_of_road | 429 | 5000429 | `sha256:65d150ce958…` | `sha256:d5d99ffc95a…` | -15.920 |
| out_of_road | 434 | 5000434 | `sha256:f3451950abd…` | `sha256:3751c9d6921…` | -20.573 |
| out_of_road | 437 | 5000437 | `sha256:9191b60d8a2…` | `sha256:9a730a4192a…` | -5.270 |
| out_of_road | 439 | 5000439 | `sha256:7532c1aa205…` | `sha256:727a9d88b87…` | -19.005 |
| out_of_road | 444 | 5000444 | `sha256:3981961e5a6…` | `sha256:c5e56c72fb1…` | -12.116 |
| out_of_road | 446 | 5000446 | `sha256:8d7f3ffa4d4…` | `sha256:a633d17af07…` | -19.118 |
| out_of_road | 450 | 5000450 | `sha256:0d96f89c7c0…` | `sha256:9ce1a0b1204…` | -17.159 |
| out_of_road | 452 | 5000452 | `sha256:1fc2b0c0b3b…` | `sha256:4ef76577c73…` | -19.177 |
| out_of_road | 455 | 5000455 | `sha256:dfbbed2ced6…` | `sha256:77a1806efa8…` | -9.556 |
| out_of_road | 460 | 5000460 | `sha256:ceed6f336f7…` | `sha256:6099ed12cf3…` | -10.601 |
| out_of_road | 473 | 5000473 | `sha256:53a43768bb7…` | `sha256:f3a75e8d9a6…` | -19.979 |
| out_of_road | 475 | 5000475 | `sha256:d1031e81981…` | `sha256:976f8ace549…` | -17.505 |
| out_of_road | 478 | 5000478 | `sha256:2887e5d4b51…` | `sha256:c62428e4e81…` | -20.493 |
| out_of_road | 484 | 5000484 | `sha256:fdfe8d450fe…` | `sha256:81ed1c3248f…` | -12.792 |
| out_of_road | 487 | 5000487 | `sha256:73c71b2786f…` | `sha256:ec46478c29e…` | -13.375 |
| out_of_road | 489 | 5000489 | `sha256:f45d7242df9…` | `sha256:b95c86844cb…` | -13.920 |
| out_of_road | 494 | 5000494 | `sha256:f9f1bdcdb54…` | `sha256:9c35a259174…` | -19.895 |
| out_of_road | 498 | 5000498 | `sha256:f63d75d9596…` | `sha256:ca52c332189…` | -15.552 |
| out_of_road | 505 | 5000505 | `sha256:0546d6f294a…` | `sha256:d735ba19353…` | -16.351 |
| out_of_road | 509 | 5000509 | `sha256:dcc53266e9c…` | `sha256:5577c62327a…` | -17.054 |
| out_of_road | 512 | 5000512 | `sha256:46e23f22738…` | `sha256:10a2127bb9d…` | -10.617 |
| out_of_road | 514 | 5000514 | `sha256:7a5daeca552…` | `sha256:69854059e3b…` | -22.533 |
| out_of_road | 518 | 5000518 | `sha256:2a928fc98f2…` | `sha256:7d325b0ea20…` | -8.414 |
| out_of_road | 523 | 5000523 | `sha256:9d5c160ee2d…` | `sha256:e5c7bad35d0…` | -19.203 |
| out_of_road | 525 | 5000525 | `sha256:9f10a671e3e…` | `sha256:fc0c45fbcc1…` | -16.402 |
| out_of_road | 527 | 5000527 | `sha256:935bbc0851e…` | `sha256:c95b84725cf…` | -20.249 |
| out_of_road | 530 | 5000530 | `sha256:fab81244d61…` | `sha256:77ec65dfa1a…` | -15.849 |
| out_of_road | 533 | 5000533 | `sha256:b7d17914081…` | `sha256:1222e30e4e3…` | -17.566 |
| out_of_road | 535 | 5000535 | `sha256:4e6bb0d3976…` | `sha256:f64a5615bc3…` | -7.888 |
| out_of_road | 541 | 5000541 | `sha256:4b4608a6d7f…` | `sha256:d4b0ba7c776…` | -16.339 |
| out_of_road | 547 | 5000547 | `sha256:bee68592ac4…` | `sha256:8c56368d507…` | -8.687 |
| out_of_road | 550 | 5000550 | `sha256:afaae883f15…` | `sha256:dd2b70a6e3a…` | -23.833 |
| out_of_road | 555 | 5000555 | `sha256:512260cfd30…` | `sha256:359a5cf898b…` | -16.195 |
| out_of_road | 557 | 5000557 | `sha256:2a96e59e4a1…` | `sha256:4e480e8bbce…` | -16.987 |
| out_of_road | 560 | 5000560 | `sha256:f4f6bcfce16…` | `sha256:649cabac67e…` | -9.554 |
| out_of_road | 565 | 5000565 | `sha256:dcd63194208…` | `sha256:65cebaaf2e2…` | -16.901 |
| out_of_road | 569 | 5000569 | `sha256:57af6c13b11…` | `sha256:94026c4e128…` | -16.733 |
| out_of_road | 574 | 5000574 | `sha256:09e581c3326…` | `sha256:4d9a3574c76…` | -15.722 |
| out_of_road | 576 | 5000576 | `sha256:e931a115cc0…` | `sha256:930e1218c00…` | -10.186 |
| out_of_road | 580 | 5000580 | `sha256:ec403a5f464…` | `sha256:0fcef5e5e82…` | -21.285 |
| out_of_road | 582 | 5000582 | `sha256:beb30996755…` | `sha256:de49071a89f…` | -18.329 |
| out_of_road | 585 | 5000585 | `sha256:064b5515530…` | `sha256:657ae97ba0d…` | -15.888 |
| out_of_road | 589 | 5000589 | `sha256:e8e42e830b3…` | `sha256:cddcc77e354…` | -12.686 |
| out_of_road | 593 | 5000593 | `sha256:b5c0d8962ae…` | `sha256:f09df94c31f…` | -16.910 |
| out_of_road | 600 | 5000600 | `sha256:d8357cfd2c1…` | `sha256:dd717a1333e…` | -16.314 |
| out_of_road | 604 | 5000604 | `sha256:3ad05cf0f01…` | `sha256:932cac0cd39…` | -17.550 |
| out_of_road | 607 | 5000607 | `sha256:aba7963c22d…` | `sha256:dc4f93329c1…` | -14.034 |
| out_of_road | 619 | 5000619 | `sha256:f4740c22f29…` | `sha256:b2d9e2faed1…` | -8.421 |
| out_of_road | 624 | 5000624 | `sha256:11f0cccca5b…` | `sha256:8455b04dcf9…` | -17.084 |
| out_of_road | 626 | 5000626 | `sha256:fbe1272df50…` | `sha256:b342223b945…` | -20.435 |
| out_of_road | 630 | 5000630 | `sha256:f736a1aebc3…` | `sha256:6d2598806ae…` | -16.892 |
| out_of_road | 634 | 5000634 | `sha256:79a6057c02f…` | `sha256:b5779c40c3d…` | -14.077 |
| out_of_road | 636 | 5000636 | `sha256:532905a8871…` | `sha256:bbe6cb93433…` | -14.452 |
| out_of_road | 640 | 5000640 | `sha256:5b766efc6dd…` | `sha256:559af343251…` | -4.822 |
| out_of_road | 645 | 5000645 | `sha256:78bf3c0e253…` | `sha256:6bd4d1500bd…` | -13.243 |
| out_of_road | 650 | 5000650 | `sha256:40212adb74a…` | `sha256:66eceff2528…` | -15.669 |
| out_of_road | 653 | 5000653 | `sha256:8b5927ee471…` | `sha256:95181f4d09b…` | -19.073 |
| out_of_road | 658 | 5000658 | `sha256:5543fb41a01…` | `sha256:3c741f91a36…` | -13.647 |
| out_of_road | 663 | 5000663 | `sha256:1323159d7fe…` | `sha256:7c2059fa1db…` | -11.524 |
| out_of_road | 667 | 5000667 | `sha256:8ab564a0012…` | `sha256:cf52619f5d1…` | -13.759 |
| out_of_road | 670 | 5000670 | `sha256:e3fc9fc7658…` | `sha256:5a21a852296…` | -6.925 |
| out_of_road | 676 | 5000676 | `sha256:3fafd8472f7…` | `sha256:0801c632ab6…` | -17.872 |
| out_of_road | 679 | 5000679 | `sha256:5a2bc246e0b…` | `sha256:a7f6744754e…` | -18.099 |
| out_of_road | 686 | 5000686 | `sha256:88b8df568c6…` | `sha256:028aeb3dc23…` | -11.499 |
| out_of_road | 688 | 5000688 | `sha256:6efd84def74…` | `sha256:6be5088376a…` | -14.092 |
| out_of_road | 694 | 5000694 | `sha256:a4126e0473a…` | `sha256:2c0ce655209…` | -6.074 |
| out_of_road | 700 | 5000700 | `sha256:e48f60ed8e5…` | `sha256:7994f2f4d1d…` | -16.246 |
| out_of_road | 707 | 5000707 | `sha256:d1c7c0bba1e…` | `sha256:8d383254772…` | -16.715 |
| out_of_road | 710 | 5000710 | `sha256:187ad483250…` | `sha256:9a366c46db7…` | -19.083 |
| out_of_road | 716 | 5000716 | `sha256:e05ef2261c1…` | `sha256:8053d033759…` | -1.758 |
| out_of_road | 724 | 5000724 | `sha256:a366939667e…` | `sha256:0c44a1dd990…` | -16.275 |
| out_of_road | 732 | 5000732 | `sha256:1b24a86bc5c…` | `sha256:b801ebbac04…` | -19.229 |
| out_of_road | 735 | 5000735 | `sha256:03876d50583…` | `sha256:b1a61d0dca1…` | -16.090 |
| out_of_road | 742 | 5000742 | `sha256:aa226c02492…` | `sha256:7467d2058f2…` | -14.303 |
| out_of_road | 746 | 5000746 | `sha256:3bbc5e70777…` | `sha256:b0b2e943183…` | -19.659 |
| out_of_road | 752 | 5000752 | `sha256:88d3d6e8b09…` | `sha256:daf74a7c496…` | -19.470 |
| out_of_road | 754 | 5000754 | `sha256:eeb67259eee…` | `sha256:5622ae43952…` | -23.560 |
| out_of_road | 759 | 5000759 | `sha256:fee1fcc4bf9…` | `sha256:cd400d3c97d…` | -14.235 |
| out_of_road | 762 | 5000762 | `sha256:f1e06673790…` | `sha256:e2983e119b6…` | -18.649 |
| out_of_road | 767 | 5000767 | `sha256:a0922535373…` | `sha256:1f07a8df7af…` | -17.876 |
| out_of_road | 770 | 5000770 | `sha256:7d19201d526…` | `sha256:1e0bfe1a851…` | -18.436 |
| out_of_road | 772 | 5000772 | `sha256:6c2a47dd70b…` | `sha256:0b30d39cbb1…` | -14.170 |
| out_of_road | 775 | 5000775 | `sha256:9512b4eb30d…` | `sha256:60266508d20…` | -16.283 |
| out_of_road | 779 | 5000779 | `sha256:9e6bc8fb902…` | `sha256:96cf92d83c5…` | -18.870 |
| out_of_road | 781 | 5000781 | `sha256:78c476e57f8…` | `sha256:6bce382bb81…` | -23.790 |
| out_of_road | 785 | 5000785 | `sha256:2e0b9d93194…` | `sha256:e8b5cf40d60…` | -21.889 |
| out_of_road | 788 | 5000788 | `sha256:7e38282c924…` | `sha256:3f7292d0918…` | -23.158 |
| out_of_road | 798 | 5000798 | `sha256:55f36f690dd…` | `sha256:944acc89708…` | -18.094 |
| out_of_road | 803 | 5000803 | `sha256:52e4a72caf5…` | `sha256:98db0765489…` | -20.846 |
| out_of_road | 806 | 5000806 | `sha256:50dac2df3a1…` | `sha256:c7694c083c7…` | -20.533 |
| out_of_road | 810 | 5000810 | `sha256:eb706396db0…` | `sha256:6637693d0a2…` | -14.493 |
| out_of_road | 812 | 5000812 | `sha256:1c43b92d06a…` | `sha256:fda8f16061a…` | -17.487 |
| out_of_road | 817 | 5000817 | `sha256:d6ab30b6592…` | `sha256:8b784ce5af9…` | -0.029 |
| out_of_road | 826 | 5000826 | `sha256:ff5b814a7ee…` | `sha256:1e5f2734663…` | -12.883 |
| out_of_road | 829 | 5000829 | `sha256:ec896145b1a…` | `sha256:2c353a1e412…` | -23.068 |
| out_of_road | 831 | 5000831 | `sha256:5a94215f3ab…` | `sha256:c1d389d6165…` | -13.084 |
| out_of_road | 841 | 5000841 | `sha256:a5ac432007f…` | `sha256:59eda865655…` | -8.174 |
| out_of_road | 845 | 5000845 | `sha256:7d256360823…` | `sha256:68c471cb9c7…` | -23.379 |
| out_of_road | 850 | 5000850 | `sha256:2c99f09cac6…` | `sha256:c60839e8152…` | -15.318 |
| out_of_road | 852 | 5000852 | `sha256:462bb972d5f…` | `sha256:046b1870e77…` | -18.561 |
| out_of_road | 856 | 5000856 | `sha256:3984241e3d6…` | `sha256:de589448e62…` | -16.081 |
| out_of_road | 860 | 5000860 | `sha256:cec58be7208…` | `sha256:ecd58a5cb90…` | -18.643 |
| out_of_road | 868 | 5000868 | `sha256:6be3c35191b…` | `sha256:1e4ab1bdf15…` | -9.463 |
| out_of_road | 874 | 5000874 | `sha256:c251954f482…` | `sha256:085458924cc…` | -16.090 |
| out_of_road | 879 | 5000879 | `sha256:ea88661bed4…` | `sha256:db588008abd…` | -12.371 |
| out_of_road | 886 | 5000886 | `sha256:0ee378520c1…` | `sha256:38296a15e2a…` | -13.240 |
| out_of_road | 890 | 5000890 | `sha256:31c760d490c…` | `sha256:65d833193e0…` | -31.008 |
| out_of_road | 892 | 5000892 | `sha256:336a5ee8238…` | `sha256:c8a4d873906…` | -16.697 |
| out_of_road | 896 | 5000896 | `sha256:8c31e12b09a…` | `sha256:30bd16c0e19…` | -20.844 |
| out_of_road | 899 | 5000899 | `sha256:f6c0547e58d…` | `sha256:49522ba6e8f…` | +2.823 |
| out_of_road | 904 | 5000904 | `sha256:df688f5ecaf…` | `sha256:bddddf0097a…` | -13.482 |
| out_of_road | 906 | 5000906 | `sha256:1610af889d5…` | `sha256:2e3e169c106…` | -16.271 |
| out_of_road | 908 | 5000908 | `sha256:2691dc9d4c3…` | `sha256:c7970d74195…` | -10.552 |
| out_of_road | 910 | 5000910 | `sha256:c6819d17ef2…` | `sha256:5f97f7a2043…` | -15.595 |
| out_of_road | 917 | 5000917 | `sha256:515c586bdc2…` | `sha256:e6475cbf248…` | -22.303 |
| out_of_road | 919 | 5000919 | `sha256:54876382e47…` | `sha256:8a4b4576a64…` | -14.816 |
| out_of_road | 924 | 5000924 | `sha256:46fffc3a734…` | `sha256:368e68da7a2…` | -11.141 |
| out_of_road | 926 | 5000926 | `sha256:9f7a9cfc4cd…` | `sha256:19254e8d292…` | -5.429 |
| out_of_road | 932 | 5000932 | `sha256:18d3bc2be15…` | `sha256:f4c98610503…` | -19.195 |
| out_of_road | 935 | 5000935 | `sha256:995ffcb738c…` | `sha256:844006f138f…` | -20.077 |
| out_of_road | 939 | 5000939 | `sha256:a1a11cfe4db…` | `sha256:1c32eb16780…` | -13.218 |
| out_of_road | 941 | 5000941 | `sha256:6b1c86ec872…` | `sha256:554136fd61a…` | -4.754 |
| out_of_road | 943 | 5000943 | `sha256:670a31c43df…` | `sha256:d3d37ba80fa…` | -16.347 |
| out_of_road | 948 | 5000948 | `sha256:bc2f45f618d…` | `sha256:2e73887cb55…` | -16.813 |
| out_of_road | 950 | 5000950 | `sha256:0dd5ffff9c6…` | `sha256:ace0a63fa84…` | -19.307 |
| out_of_road | 953 | 5000953 | `sha256:50e05a4c6e4…` | `sha256:3a2cb12515c…` | -19.434 |
| out_of_road | 962 | 5000962 | `sha256:bf849868625…` | `sha256:7d99f032877…` | -19.058 |
| out_of_road | 969 | 5000969 | `sha256:2a669ab2cf5…` | `sha256:5f0cb1ba2d9…` | -12.684 |
| out_of_road | 971 | 5000971 | `sha256:700562905c8…` | `sha256:f6806502a14…` | -14.755 |
| out_of_road | 979 | 5000979 | `sha256:cfb2b342da9…` | `sha256:58e20b47d3d…` | -12.471 |
| out_of_road | 981 | 5000981 | `sha256:0198f1a5ece…` | `sha256:39294e48e46…` | -16.510 |
| out_of_road | 984 | 5000984 | `sha256:87dc73ce339…` | `sha256:ea467e3704e…` | -16.995 |
| out_of_road | 987 | 5000987 | `sha256:a61a6f44200…` | `sha256:6c488f14b68…` | -12.286 |
| out_of_road | 991 | 5000991 | `sha256:7078aea164f…` | `sha256:f0a7d7a1b8c…` | -15.986 |
| out_of_road | 997 | 5000997 | `sha256:b4d4b0876ed…` | `sha256:dcbeb968401…` | -15.871 |
| max_step | 23 | 5000023 | `sha256:236225a3f24…` | `sha256:cc9b33d5969…` | -15.948 |
| max_step | 44 | 5000044 | `sha256:e0d043d6f5a…` | `sha256:42da2eaabdc…` | -11.137 |
| max_step | 134 | 5000134 | `sha256:3de7a11e111…` | `sha256:54eb68e429b…` | -5.322 |
| max_step | 236 | 5000236 | `sha256:0b95f76918a…` | `sha256:c1dfb22af56…` | -8.597 |
| max_step | 338 | 5000338 | `sha256:74d92ec650f…` | `sha256:f858474230d…` | -6.020 |
| max_step | 416 | 5000416 | `sha256:fb6d54c948d…` | `sha256:8899122db5e…` | -10.073 |
| max_step | 431 | 5000431 | `sha256:e344cbbd7d8…` | `sha256:b4a6a74b08f…` | -16.094 |
| max_step | 500 | 5000500 | `sha256:518db7f4539…` | `sha256:1e448d1ad3b…` | -14.037 |
| max_step | 539 | 5000539 | `sha256:6b592d956e4…` | `sha256:eeb27877a30…` | -0.890 |
| max_step | 572 | 5000572 | `sha256:cae4debb4f5…` | `sha256:4510c7b8205…` | -4.933 |
| max_step | 596 | 5000596 | `sha256:048147a3ae0…` | `sha256:0fc60811a04…` | -20.532 |
| max_step | 617 | 5000617 | `sha256:81c93c90193…` | `sha256:8cf19e532f8…` | -8.339 |
| max_step | 623 | 5000623 | `sha256:4c72c6d547b…` | `sha256:066f9134009…` | -6.166 |
| max_step | 659 | 5000659 | `sha256:1e504d187aa…` | `sha256:b76eea1c3bc…` | -7.790 |
| max_step | 731 | 5000731 | `sha256:e3da3b8273b…` | `sha256:aa5eb5805dd…` | -11.189 |
| max_step | 764 | 5000764 | `sha256:ff89afbf6ac…` | `sha256:153d6e2e4f3…` | -9.893 |
| max_step | 902 | 5000902 | `sha256:ad8097d7b63…` | `sha256:ef81ed2f23e…` | -17.989 |
| max_step | 983 | 5000983 | `sha256:6bbc13f5e63…` | `sha256:3bcdb92a49d…` | -29.108 |

### solve（499 条）

| 类 | id | seed | ctx_sha256 | 文件 sha256 | total |
|---|---|---|---|---|---|
| arrive_dest | 4 | 5000004 | `sha256:da624933706…` | `sha256:9dee8367ac6…` | +43.804 |
| arrive_dest | 6 | 5000006 | `sha256:2e3c3ba0c25…` | `sha256:e5c0ec8e00b…` | +53.355 |
| arrive_dest | 16 | 5000016 | `sha256:0eb5f4eb06d…` | `sha256:4d00e6a6247…` | +46.626 |
| arrive_dest | 27 | 5000027 | `sha256:35860c834d9…` | `sha256:1245ba6476b…` | +46.382 |
| arrive_dest | 30 | 5000030 | `sha256:a7c91b5fc50…` | `sha256:464ba79ee13…` | +43.635 |
| arrive_dest | 35 | 5000035 | `sha256:475dadbce7e…` | `sha256:1390d522d9b…` | +39.022 |
| arrive_dest | 39 | 5000039 | `sha256:930c63c7d47…` | `sha256:7bff2c87f88…` | +56.634 |
| arrive_dest | 42 | 5000042 | `sha256:f596ea82233…` | `sha256:464a24cbb9c…` | +47.816 |
| arrive_dest | 48 | 5000048 | `sha256:395c59b1484…` | `sha256:f3e9edabd77…` | +57.514 |
| arrive_dest | 51 | 5000051 | `sha256:f87af5777b5…` | `sha256:517de3b1ce9…` | +43.334 |
| arrive_dest | 57 | 5000057 | `sha256:1bd6709e461…` | `sha256:6e632b0fdf4…` | +49.150 |
| arrive_dest | 61 | 5000061 | `sha256:0ef9417ceda…` | `sha256:b6e70031a54…` | +43.287 |
| arrive_dest | 63 | 5000063 | `sha256:58eaa3339ba…` | `sha256:e0493daa717…` | +53.099 |
| arrive_dest | 66 | 5000066 | `sha256:844a1025b06…` | `sha256:8f0fd20ddd0…` | +50.951 |
| arrive_dest | 74 | 5000074 | `sha256:bdc2b259bdb…` | `sha256:922795b1a17…` | +51.799 |
| arrive_dest | 78 | 5000078 | `sha256:fb499dc5496…` | `sha256:57088a43753…` | +49.314 |
| arrive_dest | 81 | 5000081 | `sha256:4f396a810cd…` | `sha256:3cb26dcd9b6…` | +45.435 |
| arrive_dest | 86 | 5000086 | `sha256:bc9797b113c…` | `sha256:b96fce405b1…` | +41.016 |
| arrive_dest | 92 | 5000092 | `sha256:b7c31cd0d90…` | `sha256:cc7e3f4b0d3…` | +51.744 |
| arrive_dest | 96 | 5000096 | `sha256:0e99857122f…` | `sha256:539663d8050…` | +54.283 |
| arrive_dest | 105 | 5000105 | `sha256:d0dcd8f1554…` | `sha256:02a58b9a353…` | +55.426 |
| arrive_dest | 110 | 5000110 | `sha256:200c41427b9…` | `sha256:4aafe002794…` | +42.521 |
| arrive_dest | 112 | 5000112 | `sha256:eb6a1aedf70…` | `sha256:f5ab339c79c…` | +43.132 |
| arrive_dest | 115 | 5000115 | `sha256:dfc1f805c4c…` | `sha256:476f636c674…` | +56.567 |
| arrive_dest | 117 | 5000117 | `sha256:d30afc9c921…` | `sha256:d8b7b4de399…` | +48.362 |
| arrive_dest | 121 | 5000121 | `sha256:f58fe3fe209…` | `sha256:8d54b2adde3…` | +46.307 |
| arrive_dest | 127 | 5000127 | `sha256:853ee3a1ca2…` | `sha256:8cea74d17b7…` | +45.959 |
| arrive_dest | 137 | 5000137 | `sha256:cfcc7c80b34…` | `sha256:656b9bbc0ef…` | +53.135 |
| arrive_dest | 140 | 5000140 | `sha256:c9029cdd43a…` | `sha256:4cb78ebd80b…` | +54.018 |
| arrive_dest | 143 | 5000143 | `sha256:710c76fa312…` | `sha256:005e9034488…` | +50.387 |
| arrive_dest | 145 | 5000145 | `sha256:71b14e7d2f4…` | `sha256:431f9f91037…` | +43.306 |
| arrive_dest | 150 | 5000150 | `sha256:9c9346930bd…` | `sha256:f9db03f3586…` | +48.430 |
| arrive_dest | 155 | 5000155 | `sha256:878350c18c9…` | `sha256:3f8599f7f72…` | +48.844 |
| arrive_dest | 157 | 5000157 | `sha256:71953db28aa…` | `sha256:a18c5eccae4…` | +60.479 |
| arrive_dest | 160 | 5000160 | `sha256:4bb68c75aae…` | `sha256:ce329f3c9cc…` | +55.992 |
| arrive_dest | 165 | 5000165 | `sha256:26af568fad3…` | `sha256:887b255d4f8…` | +54.161 |
| arrive_dest | 171 | 5000171 | `sha256:a68b9fd9c95…` | `sha256:f4c7805e94b…` | +51.639 |
| arrive_dest | 176 | 5000176 | `sha256:d8af67d318e…` | `sha256:a780c235e36…` | +45.565 |
| arrive_dest | 179 | 5000179 | `sha256:8ef699c9eae…` | `sha256:19a32dec45e…` | +42.584 |
| arrive_dest | 186 | 5000186 | `sha256:cba273c5a63…` | `sha256:fb1b1a4d7a3…` | +43.063 |
| arrive_dest | 189 | 5000189 | `sha256:b0b27fac225…` | `sha256:b9328b3fe9c…` | +41.678 |
| arrive_dest | 192 | 5000192 | `sha256:225dd0fa956…` | `sha256:82eb82b8720…` | +44.091 |
| arrive_dest | 197 | 5000197 | `sha256:d18b89a06a7…` | `sha256:288d11b046a…` | +54.113 |
| arrive_dest | 200 | 5000200 | `sha256:ec6f6d97508…` | `sha256:07c7752b81a…` | +40.449 |
| arrive_dest | 203 | 5000203 | `sha256:5d92b73d746…` | `sha256:8d8ad81fb3e…` | +44.111 |
| arrive_dest | 206 | 5000206 | `sha256:17406d91f30…` | `sha256:e85027b0f5d…` | +53.561 |
| arrive_dest | 209 | 5000209 | `sha256:a0541146d50…` | `sha256:47017245f99…` | +52.797 |
| arrive_dest | 211 | 5000211 | `sha256:1fe1d886ea8…` | `sha256:c70b7bb28f0…` | +44.750 |
| arrive_dest | 215 | 5000215 | `sha256:920eef48d14…` | `sha256:1721386072b…` | +42.556 |
| arrive_dest | 219 | 5000219 | `sha256:a1d1170f280…` | `sha256:a9a1bd34441…` | +54.331 |
| arrive_dest | 222 | 5000222 | `sha256:b656ae5cf98…` | `sha256:e85455f3523…` | +54.051 |
| arrive_dest | 229 | 5000229 | `sha256:4a3ace80a2f…` | `sha256:2e7016849e0…` | +63.136 |
| arrive_dest | 231 | 5000231 | `sha256:372ae73a629…` | `sha256:d60d32d6e55…` | +50.542 |
| arrive_dest | 234 | 5000234 | `sha256:96025a7b8ce…` | `sha256:e2d134b5666…` | +47.114 |
| arrive_dest | 240 | 5000240 | `sha256:85315103d41…` | `sha256:c7835f10f56…` | +43.193 |
| arrive_dest | 245 | 5000245 | `sha256:ec614d84492…` | `sha256:67a976534df…` | +43.445 |
| arrive_dest | 249 | 5000249 | `sha256:b9bce4cd26e…` | `sha256:80833b93aa6…` | +44.770 |
| arrive_dest | 253 | 5000253 | `sha256:7e35475bc10…` | `sha256:85bb4b78793…` | +49.239 |
| arrive_dest | 257 | 5000257 | `sha256:aed6fa23b34…` | `sha256:c3179ccbdf7…` | +48.582 |
| arrive_dest | 264 | 5000264 | `sha256:e40527e295b…` | `sha256:6dc963e85c0…` | +44.242 |
| arrive_dest | 271 | 5000271 | `sha256:f220a801669…` | `sha256:06570e95c5b…` | +53.595 |
| arrive_dest | 274 | 5000274 | `sha256:434fbcbf03a…` | `sha256:a39c7d6d460…` | +52.607 |
| arrive_dest | 286 | 5000286 | `sha256:892caefa650…` | `sha256:6963216e2dd…` | +48.210 |
| arrive_dest | 288 | 5000288 | `sha256:95e32d9a612…` | `sha256:578703d2e42…` | +46.878 |
| arrive_dest | 294 | 5000294 | `sha256:0049494b2a1…` | `sha256:dc89b0ae5fe…` | +47.120 |
| arrive_dest | 299 | 5000299 | `sha256:affa07da5bc…` | `sha256:4a32f679c51…` | +42.070 |
| arrive_dest | 302 | 5000302 | `sha256:e77f8568229…` | `sha256:bde9853b8f1…` | +46.501 |
| arrive_dest | 305 | 5000305 | `sha256:a80190025f1…` | `sha256:341ea311964…` | +52.722 |
| arrive_dest | 308 | 5000308 | `sha256:610e3ac26c6…` | `sha256:d7e57994891…` | +50.059 |
| arrive_dest | 315 | 5000315 | `sha256:9c1578e3291…` | `sha256:0bb6748d0e2…` | +44.964 |
| arrive_dest | 321 | 5000321 | `sha256:728c8cc2b7e…` | `sha256:d1cbee2417a…` | +48.361 |
| arrive_dest | 334 | 5000334 | `sha256:362a3e956ff…` | `sha256:20d648ef93d…` | +48.069 |
| arrive_dest | 339 | 5000339 | `sha256:0e91038ea1c…` | `sha256:1a633749e8d…` | +55.378 |
| arrive_dest | 342 | 5000342 | `sha256:cc08fd358a5…` | `sha256:89739429603…` | +43.011 |
| arrive_dest | 346 | 5000346 | `sha256:83c4f62da2e…` | `sha256:7d36cd36c0b…` | +43.486 |
| arrive_dest | 351 | 5000351 | `sha256:c69f0a1c9f9…` | `sha256:da1f4c1e2a3…` | +45.297 |
| arrive_dest | 356 | 5000356 | `sha256:f092cd4ecba…` | `sha256:2eac1c55aa9…` | +54.235 |
| arrive_dest | 362 | 5000362 | `sha256:acec6fe70c9…` | `sha256:eaa5c79e62b…` | +54.662 |
| arrive_dest | 367 | 5000367 | `sha256:a6638d023fa…` | `sha256:ac02ff75f43…` | +46.902 |
| arrive_dest | 369 | 5000369 | `sha256:5dca97ae003…` | `sha256:20ad701ab8e…` | +63.119 |
| arrive_dest | 378 | 5000378 | `sha256:d2d3a7575b8…` | `sha256:59625b37179…` | +42.774 |
| arrive_dest | 384 | 5000384 | `sha256:b0272ad0b40…` | `sha256:9dc04c6d32d…` | +53.478 |
| arrive_dest | 390 | 5000390 | `sha256:7e337f79246…` | `sha256:056a27d796e…` | +44.027 |
| arrive_dest | 393 | 5000393 | `sha256:00207e91970…` | `sha256:a43c076514f…` | +48.021 |
| arrive_dest | 397 | 5000397 | `sha256:38825f1a9e2…` | `sha256:81ced80231c…` | +65.259 |
| arrive_dest | 400 | 5000400 | `sha256:d30f1e89b5e…` | `sha256:eb4dabcda05…` | +56.190 |
| arrive_dest | 405 | 5000405 | `sha256:535226768b0…` | `sha256:f970551af0d…` | +57.174 |
| arrive_dest | 408 | 5000408 | `sha256:36bfc86458b…` | `sha256:230ba850bff…` | +50.470 |
| arrive_dest | 412 | 5000412 | `sha256:4b3cb19e3e3…` | `sha256:454237fd37b…` | +69.468 |
| arrive_dest | 417 | 5000417 | `sha256:4a4093cfc1e…` | `sha256:01b3550ebc8…` | +44.292 |
| arrive_dest | 422 | 5000422 | `sha256:09b0aa201e5…` | `sha256:5d7feabbe2d…` | +49.661 |
| arrive_dest | 426 | 5000426 | `sha256:82bdf08ac4a…` | `sha256:6e76a902e8f…` | +50.954 |
| arrive_dest | 430 | 5000430 | `sha256:a2a3d41c4d0…` | `sha256:ad8a60d3355…` | +45.213 |
| arrive_dest | 435 | 5000435 | `sha256:205105a5690…` | `sha256:0cb5c540e60…` | +46.486 |
| arrive_dest | 441 | 5000441 | `sha256:82192728419…` | `sha256:6ef051fdcbf…` | +49.086 |
| arrive_dest | 447 | 5000447 | `sha256:9bb58b8e809…` | `sha256:5a0c0144d52…` | +43.469 |
| arrive_dest | 456 | 5000456 | `sha256:dcef02e5a2f…` | `sha256:3b231ea356d…` | +57.783 |
| arrive_dest | 459 | 5000459 | `sha256:0d2726f4196…` | `sha256:3f5fdd14b66…` | +55.154 |
| arrive_dest | 463 | 5000463 | `sha256:3465c435486…` | `sha256:5f862cb36a0…` | +51.672 |
| arrive_dest | 465 | 5000465 | `sha256:021dfcf9770…` | `sha256:1a347831ba9…` | +54.906 |
| arrive_dest | 467 | 5000467 | `sha256:9b0ce166c9e…` | `sha256:afd012a5e4c…` | +41.384 |
| arrive_dest | 470 | 5000470 | `sha256:4db737661c2…` | `sha256:20d0e3e6e89…` | +39.890 |
| arrive_dest | 472 | 5000472 | `sha256:acfcac243c3…` | `sha256:bcb3f9a225c…` | +56.275 |
| arrive_dest | 480 | 5000480 | `sha256:7ff67624ff6…` | `sha256:f51f95367e5…` | +47.581 |
| arrive_dest | 483 | 5000483 | `sha256:924fb73a862…` | `sha256:7d9787a9f2e…` | +48.256 |
| arrive_dest | 490 | 5000490 | `sha256:058e7a9f781…` | `sha256:2e468e8102b…` | +42.814 |
| arrive_dest | 492 | 5000492 | `sha256:6a22e5d77b3…` | `sha256:f9e3963f3a4…` | +51.488 |
| arrive_dest | 496 | 5000496 | `sha256:0f803b14910…` | `sha256:06d22dd98d3…` | +43.460 |
| arrive_dest | 501 | 5000501 | `sha256:95f166ec93b…` | `sha256:85c8197ba0c…` | +44.119 |
| arrive_dest | 506 | 5000506 | `sha256:3705faea83f…` | `sha256:32d8a2bd2a0…` | +41.021 |
| arrive_dest | 510 | 5000510 | `sha256:6a2080e3081…` | `sha256:1403929f721…` | +44.082 |
| arrive_dest | 517 | 5000517 | `sha256:4213cb21928…` | `sha256:756defeb8da…` | +43.395 |
| arrive_dest | 522 | 5000522 | `sha256:59df0fc4a08…` | `sha256:2ffabed1ba6…` | +53.972 |
| arrive_dest | 531 | 5000531 | `sha256:6f45b9f76ce…` | `sha256:d359cca4262…` | +57.869 |
| arrive_dest | 540 | 5000540 | `sha256:9f011b34ee0…` | `sha256:107621a9499…` | +46.298 |
| arrive_dest | 545 | 5000545 | `sha256:3bdac362f25…` | `sha256:c101bfcc4a2…` | +48.191 |
| arrive_dest | 549 | 5000549 | `sha256:3c4be9da306…` | `sha256:f553d639ed8…` | +43.913 |
| arrive_dest | 553 | 5000553 | `sha256:9f608cea58c…` | `sha256:611e62bdef9…` | +57.464 |
| arrive_dest | 558 | 5000558 | `sha256:13eb8ca3c96…` | `sha256:134ed120ad3…` | +47.976 |
| arrive_dest | 564 | 5000564 | `sha256:cc97a50eed8…` | `sha256:2ef309e8791…` | +48.424 |
| arrive_dest | 568 | 5000568 | `sha256:49b889ad997…` | `sha256:64c8ce9f9a0…` | +44.879 |
| arrive_dest | 573 | 5000573 | `sha256:8c5a3e7ff8e…` | `sha256:b555bf39842…` | +48.519 |
| arrive_dest | 579 | 5000579 | `sha256:fa0c57601d6…` | `sha256:3c218c7a18d…` | +44.375 |
| arrive_dest | 588 | 5000588 | `sha256:06136a5681b…` | `sha256:1dfb98d36d5…` | +53.380 |
| arrive_dest | 591 | 5000591 | `sha256:de359bd40a1…` | `sha256:841ad02ff90…` | +43.687 |
| arrive_dest | 595 | 5000595 | `sha256:04a7186695c…` | `sha256:41bcebb98ed…` | +51.525 |
| arrive_dest | 602 | 5000602 | `sha256:b944eb458d1…` | `sha256:3e4fe9a870e…` | +57.354 |
| arrive_dest | 606 | 5000606 | `sha256:ec9a9790880…` | `sha256:03d026974b7…` | +55.406 |
| arrive_dest | 609 | 5000609 | `sha256:f1257d21b36…` | `sha256:cfa596bb83f…` | +58.198 |
| arrive_dest | 612 | 5000612 | `sha256:f768dbacb9d…` | `sha256:eb601c2adea…` | +52.668 |
| arrive_dest | 615 | 5000615 | `sha256:c6190e4edca…` | `sha256:bda6791cddc…` | +66.049 |
| arrive_dest | 618 | 5000618 | `sha256:ebc545a1d26…` | `sha256:f5b2d407685…` | +47.952 |
| arrive_dest | 627 | 5000627 | `sha256:eded00bd1d7…` | `sha256:c285eb62fe5…` | +52.738 |
| arrive_dest | 632 | 5000632 | `sha256:85c99f7c46c…` | `sha256:477446178e4…` | +55.199 |
| arrive_dest | 639 | 5000639 | `sha256:cf58e107d77…` | `sha256:8e19cdd3c24…` | +49.761 |
| arrive_dest | 643 | 5000643 | `sha256:36c0526d869…` | `sha256:e58225649de…` | +48.836 |
| arrive_dest | 648 | 5000648 | `sha256:6fc1aa29a87…` | `sha256:59a50303ea4…` | +44.312 |
| arrive_dest | 651 | 5000651 | `sha256:02fb55b31d9…` | `sha256:c6a4b06b749…` | +44.571 |
| arrive_dest | 656 | 5000656 | `sha256:4e6807cd7b2…` | `sha256:f29a91bd898…` | +44.661 |
| arrive_dest | 660 | 5000660 | `sha256:94d4b290e3e…` | `sha256:28273e5f654…` | +54.758 |
| arrive_dest | 666 | 5000666 | `sha256:21463b7ddbb…` | `sha256:83130de4dc7…` | +47.958 |
| arrive_dest | 671 | 5000671 | `sha256:c653479e787…` | `sha256:decfce54613…` | +51.020 |
| arrive_dest | 674 | 5000674 | `sha256:4156cb28450…` | `sha256:1d8f80f599f…` | +47.189 |
| arrive_dest | 678 | 5000678 | `sha256:1130cd10108…` | `sha256:b9a91869e53…` | +49.393 |
| arrive_dest | 682 | 5000682 | `sha256:98fa9dee0e1…` | `sha256:b80423c83dc…` | +42.160 |
| arrive_dest | 685 | 5000685 | `sha256:1494635df66…` | `sha256:35cfb7b72a8…` | +59.016 |
| arrive_dest | 690 | 5000690 | `sha256:400b00d95d6…` | `sha256:8be0b3ba570…` | +50.286 |
| arrive_dest | 693 | 5000693 | `sha256:7a944dd8bf9…` | `sha256:362f9124d3b…` | +53.861 |
| arrive_dest | 696 | 5000696 | `sha256:93e26450b69…` | `sha256:79cf8fb4479…` | +55.394 |
| arrive_dest | 699 | 5000699 | `sha256:8d935a8f84a…` | `sha256:5493c3bcc8d…` | +50.579 |
| arrive_dest | 702 | 5000702 | `sha256:c72c7ecab51…` | `sha256:14988cf6fbd…` | +42.991 |
| arrive_dest | 704 | 5000704 | `sha256:0ca6ea8a27c…` | `sha256:450492bd0c5…` | +39.330 |
| arrive_dest | 708 | 5000708 | `sha256:de7ed327ab9…` | `sha256:6e078155cc0…` | +54.099 |
| arrive_dest | 713 | 5000713 | `sha256:19d92d37493…` | `sha256:e46585a6a55…` | +51.228 |
| arrive_dest | 715 | 5000715 | `sha256:2dfee988965…` | `sha256:c984e237561…` | +58.984 |
| arrive_dest | 718 | 5000718 | `sha256:8fa7100d492…` | `sha256:5ad5a8bd5bd…` | +44.282 |
| arrive_dest | 720 | 5000720 | `sha256:73852a03612…` | `sha256:588fe870184…` | +43.653 |
| arrive_dest | 726 | 5000726 | `sha256:72b425b1731…` | `sha256:e9effb31de9…` | +53.330 |
| arrive_dest | 729 | 5000729 | `sha256:759b0515ef5…` | `sha256:10728d462ba…` | +45.002 |
| arrive_dest | 736 | 5000736 | `sha256:babb3849684…` | `sha256:4a02f2ad5fd…` | +44.209 |
| arrive_dest | 738 | 5000738 | `sha256:921192fc4d5…` | `sha256:cce0e58e5c8…` | +44.097 |
| arrive_dest | 740 | 5000740 | `sha256:5f719f1f844…` | `sha256:453c38e82ee…` | +51.972 |
| arrive_dest | 745 | 5000745 | `sha256:2fa325b6125…` | `sha256:41304516a9b…` | +48.812 |
| arrive_dest | 748 | 5000748 | `sha256:e42b7e5e1a3…` | `sha256:de2bdfea7d2…` | +55.900 |
| arrive_dest | 750 | 5000750 | `sha256:37863610e13…` | `sha256:5c7efb8d3b1…` | +49.026 |
| arrive_dest | 757 | 5000757 | `sha256:afbb331d2a7…` | `sha256:c01f831c620…` | +63.304 |
| arrive_dest | 765 | 5000765 | `sha256:0958bebc0fa…` | `sha256:b9a8d180c0e…` | +48.625 |
| arrive_dest | 768 | 5000768 | `sha256:07aa83dbcc3…` | `sha256:39bdaa76b84…` | +44.017 |
| arrive_dest | 776 | 5000776 | `sha256:b9a5ad5ef32…` | `sha256:24c16c7ced8…` | +41.695 |
| arrive_dest | 783 | 5000783 | `sha256:c9cac9ff2b0…` | `sha256:d7f41429ac4…` | +57.103 |
| arrive_dest | 789 | 5000789 | `sha256:dcff55a473f…` | `sha256:3f00a5c677b…` | +49.432 |
| arrive_dest | 792 | 5000792 | `sha256:3dd21b722d0…` | `sha256:f2e3662e19d…` | +44.705 |
| arrive_dest | 794 | 5000794 | `sha256:184fc366abf…` | `sha256:7bf4595afda…` | +50.549 |
| arrive_dest | 796 | 5000796 | `sha256:8dca308a387…` | `sha256:59c08a7923e…` | +46.001 |
| arrive_dest | 799 | 5000799 | `sha256:fd730682fcc…` | `sha256:752a5a889a8…` | +49.244 |
| arrive_dest | 801 | 5000801 | `sha256:cdf7f5f88cf…` | `sha256:3b15287c737…` | +59.751 |
| arrive_dest | 807 | 5000807 | `sha256:d39a64873c5…` | `sha256:c521b95b31a…` | +42.969 |
| arrive_dest | 813 | 5000813 | `sha256:7d12e480664…` | `sha256:32463e90c63…` | +44.611 |
| arrive_dest | 815 | 5000815 | `sha256:53df65386c1…` | `sha256:40390fefec6…` | +40.969 |
| arrive_dest | 819 | 5000819 | `sha256:5e89486ce77…` | `sha256:7696d736386…` | +44.726 |
| arrive_dest | 821 | 5000821 | `sha256:fde91ce2d9b…` | `sha256:dfaeb87cbf8…` | +49.246 |
| arrive_dest | 823 | 5000823 | `sha256:a82aae7f2ff…` | `sha256:ef55b817280…` | +51.178 |
| arrive_dest | 828 | 5000828 | `sha256:2fc8e24e778…` | `sha256:c89e3a79725…` | +48.014 |
| arrive_dest | 834 | 5000834 | `sha256:954b7d51f19…` | `sha256:26522143bdd…` | +54.788 |
| arrive_dest | 838 | 5000838 | `sha256:f6adbc2b165…` | `sha256:66bac6b2bf0…` | +45.687 |
| arrive_dest | 840 | 5000840 | `sha256:68ca83909c0…` | `sha256:0d750af63b0…` | +53.307 |
| arrive_dest | 843 | 5000843 | `sha256:8107feb28d8…` | `sha256:ac2443d9dd5…` | +54.424 |
| arrive_dest | 849 | 5000849 | `sha256:87a013623db…` | `sha256:84af82f2d67…` | +53.016 |
| arrive_dest | 858 | 5000858 | `sha256:1e62677f5cb…` | `sha256:ac181e706ff…` | +51.950 |
| arrive_dest | 862 | 5000862 | `sha256:cb6af000723…` | `sha256:701bcadc9df…` | +55.722 |
| arrive_dest | 865 | 5000865 | `sha256:e1b8b4e33fb…` | `sha256:ba89aefba7f…` | +53.238 |
| arrive_dest | 869 | 5000869 | `sha256:be93a4faaba…` | `sha256:27cdac0b619…` | +28.897 |
| arrive_dest | 872 | 5000872 | `sha256:cdd8d1daa92…` | `sha256:bf1d5f984b8…` | +50.991 |
| arrive_dest | 875 | 5000875 | `sha256:8e0e5bdec75…` | `sha256:c08b7065139…` | +59.903 |
| arrive_dest | 880 | 5000880 | `sha256:2d84eaf5885…` | `sha256:7e737b0cb1d…` | +59.381 |
| arrive_dest | 883 | 5000883 | `sha256:d59d53be251…` | `sha256:397204d9cc2…` | +48.950 |
| arrive_dest | 885 | 5000885 | `sha256:e7f9175a252…` | `sha256:4076effd477…` | +52.609 |
| arrive_dest | 888 | 5000888 | `sha256:548551fc52e…` | `sha256:44f86cd01de…` | +45.442 |
| arrive_dest | 894 | 5000894 | `sha256:572e7feeed0…` | `sha256:c0c311fd39d…` | +54.986 |
| arrive_dest | 900 | 5000900 | `sha256:a71eb62e4f4…` | `sha256:4222b34f362…` | +54.130 |
| arrive_dest | 912 | 5000912 | `sha256:8126bf2b5c5…` | `sha256:8c9cc9b7ef8…` | +54.516 |
| arrive_dest | 915 | 5000915 | `sha256:853904f724f…` | `sha256:932e68dbe2e…` | +48.644 |
| arrive_dest | 920 | 5000920 | `sha256:50875545e53…` | `sha256:879adf4e776…` | +30.697 |
| arrive_dest | 923 | 5000923 | `sha256:acc9397a8ce…` | `sha256:ffce51d3734…` | +47.201 |
| arrive_dest | 928 | 5000928 | `sha256:528556913af…` | `sha256:883085102f3…` | +56.399 |
| arrive_dest | 931 | 5000931 | `sha256:1da00776219…` | `sha256:d26d9080525…` | +54.501 |
| arrive_dest | 936 | 5000936 | `sha256:c64a96ccc25…` | `sha256:2ee24c637a8…` | +55.458 |
| arrive_dest | 944 | 5000944 | `sha256:7b07cb6ebfc…` | `sha256:29b4228d33f…` | +53.210 |
| arrive_dest | 946 | 5000946 | `sha256:28bdb6cc5fa…` | `sha256:295cdb97940…` | +56.831 |
| arrive_dest | 954 | 5000954 | `sha256:6e1ec623ccb…` | `sha256:882c1db12ca…` | +53.759 |
| arrive_dest | 957 | 5000957 | `sha256:608edc34083…` | `sha256:d16497e0451…` | +43.299 |
| arrive_dest | 959 | 5000959 | `sha256:31100cf72aa…` | `sha256:17fd0646e04…` | +51.269 |
| arrive_dest | 961 | 5000961 | `sha256:3b949b51bc9…` | `sha256:e7edb8f4fcc…` | +59.884 |
| arrive_dest | 964 | 5000964 | `sha256:f1c019f2a5d…` | `sha256:28a26f7df1a…` | +41.594 |
| arrive_dest | 972 | 5000972 | `sha256:5d0ca620e31…` | `sha256:9589ce9faa4…` | +50.972 |
| arrive_dest | 975 | 5000975 | `sha256:24750a1c3b2…` | `sha256:00d0a9a0840…` | +50.797 |
| arrive_dest | 978 | 5000978 | `sha256:1393b80a666…` | `sha256:a5af7059933…` | +53.690 |
| arrive_dest | 990 | 5000990 | `sha256:fc49a98a304…` | `sha256:622198d0617…` | +54.629 |
| arrive_dest | 994 | 5000994 | `sha256:96df17dec93…` | `sha256:c979bb633d5…` | +43.982 |
| arrive_dest | 996 | 5000996 | `sha256:a1327ec01bd…` | `sha256:6442bfeae36…` | +52.996 |
| collision | 77 | 5000077 | `sha256:240504dd950…` | `sha256:76a853a77dd…` | -26.080 |
| collision | 131 | 5000131 | `sha256:7ee8ecc5404…` | `sha256:3ac9eca8142…` | -29.012 |
| collision | 284 | 5000284 | `sha256:1583e136e6b…` | `sha256:7f28323973a…` | -20.834 |
| collision | 311 | 5000311 | `sha256:7136331fd35…` | `sha256:a3afc7af29f…` | -23.614 |
| collision | 344 | 5000344 | `sha256:f1d16cb2a1b…` | `sha256:4814f7585fc…` | -24.341 |
| collision | 410 | 5000410 | `sha256:250be4b52d2…` | `sha256:b408adcb5ba…` | -14.020 |
| collision | 476 | 5000476 | `sha256:5db3bc731b0…` | `sha256:7e980fc1ad8…` | -25.932 |
| collision | 536 | 5000536 | `sha256:c3dd588eadd…` | `sha256:e5591216ed0…` | -17.629 |
| collision | 614 | 5000614 | `sha256:2d0697aaa10…` | `sha256:c811f8b132d…` | -10.862 |
| collision | 665 | 5000665 | `sha256:2c6dd5abcba…` | `sha256:ea2c614868e…` | -23.051 |
| collision | 725 | 5000725 | `sha256:755e3ee41a7…` | `sha256:d48859b418c…` | -13.015 |
| collision | 782 | 5000782 | `sha256:20f3d1ac9ca…` | `sha256:99741ec727a…` | -14.964 |
| collision | 836 | 5000836 | `sha256:07300b15383…` | `sha256:5d2b2a2de82…` | -15.305 |
| collision | 854 | 5000854 | `sha256:684f1b3fd94…` | `sha256:80252f068ad…` | -25.888 |
| collision | 878 | 5000878 | `sha256:076fd704bc9…` | `sha256:3781f2378c5…` | -12.261 |
| collision | 967 | 5000967 | `sha256:ee7b6cd8c14…` | `sha256:1602c143b72…` | -17.832 |
| collision | 989 | 5000989 | `sha256:93ed1c69941…` | `sha256:8130b59351f…` | -27.983 |
| out_of_road | 2 | 5000002 | `sha256:efffcbd37ed…` | `sha256:c7578ff2eba…` | +3.354 |
| out_of_road | 7 | 5000007 | `sha256:83d00113b8f…` | `sha256:399104262e9…` | -6.314 |
| out_of_road | 9 | 5000009 | `sha256:364284c8496…` | `sha256:bcf25d3431b…` | -14.096 |
| out_of_road | 11 | 5000011 | `sha256:5330a6ce590…` | `sha256:cabb99ca497…` | -18.125 |
| out_of_road | 14 | 5000014 | `sha256:ac30c98acb0…` | `sha256:d97db4ca4c8…` | -12.807 |
| out_of_road | 18 | 5000018 | `sha256:80e17ea1467…` | `sha256:d9cf2ae1393…` | -17.689 |
| out_of_road | 21 | 5000021 | `sha256:61f6d20f08d…` | `sha256:20825352523…` | -15.995 |
| out_of_road | 24 | 5000024 | `sha256:5291e8073e9…` | `sha256:60b3e9205c9…` | -15.629 |
| out_of_road | 26 | 5000026 | `sha256:3aa84530de9…` | `sha256:f87621abd54…` | -15.572 |
| out_of_road | 31 | 5000031 | `sha256:10d2b7b4130…` | `sha256:f4f8f9ba114…` | -22.375 |
| out_of_road | 36 | 5000036 | `sha256:ac35b4e5312…` | `sha256:0a2d45aa930…` | -17.525 |
| out_of_road | 40 | 5000040 | `sha256:ecd2fc4d9d4…` | `sha256:60ead84503a…` | -15.176 |
| out_of_road | 46 | 5000046 | `sha256:6926aab6276…` | `sha256:989251b7528…` | -17.172 |
| out_of_road | 49 | 5000049 | `sha256:f3489f55a78…` | `sha256:e78cc90f9ca…` | -17.112 |
| out_of_road | 53 | 5000053 | `sha256:31e332c2359…` | `sha256:b78386cdc84…` | -15.992 |
| out_of_road | 56 | 5000056 | `sha256:e8425472128…` | `sha256:029f0643179…` | -17.266 |
| out_of_road | 59 | 5000059 | `sha256:556295524ca…` | `sha256:85ebd6b44ae…` | -19.053 |
| out_of_road | 67 | 5000067 | `sha256:6c468108ab6…` | `sha256:ea8679d7b18…` | -16.151 |
| out_of_road | 70 | 5000070 | `sha256:605dd8fccab…` | `sha256:028c41b4948…` | -16.534 |
| out_of_road | 72 | 5000072 | `sha256:cabcec0078d…` | `sha256:1739902cca0…` | -13.955 |
| out_of_road | 76 | 5000076 | `sha256:c0ea85dd10b…` | `sha256:a943514a5be…` | -13.402 |
| out_of_road | 83 | 5000083 | `sha256:0079b195cf5…` | `sha256:38e69d3ff8f…` | -20.218 |
| out_of_road | 85 | 5000085 | `sha256:1f85060abda…` | `sha256:e2b1d41b684…` | -15.706 |
| out_of_road | 88 | 5000088 | `sha256:6a28f10d7c1…` | `sha256:1332be428d5…` | -17.864 |
| out_of_road | 93 | 5000093 | `sha256:9e305135ec2…` | `sha256:4d1eb3d9043…` | -16.441 |
| out_of_road | 97 | 5000097 | `sha256:95841e9379b…` | `sha256:04c2ef6d98f…` | -15.485 |
| out_of_road | 100 | 5000100 | `sha256:12fa2102904…` | `sha256:1ab83cb95ec…` | -18.172 |
| out_of_road | 103 | 5000103 | `sha256:7ceb2ef90b1…` | `sha256:296a0d39567…` | -10.722 |
| out_of_road | 106 | 5000106 | `sha256:c78b75d0aca…` | `sha256:bf0fa6311ad…` | -16.275 |
| out_of_road | 109 | 5000109 | `sha256:eeb64a5e57a…` | `sha256:5646009511f…` | -15.672 |
| out_of_road | 119 | 5000119 | `sha256:64be7277133…` | `sha256:f13f8e12a0a…` | -21.021 |
| out_of_road | 122 | 5000122 | `sha256:9221eb372e9…` | `sha256:0da497c8fa9…` | -22.825 |
| out_of_road | 124 | 5000124 | `sha256:561c884a175…` | `sha256:638e196a8e7…` | -14.633 |
| out_of_road | 128 | 5000128 | `sha256:5a1c3fb0c8c…` | `sha256:f6a83501d89…` | -21.322 |
| out_of_road | 132 | 5000132 | `sha256:4e943df57d1…` | `sha256:87346363e69…` | -12.874 |
| out_of_road | 135 | 5000135 | `sha256:5288fef47a8…` | `sha256:74913b9ae20…` | -11.720 |
| out_of_road | 139 | 5000139 | `sha256:31d92bb6fd0…` | `sha256:adda4d7d844…` | -24.212 |
| out_of_road | 146 | 5000146 | `sha256:de868167352…` | `sha256:cbb96a624d8…` | -16.480 |
| out_of_road | 148 | 5000148 | `sha256:7b5eee6d8ac…` | `sha256:f0f9a0e3bdf…` | -9.308 |
| out_of_road | 152 | 5000152 | `sha256:f0bc2d29aa9…` | `sha256:2c5938de06f…` | -21.154 |
| out_of_road | 158 | 5000158 | `sha256:78f85c4004e…` | `sha256:2fd1286a224…` | -10.763 |
| out_of_road | 163 | 5000163 | `sha256:31548aa1db1…` | `sha256:4e4fc6a22e1…` | -17.889 |
| out_of_road | 166 | 5000166 | `sha256:68211af461b…` | `sha256:2b76d0a0958…` | -9.907 |
| out_of_road | 169 | 5000169 | `sha256:764e3dd2ee4…` | `sha256:5a75a871f13…` | -17.785 |
| out_of_road | 172 | 5000172 | `sha256:3c6ff5486c2…` | `sha256:3df435163a1…` | -12.447 |
| out_of_road | 175 | 5000175 | `sha256:ed9d38fc63c…` | `sha256:6541d2571a7…` | -8.201 |
| out_of_road | 180 | 5000180 | `sha256:923621b72a4…` | `sha256:121544625ab…` | -12.399 |
| out_of_road | 182 | 5000182 | `sha256:81d3882adad…` | `sha256:7ee9c0a9140…` | -1.667 |
| out_of_road | 184 | 5000184 | `sha256:77e466bd3f4…` | `sha256:b3a5bf3a5ea…` | -24.430 |
| out_of_road | 191 | 5000191 | `sha256:4a56fc42943…` | `sha256:7b770f10d7a…` | -1.440 |
| out_of_road | 194 | 5000194 | `sha256:b428181d051…` | `sha256:94d0e90fd1d…` | -3.602 |
| out_of_road | 199 | 5000199 | `sha256:ec0d1a416aa…` | `sha256:0f486639346…` | -15.603 |
| out_of_road | 205 | 5000205 | `sha256:c2e1bd4a207…` | `sha256:cbc9dafa74a…` | -15.243 |
| out_of_road | 212 | 5000212 | `sha256:257255948b9…` | `sha256:3737ad168d3…` | -22.727 |
| out_of_road | 217 | 5000217 | `sha256:690c3a27fad…` | `sha256:99e7dc91f49…` | -7.110 |
| out_of_road | 220 | 5000220 | `sha256:3ee3cee3138…` | `sha256:e493dd15e72…` | -18.301 |
| out_of_road | 225 | 5000225 | `sha256:48b42e8c8da…` | `sha256:0e527adfc68…` | -15.342 |
| out_of_road | 232 | 5000232 | `sha256:d578f2b2943…` | `sha256:490a5b12c15…` | -17.104 |
| out_of_road | 238 | 5000238 | `sha256:15790ab9cf6…` | `sha256:0795abf1122…` | -21.451 |
| out_of_road | 241 | 5000241 | `sha256:e14a308054b…` | `sha256:bb8c974fd49…` | -14.750 |
| out_of_road | 247 | 5000247 | `sha256:1fc46225700…` | `sha256:8af553e72c5…` | -18.858 |
| out_of_road | 250 | 5000250 | `sha256:23c1937611f…` | `sha256:43e263e85fa…` | -16.899 |
| out_of_road | 254 | 5000254 | `sha256:15fc263ca4f…` | `sha256:3ace08c812e…` | -17.907 |
| out_of_road | 258 | 5000258 | `sha256:3a1f755d7d8…` | `sha256:80eb48f81bb…` | -12.347 |
| out_of_road | 260 | 5000260 | `sha256:aad10c757a9…` | `sha256:5d16b0d63da…` | -10.680 |
| out_of_road | 263 | 5000263 | `sha256:a3b2193c934…` | `sha256:5a5178fa5f3…` | +4.539 |
| out_of_road | 266 | 5000266 | `sha256:85f8cf1dc34…` | `sha256:d2d23fa0d5f…` | -18.657 |
| out_of_road | 269 | 5000269 | `sha256:f46d69849de…` | `sha256:15e23f3cf0a…` | -20.709 |
| out_of_road | 273 | 5000273 | `sha256:6c41bfd47da…` | `sha256:0c987f14cc6…` | -12.970 |
| out_of_road | 276 | 5000276 | `sha256:ce71ccd6fd7…` | `sha256:8be1fb03296…` | -13.846 |
| out_of_road | 278 | 5000278 | `sha256:ce233f75f2e…` | `sha256:8f79c9708b4…` | -20.931 |
| out_of_road | 280 | 5000280 | `sha256:0c81a6497fc…` | `sha256:5413e0c2f14…` | -14.815 |
| out_of_road | 283 | 5000283 | `sha256:4c33f8d88f6…` | `sha256:1ec02577721…` | -16.023 |
| out_of_road | 289 | 5000289 | `sha256:fd04c891e2b…` | `sha256:d9bcd198dbb…` | -17.893 |
| out_of_road | 292 | 5000292 | `sha256:7aea56c3ed6…` | `sha256:cb5600f8386…` | -12.584 |
| out_of_road | 296 | 5000296 | `sha256:0edad922f54…` | `sha256:6177bd8f3e9…` | -26.951 |
| out_of_road | 301 | 5000301 | `sha256:d9eb873322f…` | `sha256:f0a907c41bc…` | -9.278 |
| out_of_road | 307 | 5000307 | `sha256:c80ada84726…` | `sha256:f315e9e0aca…` | -25.523 |
| out_of_road | 312 | 5000312 | `sha256:81f164a36fb…` | `sha256:fdfce22e434…` | -13.832 |
| out_of_road | 314 | 5000314 | `sha256:51f91cccc60…` | `sha256:fb62e4fc81b…` | -6.231 |
| out_of_road | 317 | 5000317 | `sha256:6918dc83fd2…` | `sha256:5814fd3bd25…` | -16.943 |
| out_of_road | 322 | 5000322 | `sha256:390ac112069…` | `sha256:ddbbbf512ad…` | -7.959 |
| out_of_road | 324 | 5000324 | `sha256:92b605c45bc…` | `sha256:2c4cf4be367…` | -12.350 |
| out_of_road | 326 | 5000326 | `sha256:d5e76a7cfdb…` | `sha256:d02aa8861f5…` | -19.296 |
| out_of_road | 328 | 5000328 | `sha256:9eedffd7f31…` | `sha256:d53f31cb900…` | -6.677 |
| out_of_road | 331 | 5000331 | `sha256:09828d45603…` | `sha256:9609d3929f0…` | -22.534 |
| out_of_road | 333 | 5000333 | `sha256:7d309d30629…` | `sha256:987fbfb174a…` | -16.646 |
| out_of_road | 337 | 5000337 | `sha256:a53b233a89f…` | `sha256:976acc74f68…` | -23.649 |
| out_of_road | 343 | 5000343 | `sha256:6323d61116e…` | `sha256:47d37375de2…` | -13.272 |
| out_of_road | 350 | 5000350 | `sha256:2466e1e803b…` | `sha256:754c0e6fa05…` | -13.750 |
| out_of_road | 353 | 5000353 | `sha256:1b97f62f803…` | `sha256:f370f0e4b0b…` | -21.908 |
| out_of_road | 358 | 5000358 | `sha256:8fc828a2d71…` | `sha256:d787310ae7d…` | -10.372 |
| out_of_road | 361 | 5000361 | `sha256:3fff86698da…` | `sha256:a21465ec9e6…` | -16.070 |
| out_of_road | 364 | 5000364 | `sha256:c90d77a9e4e…` | `sha256:f04e9fafddd…` | -26.172 |
| out_of_road | 370 | 5000370 | `sha256:cfacfd0156a…` | `sha256:65362d75ffc…` | -17.834 |
| out_of_road | 372 | 5000372 | `sha256:5bca2f410ad…` | `sha256:6e9a9614b13…` | -19.331 |
| out_of_road | 374 | 5000374 | `sha256:bc26d7fea93…` | `sha256:bbce3fdfd4f…` | -19.748 |
| out_of_road | 377 | 5000377 | `sha256:8ed24f71460…` | `sha256:636eaeb2e13…` | -10.723 |
| out_of_road | 380 | 5000380 | `sha256:5ac0d15256c…` | `sha256:f74e8323f77…` | -7.126 |
| out_of_road | 382 | 5000382 | `sha256:094d7564a55…` | `sha256:9b0269267ec…` | -17.036 |
| out_of_road | 386 | 5000386 | `sha256:528edab02fa…` | `sha256:0e048801095…` | -12.226 |
| out_of_road | 389 | 5000389 | `sha256:64317cf6cfe…` | `sha256:e96b2b185de…` | -23.489 |
| out_of_road | 394 | 5000394 | `sha256:d8f2b3408c2…` | `sha256:3c00a4e9ca1…` | -17.237 |
| out_of_road | 399 | 5000399 | `sha256:ccefc51d96d…` | `sha256:a9a0e4f587d…` | -12.514 |
| out_of_road | 402 | 5000402 | `sha256:fe6ced36968…` | `sha256:fd243c8c407…` | -13.121 |
| out_of_road | 406 | 5000406 | `sha256:9ff05ce2565…` | `sha256:623e1f645d6…` | -7.199 |
| out_of_road | 413 | 5000413 | `sha256:00fc47aec18…` | `sha256:efe7c274fc9…` | -10.912 |
| out_of_road | 418 | 5000418 | `sha256:551a99392a6…` | `sha256:057e47c0268…` | -15.520 |
| out_of_road | 421 | 5000421 | `sha256:7683ad6d63c…` | `sha256:7239783c927…` | -16.150 |
| out_of_road | 428 | 5000428 | `sha256:954acac797d…` | `sha256:65a5217cf2b…` | +4.206 |
| out_of_road | 433 | 5000433 | `sha256:5978e5625dc…` | `sha256:7f65c8530b1…` | -14.490 |
| out_of_road | 436 | 5000436 | `sha256:51872296075…` | `sha256:b0b058fb48f…` | -6.340 |
| out_of_road | 438 | 5000438 | `sha256:a184efb154c…` | `sha256:a0ab5e5ba6c…` | -17.319 |
| out_of_road | 442 | 5000442 | `sha256:bda5ff6ec3e…` | `sha256:f52d4e7f32b…` | -17.102 |
| out_of_road | 445 | 5000445 | `sha256:391d4d310e2…` | `sha256:fc159510f1c…` | -18.691 |
| out_of_road | 448 | 5000448 | `sha256:96f8ec20657…` | `sha256:061dd8704de…` | -11.586 |
| out_of_road | 451 | 5000451 | `sha256:78c53f55150…` | `sha256:6ab58b76d71…` | -17.961 |
| out_of_road | 454 | 5000454 | `sha256:cb95dc76d81…` | `sha256:581f28e546f…` | -17.611 |
| out_of_road | 457 | 5000457 | `sha256:b196851d12c…` | `sha256:7ed4ab10614…` | -20.648 |
| out_of_road | 469 | 5000469 | `sha256:154e79ea0b0…` | `sha256:110791d06d2…` | -16.075 |
| out_of_road | 474 | 5000474 | `sha256:20a05e827c1…` | `sha256:e535259c9be…` | -11.401 |
| out_of_road | 477 | 5000477 | `sha256:c82d8139251…` | `sha256:8e6f1cf9503…` | -12.282 |
| out_of_road | 481 | 5000481 | `sha256:5f3e2a2ba1f…` | `sha256:46088c68123…` | -6.679 |
| out_of_road | 486 | 5000486 | `sha256:27ba5a3f8d8…` | `sha256:69557cc0e1f…` | -18.529 |
| out_of_road | 488 | 5000488 | `sha256:0ec75e71004…` | `sha256:8bdf1dbe56c…` | -8.295 |
| out_of_road | 493 | 5000493 | `sha256:f1447890b88…` | `sha256:5baec8c9af1…` | -13.696 |
| out_of_road | 497 | 5000497 | `sha256:cf01050cbf2…` | `sha256:8363b76caaf…` | -19.295 |
| out_of_road | 502 | 5000502 | `sha256:c890c9a6e4a…` | `sha256:dc9514a894e…` | -17.212 |
| out_of_road | 508 | 5000508 | `sha256:00b6a837a0b…` | `sha256:b0b27f51493…` | -16.147 |
| out_of_road | 511 | 5000511 | `sha256:4f4d221e8d0…` | `sha256:50aa9cdd05c…` | -3.968 |
| out_of_road | 513 | 5000513 | `sha256:5fc942956a8…` | `sha256:29637ae917a…` | -17.422 |
| out_of_road | 515 | 5000515 | `sha256:cfab74e5762…` | `sha256:c3d6a79f9d1…` | -3.905 |
| out_of_road | 520 | 5000520 | `sha256:9e8caf0d1d6…` | `sha256:88ad98466ec…` | -16.979 |
| out_of_road | 524 | 5000524 | `sha256:60c13fb8bad…` | `sha256:5d26d220891…` | -20.006 |
| out_of_road | 526 | 5000526 | `sha256:ddb822f5069…` | `sha256:3d6fe802b4a…` | -18.413 |
| out_of_road | 529 | 5000529 | `sha256:47f0e80ae4c…` | `sha256:50ecf84d18a…` | -6.939 |
| out_of_road | 532 | 5000532 | `sha256:10d3c3bc85f…` | `sha256:d1b981fce9f…` | -19.504 |
| out_of_road | 534 | 5000534 | `sha256:5a5208ba4fd…` | `sha256:0ee82d1cb26…` | -15.151 |
| out_of_road | 538 | 5000538 | `sha256:a7a31e4f236…` | `sha256:56781bd4935…` | -9.714 |
| out_of_road | 544 | 5000544 | `sha256:b8b1e8b7912…` | `sha256:7f9a2852460…` | -17.533 |
| out_of_road | 548 | 5000548 | `sha256:df222b49f00…` | `sha256:2eaf22e23c1…` | -17.262 |
| out_of_road | 551 | 5000551 | `sha256:6f8390318a4…` | `sha256:6b0c60ea837…` | -25.424 |
| out_of_road | 556 | 5000556 | `sha256:c277e7023ac…` | `sha256:1ec9e9cf76b…` | -18.240 |
| out_of_road | 559 | 5000559 | `sha256:41cd6e24e1c…` | `sha256:d5ba5c7e7f9…` | -15.304 |
| out_of_road | 562 | 5000562 | `sha256:fc742733c87…` | `sha256:31bb0e2bf3b…` | -14.535 |
| out_of_road | 566 | 5000566 | `sha256:8c6c8d1c58e…` | `sha256:1edbcaaf0c2…` | -20.283 |
| out_of_road | 571 | 5000571 | `sha256:bf534535de8…` | `sha256:8308fa1059f…` | -18.757 |
| out_of_road | 575 | 5000575 | `sha256:b8658f9fb19…` | `sha256:7b54f3c9035…` | -18.640 |
| out_of_road | 577 | 5000577 | `sha256:6fa88bae698…` | `sha256:56485b86381…` | -6.794 |
| out_of_road | 581 | 5000581 | `sha256:05a1b406a02…` | `sha256:a0670ffca0a…` | -13.332 |
| out_of_road | 583 | 5000583 | `sha256:5a196706f40…` | `sha256:bba79605e00…` | -17.658 |
| out_of_road | 586 | 5000586 | `sha256:2846ab3229a…` | `sha256:712b1a97742…` | -17.603 |
| out_of_road | 592 | 5000592 | `sha256:f687510af61…` | `sha256:4cab99619a7…` | -14.844 |
| out_of_road | 598 | 5000598 | `sha256:df58a480944…` | `sha256:fae05df5842…` | -18.205 |
| out_of_road | 601 | 5000601 | `sha256:3d43f4fe462…` | `sha256:c938e3c9b72…` | -16.037 |
| out_of_road | 605 | 5000605 | `sha256:8509e6263ca…` | `sha256:483dcf68db1…` | -20.622 |
| out_of_road | 611 | 5000611 | `sha256:4effb22f0da…` | `sha256:869dcd275c4…` | -24.706 |
| out_of_road | 621 | 5000621 | `sha256:67881f3825b…` | `sha256:fca8d356eab…` | -12.431 |
| out_of_road | 625 | 5000625 | `sha256:921796fd2ac…` | `sha256:fda72136be2…` | -22.396 |
| out_of_road | 629 | 5000629 | `sha256:e53e9cd111f…` | `sha256:e26d060e2a8…` | -8.318 |
| out_of_road | 631 | 5000631 | `sha256:08354e8efce…` | `sha256:9e34175d1ba…` | -17.421 |
| out_of_road | 635 | 5000635 | `sha256:6aad4c0f740…` | `sha256:d988e88ac20…` | -20.615 |
| out_of_road | 637 | 5000637 | `sha256:6c0d7af6df7…` | `sha256:b9ef4692581…` | -16.678 |
| out_of_road | 644 | 5000644 | `sha256:e58ff9ea82e…` | `sha256:2d7558903d7…` | -20.906 |
| out_of_road | 646 | 5000646 | `sha256:4895f99b660…` | `sha256:7e559e8f013…` | -17.362 |
| out_of_road | 652 | 5000652 | `sha256:2e2deb0d029…` | `sha256:c7018c1e9d4…` | -5.984 |
| out_of_road | 655 | 5000655 | `sha256:2830120c2c6…` | `sha256:bd521cab99e…` | -15.757 |
| out_of_road | 661 | 5000661 | `sha256:db9aaca166d…` | `sha256:a8975d745fc…` | -17.183 |
| out_of_road | 664 | 5000664 | `sha256:1e83eaffe6c…` | `sha256:fdd4ea19d7e…` | -25.184 |
| out_of_road | 669 | 5000669 | `sha256:e501ed6533c…` | `sha256:1e09b44e9fc…` | -17.207 |
| out_of_road | 673 | 5000673 | `sha256:454fe394b1b…` | `sha256:e7159e55556…` | -9.542 |
| out_of_road | 677 | 5000677 | `sha256:192ef010e48…` | `sha256:5087279963a…` | -14.252 |
| out_of_road | 683 | 5000683 | `sha256:c1c28ffbf6b…` | `sha256:a91b7bd8f41…` | -19.043 |
| out_of_road | 687 | 5000687 | `sha256:031ba116ab1…` | `sha256:25c85db2d55…` | -16.306 |
| out_of_road | 691 | 5000691 | `sha256:a96efbddb48…` | `sha256:2c9d426f508…` | -16.094 |
| out_of_road | 697 | 5000697 | `sha256:1bb6f1e3108…` | `sha256:bc892ee94c0…` | -15.563 |
| out_of_road | 706 | 5000706 | `sha256:cdd649fd390…` | `sha256:cd1c2c86310…` | -5.359 |
| out_of_road | 709 | 5000709 | `sha256:b5e6e279e39…` | `sha256:2cc01cb36ad…` | -13.409 |
| out_of_road | 712 | 5000712 | `sha256:33d29193667…` | `sha256:8280f0060fe…` | -15.675 |
| out_of_road | 721 | 5000721 | `sha256:0662a2dcbd8…` | `sha256:546914a402b…` | -8.667 |
| out_of_road | 727 | 5000727 | `sha256:62b9b8753a7…` | `sha256:2c1d46c1179…` | -16.101 |
| out_of_road | 733 | 5000733 | `sha256:0368336f73a…` | `sha256:aafef5ec7dd…` | -15.870 |
| out_of_road | 741 | 5000741 | `sha256:b453cbb6d91…` | `sha256:3a263d40584…` | -13.061 |
| out_of_road | 743 | 5000743 | `sha256:29b1430b01e…` | `sha256:9f73b0f93f8…` | -18.961 |
| out_of_road | 751 | 5000751 | `sha256:9d34a52f87f…` | `sha256:f8c44cf8cf0…` | -17.922 |
| out_of_road | 753 | 5000753 | `sha256:faad6fbaf05…` | `sha256:eff0abfab29…` | -10.555 |
| out_of_road | 758 | 5000758 | `sha256:0c3ca2ce8ae…` | `sha256:4d20bc52a14…` | -2.319 |
| out_of_road | 761 | 5000761 | `sha256:7834a6e3a01…` | `sha256:c1a95c7e24f…` | -29.002 |
| out_of_road | 763 | 5000763 | `sha256:69acd4af553…` | `sha256:a64549e6020…` | -9.407 |
| out_of_road | 769 | 5000769 | `sha256:8aa79fab185…` | `sha256:c0b6a27aa78…` | -17.148 |
| out_of_road | 771 | 5000771 | `sha256:1c61a1e183f…` | `sha256:e1b18cb76dd…` | -11.103 |
| out_of_road | 774 | 5000774 | `sha256:debbebfbde6…` | `sha256:4e8e8b01c5b…` | -15.773 |
| out_of_road | 777 | 5000777 | `sha256:dc92f355af1…` | `sha256:721ba6ab612…` | -12.518 |
| out_of_road | 780 | 5000780 | `sha256:7c9526b8dc7…` | `sha256:de04ba6c33a…` | -15.952 |
| out_of_road | 784 | 5000784 | `sha256:2974cb0af93…` | `sha256:5be55331588…` | -17.983 |
| out_of_road | 787 | 5000787 | `sha256:115ae003bbe…` | `sha256:f5e76b9a62f…` | -8.078 |
| out_of_road | 790 | 5000790 | `sha256:231064628f6…` | `sha256:2a8019e4116…` | -12.233 |
| out_of_road | 802 | 5000802 | `sha256:2dc2f365f58…` | `sha256:e125a656c80…` | -17.184 |
| out_of_road | 805 | 5000805 | `sha256:6de84d93177…` | `sha256:93f14e04359…` | -14.296 |
| out_of_road | 808 | 5000808 | `sha256:e9d0bcbf579…` | `sha256:16c2429a44b…` | -5.344 |
| out_of_road | 811 | 5000811 | `sha256:0aa4f0be6be…` | `sha256:31161ae3f00…` | -14.550 |
| out_of_road | 816 | 5000816 | `sha256:cde8f857468…` | `sha256:7c073557161…` | -13.606 |
| out_of_road | 824 | 5000824 | `sha256:cba0480f12a…` | `sha256:38167959486…` | -25.350 |
| out_of_road | 827 | 5000827 | `sha256:104db87cad1…` | `sha256:00006b77d13…` | -18.778 |
| out_of_road | 830 | 5000830 | `sha256:31bc589e529…` | `sha256:1bf2c7d2e3a…` | +7.319 |
| out_of_road | 835 | 5000835 | `sha256:dc9ebd3dac4…` | `sha256:918c8ef2ad3…` | -22.801 |
| out_of_road | 844 | 5000844 | `sha256:7311f13fbf6…` | `sha256:4071258e8d6…` | -19.037 |
| out_of_road | 847 | 5000847 | `sha256:1c3adb50442…` | `sha256:dd486f0cca1…` | -16.331 |
| out_of_road | 851 | 5000851 | `sha256:a645951895e…` | `sha256:e340776b2cd…` | -25.907 |
| out_of_road | 853 | 5000853 | `sha256:6ed5030de53…` | `sha256:b694d784605…` | -15.673 |
| out_of_road | 859 | 5000859 | `sha256:fb87fb18c01…` | `sha256:b30da1d02d9…` | -23.344 |
| out_of_road | 866 | 5000866 | `sha256:4c1201e7d14…` | `sha256:080e347a045…` | -20.984 |
| out_of_road | 871 | 5000871 | `sha256:79d497efaae…` | `sha256:01d6b56d974…` | -14.953 |
| out_of_road | 877 | 5000877 | `sha256:4eac0de23f1…` | `sha256:2f8568fa4a5…` | -17.590 |
| out_of_road | 881 | 5000881 | `sha256:9dc80de417f…` | `sha256:24752404e75…` | -18.114 |
| out_of_road | 889 | 5000889 | `sha256:48a19797fd1…` | `sha256:db5c1964f60…` | -7.325 |
| out_of_road | 891 | 5000891 | `sha256:2f81bc3a6af…` | `sha256:f59f781e82d…` | -10.280 |
| out_of_road | 895 | 5000895 | `sha256:357e8282ec4…` | `sha256:68119c4148d…` | -17.772 |
| out_of_road | 898 | 5000898 | `sha256:599c0d2c48c…` | `sha256:0e23c28f898…` | -16.016 |
| out_of_road | 901 | 5000901 | `sha256:991f7c4996f…` | `sha256:46a5942509c…` | -16.094 |
| out_of_road | 905 | 5000905 | `sha256:6eec2b510f8…` | `sha256:cc575aedc1d…` | -17.219 |
| out_of_road | 907 | 5000907 | `sha256:2f909845b51…` | `sha256:4c836318932…` | -22.413 |
| out_of_road | 909 | 5000909 | `sha256:c005f954d3f…` | `sha256:4b88d87f179…` | -18.023 |
| out_of_road | 914 | 5000914 | `sha256:3a1e1f6eecd…` | `sha256:612249a645f…` | -6.113 |
| out_of_road | 918 | 5000918 | `sha256:1e633474863…` | `sha256:72e0621ffe9…` | -15.347 |
| out_of_road | 922 | 5000922 | `sha256:0c0f2678d30…` | `sha256:d4806ecf413…` | -17.432 |
| out_of_road | 925 | 5000925 | `sha256:c0201d1cd3b…` | `sha256:632c865c489…` | -15.485 |
| out_of_road | 930 | 5000930 | `sha256:1ce65fd0ab0…` | `sha256:7cd26cc7a5b…` | -14.871 |
| out_of_road | 934 | 5000934 | `sha256:5fdbb462b92…` | `sha256:e03301b8caa…` | -16.597 |
| out_of_road | 938 | 5000938 | `sha256:3dfc29ddf28…` | `sha256:03f975a7a5e…` | -18.900 |
| out_of_road | 940 | 5000940 | `sha256:20b22f0189e…` | `sha256:d7d48bd02ac…` | -12.224 |
| out_of_road | 942 | 5000942 | `sha256:057d06078af…` | `sha256:b6ec4b229ef…` | -17.725 |
| out_of_road | 947 | 5000947 | `sha256:44ce362331d…` | `sha256:abf9be10dd4…` | -14.089 |
| out_of_road | 949 | 5000949 | `sha256:bc4d9cf5741…` | `sha256:2a99360af31…` | -17.331 |
| out_of_road | 952 | 5000952 | `sha256:105498d3abb…` | `sha256:600fd5dbd9a…` | -14.162 |
| out_of_road | 956 | 5000956 | `sha256:7d971036d46…` | `sha256:6181a1b9ecd…` | -6.947 |
| out_of_road | 968 | 5000968 | `sha256:e7339f9e14e…` | `sha256:94e26427474…` | -22.452 |
| out_of_road | 970 | 5000970 | `sha256:ae72e5dc2ff…` | `sha256:6462dba299d…` | -18.361 |
| out_of_road | 973 | 5000973 | `sha256:7e519d632a1…` | `sha256:40e098129b3…` | -16.630 |
| out_of_road | 980 | 5000980 | `sha256:711bd8960d7…` | `sha256:d443e9f3dbe…` | -22.230 |
| out_of_road | 982 | 5000982 | `sha256:2b868ff955b…` | `sha256:7d09cbe9437…` | -13.887 |
| out_of_road | 986 | 5000986 | `sha256:380e8795246…` | `sha256:ac5dff3e3a0…` | -18.038 |
| out_of_road | 988 | 5000988 | `sha256:e409995eaa5…` | `sha256:e4a4eb15317…` | -14.438 |
| out_of_road | 993 | 5000993 | `sha256:6994ed29837…` | `sha256:779dff35ccb…` | -11.658 |
| out_of_road | 998 | 5000998 | `sha256:d18b936c2f3…` | `sha256:1c929652b55…` | -25.326 |
| max_step | 32 | 5000032 | `sha256:4d4b365a343…` | `sha256:0c5c1d46b38…` | -33.317 |
| max_step | 89 | 5000089 | `sha256:c9485c8b9eb…` | `sha256:fed0130c1fe…` | -8.809 |
| max_step | 224 | 5000224 | `sha256:66181984ea7…` | `sha256:d916afa81fe…` | -18.274 |
| max_step | 242 | 5000242 | `sha256:df33b57d7c9…` | `sha256:e94ddae6bf0…` | +3.182 |
| max_step | 359 | 5000359 | `sha256:61cb6ed4609…` | `sha256:5fe981c9516…` | -1.752 |
| max_step | 425 | 5000425 | `sha256:82fcdf63e3a…` | `sha256:533eaeb04df…` | -17.631 |
| max_step | 449 | 5000449 | `sha256:b91b84a8e95…` | `sha256:2c2f8d8a28b…` | -20.998 |
| max_step | 521 | 5000521 | `sha256:208f5565f5b…` | `sha256:fb4b5dca719…` | +0.800 |
| max_step | 563 | 5000563 | `sha256:0c1332ff8df…` | `sha256:fb080b48fdb…` | -6.580 |
| max_step | 584 | 5000584 | `sha256:dbad58cd090…` | `sha256:ad4c0792f3b…` | -8.311 |
| max_step | 599 | 5000599 | `sha256:6f56173cdeb…` | `sha256:1d5d83fe9db…` | -10.632 |
| max_step | 620 | 5000620 | `sha256:2bbfb878221…` | `sha256:87ac9ca40a1…` | -13.469 |
| max_step | 641 | 5000641 | `sha256:18e22c412db…` | `sha256:a9539cfa10a…` | -7.429 |
| max_step | 722 | 5000722 | `sha256:9f0994c4b59…` | `sha256:33f9ed0e244…` | +0.772 |
| max_step | 755 | 5000755 | `sha256:94e9d6e2cc3…` | `sha256:03a2a08e724…` | -18.289 |
| max_step | 857 | 5000857 | `sha256:24b8d9a9cab…` | `sha256:31ea1c58aa4…` | -15.324 |
| max_step | 965 | 5000965 | `sha256:c4ee46e167f…` | `sha256:ff6e83f2f79…` | -2.345 |

## 2. 分类剖面 vs 目标（未折扣 + 折扣两层）

| 档 | 类 | n | 未折扣 mean±std | 目标 | |Δ| | 折扣 mean±std | dense+ | dense− | 终止项 | carl_pen | 终局值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rc=1 | arrive_dest | 221 | +49.120±+5.60 | +50 | +0.880 | +20.663±+3.65 | +20.433 | -1.313 | +0.000 | +0.000 | +30.000 |
| rc=1 | collision | 17 | -23.656±+6.63 | -20 | +3.656 | -6.819±+5.72 | +12.201 | -6.779 | -10.000 | +0.000 | -19.000 |
| rc=1 | out_of_road | 245 | -15.668±+5.02 | -15 | +0.668 | -7.886±+6.02 | +11.080 | -3.577 | -8.000 | +0.000 | -15.000 |
| rc=1 | max_step | 18 | -11.337±+6.54 | -10 | +1.337 | +4.321±+2.60 | +19.785 | -8.122 | +0.000 | +0.000 | -23.000 |
| rc=3 | arrive_dest | 221 | +49.083±+5.61 | +50 | +0.917 | +21.222±+3.63 | +22.396 | -1.313 | +0.000 | +0.000 | +28.000 |
| rc=3 | collision | 17 | -23.568±+6.70 | -20 | +3.568 | -6.412±+5.88 | +13.293 | -6.779 | -10.000 | +0.000 | -20.000 |
| rc=3 | out_of_road | 245 | -15.826±+5.30 | -15 | +0.826 | -7.837±+6.35 | +11.930 | -3.577 | -8.000 | +0.000 | -16.000 |
| rc=3 | max_step | 18 | -10.981±+6.65 | -10 | +0.981 | +4.922±+2.62 | +21.141 | -8.122 | +0.000 | +0.000 | -24.000 |
| rc=10 | arrive_dest | 221 | +49.952±+5.62 | +50 | +0.048 | +23.498±+3.58 | +29.264 | -1.313 | +0.000 | +0.000 | +22.000 |
| rc=10 | collision | 17 | -22.759±+7.02 | -20 | +2.759 | -4.788±+6.45 | +17.117 | -6.779 | -10.000 | +0.000 | -23.000 |
| rc=10 | out_of_road | 245 | -14.882±+6.41 | -15 | +0.118 | -6.781±+7.34 | +14.904 | -3.577 | -8.000 | +0.000 | -18.000 |
| rc=10 | max_step | 18 | -11.237±+7.13 | -10 | +1.237 | +6.822±+2.76 | +25.885 | -8.122 | +0.000 | +0.000 | -29.000 |
| rc=30 | arrive_dest | 221 | +49.576±+5.67 | +50 | +0.424 | +29.086±+3.37 | +48.889 | -1.313 | +0.000 | +0.000 | +2.000 |
| rc=30 | collision | 17 | -21.877±+8.35 | -20 | +1.877 | -0.714±+8.39 | +28.042 | -6.779 | -10.000 | +0.000 | -33.000 |
| rc=30 | out_of_road | 245 | -14.469±+10.25 | -15 | +0.531 | -5.114±+10.79 | +23.400 | -3.577 | -8.000 | +0.000 | -26.000 |
| rc=30 | max_step | 18 | -10.681±+8.99 | -10 | +0.681 | +12.424±+3.48 | +39.441 | -8.122 | +0.000 | +0.000 | -42.000 |

## 3. 终局值反解（每档一套；样本与审计互斥）

| 档 | 类 | n | 稠密贡献(effective) | 终止项 | carl_pen | 反解 raw | 定稿(整数) | 复算 | 复算差 |
|---|---|---|---|---|---|---|---|---|---|
| rc=1 | arrive_dest | 220 | +19.591 | +0.000 | +0.000 | +30.409 | +30.0 | +49.591 | -0.409 |
| rc=1 | collision | 17 | +8.846 | -10.000 | +0.000 | -18.846 | -19.0 | -20.154 | -0.154 |
| rc=1 | out_of_road | 245 | +7.705 | -8.000 | +0.000 | -14.705 | -15.0 | -15.295 | -0.295 |
| rc=1 | max_step | 17 | +12.506 | +0.000 | +0.000 | -22.506 | -23.0 | -10.494 | -0.494 |
| rc=3 | arrive_dest | 220 | +21.553 | +0.000 | +0.000 | +28.447 | +28.0 | +49.553 | -0.447 |
| rc=3 | collision | 17 | +9.818 | -10.000 | +0.000 | -19.818 | -20.0 | -20.182 | -0.182 |
| rc=3 | out_of_road | 245 | +8.512 | -8.000 | +0.000 | -15.512 | -16.0 | -15.488 | -0.488 |
| rc=3 | max_step | 17 | +13.882 | +0.000 | +0.000 | -23.882 | -24.0 | -10.118 | -0.118 |
| rc=10 | arrive_dest | 220 | +28.420 | +0.000 | +0.000 | +21.580 | +22.0 | +50.420 | +0.420 |
| rc=10 | collision | 17 | +13.222 | -10.000 | +0.000 | -23.222 | -23.0 | -19.778 | +0.222 |
| rc=10 | out_of_road | 245 | +11.336 | -8.000 | +0.000 | -18.336 | -18.0 | -14.664 | +0.336 |
| rc=10 | max_step | 17 | +18.699 | +0.000 | +0.000 | -28.699 | -29.0 | -10.301 | -0.301 |
| rc=30 | arrive_dest | 220 | +48.040 | +0.000 | +0.000 | +1.960 | +2.0 | +50.040 | +0.040 |
| rc=30 | collision | 17 | +22.946 | -10.000 | +0.000 | -32.946 | -33.0 | -20.054 | -0.054 |
| rc=30 | out_of_road | 245 | +19.405 | -8.000 | +0.000 | -26.405 | -26.0 | -14.595 | +0.405 |
| rc=30 | max_step | 17 | +32.462 | +0.000 | +0.000 | -42.462 | -42.0 | -9.538 | +0.462 |

各档 `terminal_values`（`error` 保留 −5）：

- rc=1：`{"arrive_dest": 30.0, "collision": -19.0, "out_of_road": -15.0, "max_step": -23.0, "error": -5.0}`
- rc=3：`{"arrive_dest": 28.0, "collision": -20.0, "out_of_road": -16.0, "max_step": -24.0, "error": -5.0}`
- rc=10：`{"arrive_dest": 22.0, "collision": -23.0, "out_of_road": -18.0, "max_step": -29.0, "error": -5.0}`
- rc=30：`{"arrive_dest": 2.0, "collision": -33.0, "out_of_road": -26.0, "max_step": -42.0, "error": -5.0}`

## 4. dense 语义拆分与 CaRL / 终局掩码命中

| 档 | 类 | n | pos×mult+neg | pos+neg（忽略乘子） | 被 CaRL 清零 | carl 命中 episode/step | 终局帧命中 step | 终局掩码回退 step |
|---|---|---|---|---|---|---|---|---|
| rc=1 | arrive_dest | 221 | +19.120 | +19.120 | +0.000 | 0/0 | 221 | 0 |
| rc=1 | collision | 17 | +5.344 | +5.422 | +0.078 | 17/17 | 17 | 0 |
| rc=1 | out_of_road | 245 | +7.332 | +7.503 | +0.171 | 245/245 | 245 | 0 |
| rc=1 | max_step | 18 | +11.663 | +11.663 | +0.000 | 0/0 | 18 | 0 |
| rc=3 | arrive_dest | 221 | +21.083 | +21.083 | +0.000 | 0/0 | 221 | 0 |
| rc=3 | collision | 17 | +6.432 | +6.514 | +0.082 | 17/17 | 17 | 0 |
| rc=3 | out_of_road | 245 | +8.174 | +8.353 | +0.179 | 245/245 | 245 | 0 |
| rc=3 | max_step | 18 | +13.019 | +13.019 | +0.000 | 0/0 | 18 | 0 |
| rc=10 | arrive_dest | 221 | +27.952 | +27.952 | +0.000 | 0/0 | 221 | 0 |
| rc=10 | collision | 17 | +10.241 | +10.338 | +0.097 | 17/17 | 17 | 0 |
| rc=10 | out_of_road | 245 | +11.118 | +11.326 | +0.208 | 245/245 | 245 | 0 |
| rc=10 | max_step | 18 | +17.763 | +17.763 | +0.000 | 0/0 | 18 | 0 |
| rc=30 | arrive_dest | 221 | +47.576 | +47.576 | +0.000 | 0/0 | 221 | 0 |
| rc=30 | collision | 17 | +21.123 | +21.263 | +0.139 | 17/17 | 17 | 0 |
| rc=30 | out_of_road | 245 | +19.531 | +19.823 | +0.292 | 245/245 | 245 | 0 |
| rc=30 | max_step | 18 | +31.319 | +31.319 | +0.000 | 0/0 | 18 | 0 |

### 4b. 违规/触发帧统计（审计样本，各类合计触发帧数）

| 档 | 类 | n | 触发帧（raw ≠ 0） |
|---|---|---|---|
| rc=1 | arrive_dest | 221 | `route_completion`=26037; `speed_ratio`=26037; `comfort_lat`=3057; `low_speed`=1041; `comfort_lon`=209; `comfort_jerk`=191; `speed_limit`=44 |
| rc=1 | collision | 17 | `route_completion`=1712; `speed_ratio`=1712; `low_speed`=734; `comfort_lat`=55; `comfort_lon`=31; `comfort_jerk`=24; `crash`=17; `speed_limit`=9 |
| rc=1 | out_of_road | 245 | `route_completion`=13928; `speed_ratio`=13928; `comfort_lat`=1532; `low_speed`=599; `comfort_lon`=313; `comfort_jerk`=258; `out_of_road`=245; `solid_line`=245; `speed_limit`=114 |
| rc=1 | max_step | 18 | `route_completion`=3600; `speed_ratio`=3600; `low_speed`=1551; `comfort_lat`=47; `comfort_lon`=34; `comfort_jerk`=24 |
| rc=3 | arrive_dest | 221 | `route_completion`=26037; `speed_ratio`=26037; `comfort_lat`=3057; `low_speed`=1041; `comfort_lon`=209; `comfort_jerk`=191; `speed_limit`=44 |
| rc=3 | collision | 17 | `route_completion`=1712; `speed_ratio`=1712; `low_speed`=734; `comfort_lat`=55; `comfort_lon`=31; `comfort_jerk`=24; `crash`=17; `speed_limit`=9 |
| rc=3 | out_of_road | 245 | `route_completion`=13928; `speed_ratio`=13928; `comfort_lat`=1532; `low_speed`=599; `comfort_lon`=313; `comfort_jerk`=258; `out_of_road`=245; `solid_line`=245; `speed_limit`=114 |
| rc=3 | max_step | 18 | `route_completion`=3600; `speed_ratio`=3600; `low_speed`=1551; `comfort_lat`=47; `comfort_lon`=34; `comfort_jerk`=24 |
| rc=10 | arrive_dest | 221 | `route_completion`=26037; `speed_ratio`=26037; `comfort_lat`=3057; `low_speed`=1041; `comfort_lon`=209; `comfort_jerk`=191; `speed_limit`=44 |
| rc=10 | collision | 17 | `route_completion`=1712; `speed_ratio`=1712; `low_speed`=734; `comfort_lat`=55; `comfort_lon`=31; `comfort_jerk`=24; `crash`=17; `speed_limit`=9 |
| rc=10 | out_of_road | 245 | `route_completion`=13928; `speed_ratio`=13928; `comfort_lat`=1532; `low_speed`=599; `comfort_lon`=313; `comfort_jerk`=258; `out_of_road`=245; `solid_line`=245; `speed_limit`=114 |
| rc=10 | max_step | 18 | `route_completion`=3600; `speed_ratio`=3600; `low_speed`=1551; `comfort_lat`=47; `comfort_lon`=34; `comfort_jerk`=24 |
| rc=30 | arrive_dest | 221 | `route_completion`=26037; `speed_ratio`=26037; `comfort_lat`=3057; `low_speed`=1041; `comfort_lon`=209; `comfort_jerk`=191; `speed_limit`=44 |
| rc=30 | collision | 17 | `route_completion`=1712; `speed_ratio`=1712; `low_speed`=734; `comfort_lat`=55; `comfort_lon`=31; `comfort_jerk`=24; `crash`=17; `speed_limit`=9 |
| rc=30 | out_of_road | 245 | `route_completion`=13928; `speed_ratio`=13928; `comfort_lat`=1532; `low_speed`=599; `comfort_lon`=313; `comfort_jerk`=258; `out_of_road`=245; `solid_line`=245; `speed_limit`=114 |
| rc=30 | max_step | 18 | `route_completion`=3600; `speed_ratio`=3600; `low_speed`=1551; `comfort_lat`=47; `comfort_lon`=34; `comfort_jerk`=24 |

## 5. low_speed 拆分（有无前车；审计样本）

| 档 | 类 | n | 有前车 Σ/步 | 无前车 Σ/步 | 有前车步 | 无前车步 |
|---|---|---|---|---|---|---|
| rc=1 | arrive_dest | 221 | -0.239/-0.000 | -0.053/-0.000 | 9563 | 16474 |
| rc=1 | collision | 17 | -4.044/-0.003 | -0.420/-0.002 | 1504 | 208 |
| rc=1 | out_of_road | 245 | -0.171/-0.000 | -0.033/-0.000 | 4644 | 9284 |
| rc=1 | max_step | 18 | -6.574/-0.002 | -0.291/-0.001 | 3087 | 513 |
| rc=3 | arrive_dest | 221 | -0.239/-0.000 | -0.053/-0.000 | 9563 | 16474 |
| rc=3 | collision | 17 | -4.044/-0.003 | -0.420/-0.002 | 1504 | 208 |
| rc=3 | out_of_road | 245 | -0.171/-0.000 | -0.033/-0.000 | 4644 | 9284 |
| rc=3 | max_step | 18 | -6.574/-0.002 | -0.291/-0.001 | 3087 | 513 |
| rc=10 | arrive_dest | 221 | -0.239/-0.000 | -0.053/-0.000 | 9563 | 16474 |
| rc=10 | collision | 17 | -4.044/-0.003 | -0.420/-0.002 | 1504 | 208 |
| rc=10 | out_of_road | 245 | -0.171/-0.000 | -0.033/-0.000 | 4644 | 9284 |
| rc=10 | max_step | 18 | -6.574/-0.002 | -0.291/-0.001 | 3087 | 513 |
| rc=30 | arrive_dest | 221 | -0.239/-0.000 | -0.053/-0.000 | 9563 | 16474 |
| rc=30 | collision | 17 | -4.044/-0.003 | -0.420/-0.002 | 1504 | 208 |
| rc=30 | out_of_road | 245 | -0.171/-0.000 | -0.033/-0.000 | 4644 | 9284 |
| rc=30 | max_step | 18 | -6.574/-0.002 | -0.291/-0.001 | 3087 | 513 |

## 6. 逐项贡献 top（审计样本；各类 mean）

| 类 | 项贡献 mean（按 |mean| 降序） |
|---|---|
| arrive_dest | `speed_ratio`=+19.452; `route_completion`=+0.981; `comfort_lon`=-0.727; `low_speed`=-0.292; `speed_limit`=-0.108; `comfort_jerk`=-0.095; `comfort_lat`=-0.092; `crash`=+0.000 |
| collision | `speed_ratio`=+11.654; `crash`=-10.000; `low_speed`=-4.465; `comfort_lon`=-1.562; `route_completion`=+0.546; `speed_limit`=-0.480; `comfort_jerk`=-0.223; `comfort_lat`=-0.049 |
| out_of_road | `speed_ratio`=+10.655; `out_of_road`=-8.000; `solid_line`=-2.000; `comfort_lon`=-0.915; `route_completion`=+0.425; `speed_limit`=-0.290; `low_speed`=-0.204; `comfort_jerk`=-0.119 |
| max_step | `speed_ratio`=+19.108; `low_speed`=-6.866; `comfort_lon`=-1.035; `route_completion`=+0.678; `comfort_jerk`=-0.201; `comfort_lat`=-0.021; `crash`=+0.000; `out_of_road`=+0.000 |

## 7. 未决问题 / 说明

- 审计样本与反解样本同类内交替分配（互斥）；每 episode 记录 id/seed/ctx_sha256/文件 sha256。
- max_step 类：rollout 循环在 max_steps 截断时 env 未置 info.max_step（与 eval 分类同口径按步数），重放时对终局帧补 max_step=True 以结算终局值。
- rc=1.0 为代码默认基准档（回填 default_terminal_values）；3/10/30 为 P4 扫档（每档一套终局值，不得跨档复用）。
- dense 语义两列：pos×mult+neg（真实聚合器口径，用于反解）与 pos+neg（忽略 CaRL 乘子，仅对照）。
- E-β′ vs E-β″：本报告 = E-β′（旧架构，P2/Gate2）；E-β″ 在 P3 重训后按 docs/rl_reward_v5.md §7 用同一工具复算（P3 后、P4 前，作为 P4 奖励口径）。
- error 类 n=0（不在剖面目标内；按 §7 只报 n 与区间）。
