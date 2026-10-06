# v7 phase3 安全配方重跑（fix-11）原始数据

- 生成：2026-10-05 16:45:10+0800
- run：`BTC20261005-0856_v7struct_v5_p3fix`（driver v2：单 epoch 子进程 + 进程间离线 clean150 守护）

## 训练（逐 epoch）
```json
{}
```

## 守护（clean150）
```json
{
  "min_success": 0.13,
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
  "records": [
    {
      "epoch": 1,
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch001.pt",
      "min_success": 0.13,
      "metrics": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/guard/epoch001/metrics.json",
      "reused": true,
      "rc": null,
      "success": 0.32,
      "collapsed": false,
      "reason": "",
      "status": "ok"
    },
    {
      "epoch": 2,
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch002.pt",
      "min_success": 0.13,
      "metrics": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/guard/epoch002/metrics.json",
      "reused": false,
      "rc": 0,
      "success": 0.32666666666666666,
      "collapsed": false,
      "reason": "",
      "wall_s": 233.45458579063416,
      "killed": null,
      "status": "ok"
    },
    {
      "epoch": 3,
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch003.pt",
      "min_success": 0.13,
      "metrics": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/guard/epoch003/metrics.json",
      "reused": false,
      "rc": 0,
      "success": 0.26,
      "collapsed": false,
      "reason": "",
      "wall_s": 243.79305744171143,
      "killed": null,
      "status": "ok"
    },
    {
      "epoch": 4,
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch004.pt",
      "min_success": 0.13,
      "metrics": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/guard/epoch004/metrics.json",
      "reused": false,
      "rc": 0,
      "success": 0.25333333333333335,
      "collapsed": false,
      "reason": "",
      "wall_s": 248.26874661445618,
      "killed": null,
      "status": "ok"
    },
    {
      "epoch": 5,
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch005.pt",
      "min_success": 0.13,
      "metrics": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/guard/epoch005/metrics.json",
      "reused": false,
      "rc": 0,
      "success": 0.18666666666666668,
      "collapsed": false,
      "reason": "",
      "wall_s": 246.02278208732605,
      "killed": null,
      "status": "ok"
    }
  ]
}
```

## keep-best / 终评
```json
{
  "best": {
    "tag": "epoch002",
    "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch002.pt",
    "clean150": 0.32666666666666666,
    "source": "guard",
    "epoch": 2
  },
  "candidates": [
    {
      "tag": "init_b10",
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5/stage_b/ckpt_epoch010.pt",
      "clean150": 0.26,
      "source": "ref:runs/BTC20261005-115035_v7sb_b_ckpt_epoch010_sub150",
      "epoch": 0
    },
    {
      "tag": "epoch001",
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch001.pt",
      "clean150": 0.32,
      "source": "guard",
      "epoch": 1
    },
    {
      "tag": "epoch002",
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch002.pt",
      "clean150": 0.32666666666666666,
      "source": "guard",
      "epoch": 2
    },
    {
      "tag": "epoch003",
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch003.pt",
      "clean150": 0.26,
      "source": "guard",
      "epoch": 3
    },
    {
      "tag": "epoch004",
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch004.pt",
      "clean150": 0.25333333333333335,
      "source": "guard",
      "epoch": 4
    },
    {
      "tag": "epoch005",
      "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3fix/stage_b/ckpt_epoch005.pt",
      "clean150": 0.18666666666666668,
      "source": "guard",
      "epoch": 5
    }
  ]
}
```

## 配对
```json
{
  "w1_clean500": {
    "tag": "w1_clean500",
    "baseline": "runs/BTC20261003-045912_v7p1dagger_w1_clean500",
    "rc": 0,
    "wall_s": 0.25391578674316406,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": -19.8,
      "z": 8.165382378538993,
      "mcnemar_exact_p": 3.1772595880751415e-17,
      "ci95_pp": [
        -24.2,
        -15.6
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 24,
        "broken": 123,
        "both_pass": 140,
        "both_fail": 213,
        "net": -99,
        "z": 8.165382378538993,
        "mcnemar_exact_p": 3.1772595880751415e-17,
        "rate_base": 0.526,
        "rate_agent": 0.328,
        "delta": -0.198,
        "delta_pp": -19.8,
        "ci95": [
          -0.242,
          -0.156
        ],
        "ci95_pp": [
          -24.2,
          -15.6
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 16,
        "broken": 27,
        "both_pass": 6,
        "both_fail": 451,
        "net": -11,
        "z": 1.6774842736586515,
        "mcnemar_exact_p": 0.12628947438543037,
        "rate_base": 0.066,
        "rate_agent": 0.044,
        "delta": -0.022,
        "delta_pp": -2.1999999999999997,
        "ci95": [
          -0.048,
          0.004
        ],
        "ci95_pp": [
          -4.8,
          0.4
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 110,
        "broken": 49,
        "both_pass": 150,
        "both_fail": 191,
        "net": 61,
        "z": 4.8376146728806795,
        "mcnemar_exact_p": 1.4699503840243076e-06,
        "rate_base": 0.398,
        "rate_agent": 0.52,
        "delta": 0.122,
        "delta_pp": 12.2,
        "ci95": [
          0.074,
          0.17
        ],
        "ci95_pp": [
          7.3999999999999995,
          17.0
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 59,
        "broken": 2,
        "both_pass": 3,
        "both_fail": 436,
        "net": 57,
        "z": 7.298102156175071,
        "mcnemar_exact_p": 1.6410484082740595e-15,
        "rate_base": 0.01,
        "rate_agent": 0.124,
        "delta": 0.114,
        "delta_pp": 11.4,
        "ci95": [
          0.086,
          0.144
        ],
        "ci95_pp": [
          8.6,
          14.399999999999999
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "s11_clean500": {
    "tag": "s11_clean500",
    "baseline": "runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean",
    "rc": 0,
    "wall_s": 0.24104905128479004,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": -34.0,
      "z": 11.258525035052871,
      "mcnemar_exact_p": 2.2885715145478462e-32,
      "ci95_pp": [
        -39.2,
        -28.999999999999996
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 29,
        "broken": 199,
        "both_pass": 135,
        "both_fail": 137,
        "net": -170,
        "z": 11.258525035052871,
        "mcnemar_exact_p": 2.2885715145478462e-32,
        "rate_base": 0.668,
        "rate_agent": 0.328,
        "delta": -0.34,
        "delta_pp": -34.0,
        "ci95": [
          -0.392,
          -0.29
        ],
        "ci95_pp": [
          -39.2,
          -28.999999999999996
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 12,
        "broken": 81,
        "both_pass": 10,
        "both_fail": 397,
        "net": -69,
        "z": 7.154966693639935,
        "mcnemar_exact_p": 9.828750211474817e-14,
        "rate_base": 0.182,
        "rate_agent": 0.044,
        "delta": -0.138,
        "delta_pp": -13.8,
        "ci95": [
          -0.174,
          -0.102
        ],
        "ci95_pp": [
          -17.4,
          -10.2
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 214,
        "broken": 23,
        "both_pass": 46,
        "both_fail": 217,
        "net": 191,
        "z": 12.40678322701715,
        "mcnemar_exact_p": 5.4132454511172875e-40,
        "rate_base": 0.138,
        "rate_agent": 0.52,
        "delta": 0.382,
        "delta_pp": 38.2,
        "ci95": [
          0.332,
          0.432
        ],
        "ci95_pp": [
          33.2,
          43.2
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 62,
        "broken": 6,
        "both_pass": 0,
        "both_fail": 432,
        "net": 56,
        "z": 6.790997501017324,
        "mcnemar_exact_p": 8.181953378705309e-13,
        "rate_base": 0.012,
        "rate_agent": 0.124,
        "delta": 0.112,
        "delta_pp": 11.200000000000001,
        "ci95": [
          0.082,
          0.144
        ],
        "ci95_pp": [
          8.200000000000001,
          14.399999999999999
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "idm_clean500": {
    "tag": "idm_clean500",
    "baseline": "runs/BTC20261002-100413_v7p0_idm_clean500",
    "rc": 0,
    "wall_s": 0.23819470405578613,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": -41.4,
      "z": 12.91230523412237,
      "mcnemar_exact_p": 3.3047638996417964e-43,
      "ci95_pp": [
        -46.6,
        -36.199999999999996
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 25,
        "broken": 232,
        "both_pass": 139,
        "both_fail": 104,
        "net": -207,
        "z": 12.91230523412237,
        "mcnemar_exact_p": 3.3047638996417964e-43,
        "rate_base": 0.742,
        "rate_agent": 0.328,
        "delta": -0.414,
        "delta_pp": -41.4,
        "ci95": [
          -0.466,
          -0.362
        ],
        "ci95_pp": [
          -46.6,
          -36.199999999999996
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 17,
        "broken": 82,
        "both_pass": 5,
        "both_fail": 396,
        "net": -65,
        "z": 6.5327457991848785,
        "mcnemar_exact_p": 2.180641080014394e-11,
        "rate_base": 0.174,
        "rate_agent": 0.044,
        "delta": -0.13,
        "delta_pp": -13.0,
        "ci95": [
          -0.168,
          -0.092
        ],
        "ci95_pp": [
          -16.8,
          -9.2
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 235,
        "broken": 2,
        "both_pass": 25,
        "both_fail": 238,
        "net": 233,
        "z": 15.134976397356,
        "mcnemar_exact_p": 2.554063727392286e-67,
        "rate_base": 0.054,
        "rate_agent": 0.52,
        "delta": 0.466,
        "delta_pp": 46.6,
        "ci95": [
          0.422,
          0.51
        ],
        "ci95_pp": [
          42.199999999999996,
          51.0
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 55,
        "broken": 8,
        "both_pass": 7,
        "both_fail": 430,
        "net": 47,
        "z": 5.921443410477893,
        "mcnemar_exact_p": 9.761673086614714e-10,
        "rate_base": 0.03,
        "rate_agent": 0.124,
        "delta": 0.094,
        "delta_pp": 9.4,
        "ci95": [
          0.064,
          0.124
        ],
        "ci95_pp": [
          6.4,
          12.4
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "p1b_clean500": {
    "tag": "p1b_clean500",
    "baseline": "runs/BTC20261002-162639_v7p1b_screen_epoch010",
    "rc": 0,
    "wall_s": 0.24747180938720703,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": 1.4000000000000001,
      "z": 0.741998516004452,
      "mcnemar_exact_p": 0.5250147872924269,
      "ci95_pp": [
        -2.4,
        5.004999999999928
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 48,
        "broken": 41,
        "both_pass": 116,
        "both_fail": 295,
        "net": 7,
        "z": 0.741998516004452,
        "mcnemar_exact_p": 0.5250147872924269,
        "rate_base": 0.314,
        "rate_agent": 0.328,
        "delta": 0.014,
        "delta_pp": 1.4000000000000001,
        "ci95": [
          -0.024,
          0.050049999999999276
        ],
        "ci95_pp": [
          -2.4,
          5.004999999999928
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 17,
        "broken": 41,
        "both_pass": 5,
        "both_fail": 437,
        "net": -24,
        "z": 3.151354388633341,
        "mcnemar_exact_p": 0.0022325516199429227,
        "rate_base": 0.092,
        "rate_agent": 0.044,
        "delta": -0.048,
        "delta_pp": -4.8,
        "ci95": [
          -0.078,
          -0.018
        ],
        "ci95_pp": [
          -7.8,
          -1.7999999999999998
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 41,
        "broken": 79,
        "both_pass": 219,
        "both_fail": 161,
        "net": -38,
        "z": 3.468909530866052,
        "mcnemar_exact_p": 0.00066672389700352,
        "rate_base": 0.596,
        "rate_agent": 0.52,
        "delta": -0.076,
        "delta_pp": -7.6,
        "ci95": [
          -0.118,
          -0.034
        ],
        "ci95_pp": [
          -11.799999999999999,
          -3.4000000000000004
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 62,
        "broken": 0,
        "both_pass": 0,
        "both_fail": 438,
        "net": 62,
        "z": 7.874007874011811,
        "mcnemar_exact_p": 4.336808689942018e-19,
        "rate_base": 0.0,
        "rate_agent": 0.124,
        "delta": 0.124,
        "delta_pp": 12.4,
        "ci95": [
          0.096,
          0.154
        ],
        "ci95_pp": [
          9.6,
          15.4
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "w1_eval500": {
    "tag": "w1_eval500",
    "baseline": "runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory",
    "rc": 0,
    "wall_s": 0.24652814865112305,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": -20.599999999999998,
      "z": 8.327056459580765,
      "mcnemar_exact_p": 7.268499971220178e-18,
      "ci95_pp": [
        -25.2,
        -16.2
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 25,
        "broken": 128,
        "both_pass": 137,
        "both_fail": 210,
        "net": -103,
        "z": 8.327056459580765,
        "mcnemar_exact_p": 7.268499971220178e-18,
        "rate_base": 0.53,
        "rate_agent": 0.324,
        "delta": -0.206,
        "delta_pp": -20.599999999999998,
        "ci95": [
          -0.252,
          -0.162
        ],
        "ci95_pp": [
          -25.2,
          -16.2
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 16,
        "broken": 36,
        "both_pass": 1,
        "both_fail": 447,
        "net": -20,
        "z": 2.773500981126146,
        "mcnemar_exact_p": 0.0077874363057435225,
        "rate_base": 0.074,
        "rate_agent": 0.034,
        "delta": -0.04,
        "delta_pp": -4.0,
        "ci95": [
          -0.068,
          -0.012
        ],
        "ci95_pp": [
          -6.800000000000001,
          -1.2
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 124,
        "broken": 46,
        "both_pass": 143,
        "both_fail": 187,
        "net": 78,
        "z": 5.982326913009489,
        "mcnemar_exact_p": 1.8343461557498238e-09,
        "rate_base": 0.378,
        "rate_agent": 0.534,
        "delta": 0.156,
        "delta_pp": 15.6,
        "ci95": [
          0.108,
          0.204
        ],
        "ci95_pp": [
          10.8,
          20.4
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 62,
        "broken": 6,
        "both_pass": 3,
        "both_fail": 429,
        "net": 56,
        "z": 6.790997501017324,
        "mcnemar_exact_p": 8.181953378705309e-13,
        "rate_base": 0.018,
        "rate_agent": 0.13,
        "delta": 0.112,
        "delta_pp": 11.200000000000001,
        "ci95": [
          0.082,
          0.144
        ],
        "ci95_pp": [
          8.200000000000001,
          14.399999999999999
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "s11_eval500": {
    "tag": "s11_eval500",
    "baseline": "runs/BTC20261005-074755_v7p2_s11_eval500_clean",
    "rc": 0,
    "wall_s": 0.24585318565368652,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": -32.2,
      "z": 10.685945316759549,
      "mcnemar_exact_p": 6.239258105730362e-29,
      "ci95_pp": [
        -37.4,
        -27.0
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 33,
        "broken": 194,
        "both_pass": 129,
        "both_fail": 144,
        "net": -161,
        "z": 10.685945316759549,
        "mcnemar_exact_p": 6.239258105730362e-29,
        "rate_base": 0.646,
        "rate_agent": 0.324,
        "delta": -0.322,
        "delta_pp": -32.2,
        "ci95": [
          -0.374,
          -0.27
        ],
        "ci95_pp": [
          -37.4,
          -27.0
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 11,
        "broken": 95,
        "both_pass": 6,
        "both_fail": 388,
        "net": -84,
        "z": 8.158801243801019,
        "mcnemar_exact_p": 7.727370135207909e-18,
        "rate_base": 0.202,
        "rate_agent": 0.034,
        "delta": -0.168,
        "delta_pp": -16.8,
        "ci95": [
          -0.206,
          -0.13
        ],
        "ci95_pp": [
          -20.599999999999998,
          -13.0
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 211,
        "broken": 14,
        "both_pass": 56,
        "both_fail": 219,
        "net": 197,
        "z": 13.133333333333333,
        "mcnemar_exact_p": 2.568418773333829e-46,
        "rate_base": 0.14,
        "rate_agent": 0.534,
        "delta": 0.394,
        "delta_pp": 39.4,
        "ci95": [
          0.346,
          0.442
        ],
        "ci95_pp": [
          34.599999999999994,
          44.2
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 63,
        "broken": 5,
        "both_pass": 2,
        "both_fail": 430,
        "net": 58,
        "z": 7.033533126053656,
        "mcnemar_exact_p": 7.651062942926057e-14,
        "rate_base": 0.014,
        "rate_agent": 0.13,
        "delta": 0.116,
        "delta_pp": 11.600000000000001,
        "ci95": [
          0.086,
          0.146
        ],
        "ci95_pp": [
          8.6,
          14.6
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "idm_eval500": {
    "tag": "idm_eval500",
    "baseline": "runs/BTC20260927-1839_eval500_baseline",
    "rc": 0,
    "wall_s": 0.23393726348876953,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": -43.2,
      "z": 13.716013716020575,
      "mcnemar_exact_p": 2.833197209660818e-50,
      "ci95_pp": [
        -48.0,
        -38.4
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 16,
        "broken": 232,
        "both_pass": 146,
        "both_fail": 106,
        "net": -216,
        "z": 13.716013716020575,
        "mcnemar_exact_p": 2.833197209660818e-50,
        "rate_base": 0.756,
        "rate_agent": 0.324,
        "delta": -0.432,
        "delta_pp": -43.2,
        "ci95": [
          -0.48,
          -0.384
        ],
        "ci95_pp": [
          -48.0,
          -38.4
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 15,
        "broken": 70,
        "both_pass": 2,
        "both_fail": 413,
        "net": -55,
        "z": 5.965587590013045,
        "mcnemar_exact_p": 1.1724940227199809e-09,
        "rate_base": 0.144,
        "rate_agent": 0.034,
        "delta": -0.11,
        "delta_pp": -11.0,
        "ci95": [
          -0.144,
          -0.076
        ],
        "ci95_pp": [
          -14.399999999999999,
          -7.6
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 236,
        "broken": 3,
        "both_pass": 31,
        "both_fail": 230,
        "net": 233,
        "z": 15.071517097328416,
        "mcnemar_exact_p": 5.15159471436294e-66,
        "rate_base": 0.068,
        "rate_agent": 0.534,
        "delta": 0.466,
        "delta_pp": 46.6,
        "ci95": [
          0.422,
          0.51
        ],
        "ci95_pp": [
          42.199999999999996,
          51.0
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 56,
        "broken": 7,
        "both_pass": 9,
        "both_fail": 428,
        "net": 49,
        "z": 6.173419725817378,
        "mcnemar_exact_p": 1.363671398024735e-10,
        "rate_base": 0.032,
        "rate_agent": 0.13,
        "delta": 0.098,
        "delta_pp": 9.8,
        "ci95": [
          0.068,
          0.128
        ],
        "ci95_pp": [
          6.800000000000001,
          12.8
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "p1b_eval500": {
    "tag": "p1b_eval500",
    "baseline": "runs/BTC20261002-164209_v7p1b_eval500_sel",
    "rc": 0,
    "wall_s": 0.24355697631835938,
    "n_pairs": 500,
    "verdict": {
      "delta_pp": 1.2,
      "z": 0.6546536707079772,
      "mcnemar_exact_p": 0.5856467947212246,
      "ci95_pp": [
        -2.4,
        4.8
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 500,
        "fixed": 45,
        "broken": 39,
        "both_pass": 117,
        "both_fail": 299,
        "net": 6,
        "z": 0.6546536707079772,
        "mcnemar_exact_p": 0.5856467947212246,
        "rate_base": 0.312,
        "rate_agent": 0.324,
        "delta": 0.012,
        "delta_pp": 1.2,
        "ci95": [
          -0.024,
          0.048
        ],
        "ci95_pp": [
          -2.4,
          4.8
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 500,
        "fixed": 14,
        "broken": 33,
        "both_pass": 3,
        "both_fail": 450,
        "net": -19,
        "z": 2.7714348384599967,
        "mcnemar_exact_p": 0.007942726509483578,
        "rate_base": 0.072,
        "rate_agent": 0.034,
        "delta": -0.038,
        "delta_pp": -3.8,
        "ci95": [
          -0.064,
          -0.012
        ],
        "ci95_pp": [
          -6.4,
          -1.2
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 500,
        "fixed": 33,
        "broken": 75,
        "both_pass": 234,
        "both_fail": 158,
        "net": -42,
        "z": 4.041451884327381,
        "mcnemar_exact_p": 6.550370792480044e-05,
        "rate_base": 0.618,
        "rate_agent": 0.534,
        "delta": -0.084,
        "delta_pp": -8.4,
        "ci95": [
          -0.124,
          -0.044
        ],
        "ci95_pp": [
          -12.4,
          -4.3999999999999995
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 500,
        "fixed": 64,
        "broken": 0,
        "both_pass": 1,
        "both_fail": 435,
        "net": 64,
        "z": 8.0,
        "mcnemar_exact_p": 1.0842021724855044e-19,
        "rate_base": 0.002,
        "rate_agent": 0.13,
        "delta": 0.128,
        "delta_pp": 12.8,
        "ci95": [
          0.1,
          0.158
        ],
        "ci95_pp": [
          10.0,
          15.8
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "idm_tg45": {
    "tag": "idm_tg45",
    "baseline": "runs/BTC20261002-160154_v7p1b_tg45_idm",
    "rc": 0,
    "wall_s": 0.12283205986022949,
    "n_pairs": 45,
    "verdict": {
      "delta_pp": -60.0,
      "z": 5.196152422706632,
      "mcnemar_exact_p": 1.4901161193847656e-08,
      "ci95_pp": [
        -73.33333333333333,
        -46.666666666666664
      ],
      "ci_lower_gt_zero": false,
      "positive_3pt_z196": false
    },
    "items": {
      "success": {
        "n": 45,
        "fixed": 0,
        "broken": 27,
        "both_pass": 8,
        "both_fail": 10,
        "net": -27,
        "z": 5.196152422706632,
        "mcnemar_exact_p": 1.4901161193847656e-08,
        "rate_base": 0.7777777777777778,
        "rate_agent": 0.17777777777777778,
        "delta": -0.6,
        "delta_pp": -60.0,
        "ci95": [
          -0.7333333333333333,
          -0.4666666666666667
        ],
        "ci95_pp": [
          -73.33333333333333,
          -46.666666666666664
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 45,
        "fixed": 0,
        "broken": 3,
        "both_pass": 0,
        "both_fail": 42,
        "net": -3,
        "z": 1.7320508075688774,
        "mcnemar_exact_p": 0.25,
        "rate_base": 0.06666666666666667,
        "rate_agent": 0.0,
        "delta": -0.06666666666666667,
        "delta_pp": -6.666666666666667,
        "ci95": [
          -0.15555555555555556,
          0.0
        ],
        "ci95_pp": [
          -15.555555555555555,
          0.0
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 45,
        "fixed": 25,
        "broken": 0,
        "both_pass": 2,
        "both_fail": 18,
        "net": 25,
        "z": 5.0,
        "mcnemar_exact_p": 5.960464477539063e-08,
        "rate_base": 0.044444444444444446,
        "rate_agent": 0.6,
        "delta": 0.5555555555555556,
        "delta_pp": 55.55555555555556,
        "ci95": [
          0.4,
          0.7111111111111111
        ],
        "ci95_pp": [
          40.0,
          71.11111111111111
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 45,
        "fixed": 6,
        "broken": 1,
        "both_pass": 4,
        "both_fail": 34,
        "net": 5,
        "z": 1.889822365046136,
        "mcnemar_exact_p": 0.125,
        "rate_base": 0.1111111111111111,
        "rate_agent": 0.2222222222222222,
        "delta": 0.1111111111111111,
        "delta_pp": 11.11111111111111,
        "ci95": [
          0.0,
          0.2222222222222222
        ],
        "ci95_pp": [
          0.0,
          22.22222222222222
        ],
        "direction": "lower_is_better"
      }
    }
  },
  "p1b_tg45": {
    "tag": "p1b_tg45",
    "baseline": "runs/BTC20261002-160114_v7p1b_tg45",
    "rc": 0,
    "wall_s": 0.12773656845092773,
    "n_pairs": 45,
    "verdict": {
      "delta_pp": 17.77777777777778,
      "z": 2.82842712474619,
      "mcnemar_exact_p": 0.0078125,
      "ci95_pp": [
        6.666666666666667,
        28.888888888888886
      ],
      "ci_lower_gt_zero": true,
      "positive_3pt_z196": true
    },
    "items": {
      "success": {
        "n": 45,
        "fixed": 8,
        "broken": 0,
        "both_pass": 0,
        "both_fail": 37,
        "net": 8,
        "z": 2.82842712474619,
        "mcnemar_exact_p": 0.0078125,
        "rate_base": 0.0,
        "rate_agent": 0.17777777777777778,
        "delta": 0.17777777777777778,
        "delta_pp": 17.77777777777778,
        "ci95": [
          0.06666666666666667,
          0.28888888888888886
        ],
        "ci95_pp": [
          6.666666666666667,
          28.888888888888886
        ],
        "direction": "higher_is_better"
      },
      "collision": {
        "n": 45,
        "fixed": 0,
        "broken": 3,
        "both_pass": 0,
        "both_fail": 42,
        "net": -3,
        "z": 1.7320508075688774,
        "mcnemar_exact_p": 0.25,
        "rate_base": 0.06666666666666667,
        "rate_agent": 0.0,
        "delta": -0.06666666666666667,
        "delta_pp": -6.666666666666667,
        "ci95": [
          -0.15555555555555556,
          0.0
        ],
        "ci95_pp": [
          -15.555555555555555,
          0.0
        ],
        "direction": "lower_is_better"
      },
      "off_road": {
        "n": 45,
        "fixed": 0,
        "broken": 15,
        "both_pass": 27,
        "both_fail": 3,
        "net": -15,
        "z": 3.8729833462074166,
        "mcnemar_exact_p": 6.103515625e-05,
        "rate_base": 0.9333333333333333,
        "rate_agent": 0.6,
        "delta": -0.3333333333333333,
        "delta_pp": -33.33333333333333,
        "ci95": [
          -0.4666666666666667,
          -0.2
        ],
        "ci95_pp": [
          -46.666666666666664,
          -20.0
        ],
        "direction": "lower_is_better"
      },
      "max_step": {
        "n": 45,
        "fixed": 10,
        "broken": 0,
        "both_pass": 0,
        "both_fail": 35,
        "net": 10,
        "z": 3.162277660168379,
        "mcnemar_exact_p": 0.001953125,
        "rate_base": 0.0,
        "rate_agent": 0.2222222222222222,
        "delta": 0.2222222222222222,
        "delta_pp": 22.22222222222222,
        "ci95": [
          0.1111111111111111,
          0.35555555555555557
        ],
        "ci95_pp": [
          11.11111111111111,
          35.55555555555556
        ],
        "direction": "lower_is_better"
      }
    }
  }
}
```

## T3
```json
{
  "n_ids": 9,
  "n_s1_resolved": 0,
  "rows": [
    {
      "id": 34,
      "ckpt_term": "max_step",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": false,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 243,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": true,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 164,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": true,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 84,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": false,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 166,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": false,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 131,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": true,
      "s1_resolved": false,
      "plan_crosses_booth": true,
      "ckpt_error": null
    },
    {
      "id": 76,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": false,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 147,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": false,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    },
    {
      "id": 239,
      "ckpt_term": "out_of_road",
      "base_term": "arrive_dest",
      "ckpt_clearance": null,
      "moved_free_lane": false,
      "s1_resolved": false,
      "plan_crosses_booth": false,
      "ckpt_error": null
    }
  ],
  "json": "/tmp/opencode/v7_p3fix_t3.json",
  "rc": 0,
  "wall_s": 54.97192621231079
}
```

