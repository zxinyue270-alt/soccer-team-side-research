"""Run TrackEval (sn-trackeval 0.4.0, MIT) on the MOTChallenge files ``python -m gbf eval-gsr-dev``
wrote for several sequences: measure T of prereg/eval-gsr-dev-v1.md, per sequence and combined.

Not part of the gbf package and not imported by it: it runs in a separate interpreter that has
sn-trackeval installed (Python 3.9 or later), out of gbf's process:

    <python with sn-trackeval> tools/run_trackeval_dev.py <root> <seq_info> <ignored> <out json>

<root> holds gt/GSR-eval/<seq>/ and trackers/GSR-eval/<tracker>/data/<seq>.txt for every
sequence named in the JSON file <seq_info> ({seq: number of frames}); the third argument keeps the
argument order of tools/run_trackeval.py. Every tracker under trackers/GSR-eval is evaluated
(class pedestrian, MOT preprocessing on: predictions matched to distractor boxes are
dropped). The summary JSON names the installed sn-trackeval version and the Python version
and lists, per tracker, the fields of tools/run_trackeval.py for every sequence
(``per_seq``) and for TrackEval's combination over all of them (``combined``).
"""

import json
import os
import platform
import sys
from importlib import metadata

import numpy as np
import trackeval


def fields(res):
    hota, clear, ident, count = res["HOTA"], res["CLEAR"], res["Identity"], res["Count"]
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "LocA": float(np.mean(hota["LocA"])),
        "MOTA": float(clear["MOTA"]),
        "MOTP": float(clear["MOTP"]),
        "IDSW": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]),
        "Rcll": float(clear["CLR_Re"]),
        "Prcn": float(clear["CLR_Pr"]),
        "TP": int(clear["CLR_TP"]),
        "FN": int(clear["CLR_FN"]),
        "FP": int(clear["CLR_FP"]),
        "IDF1": float(ident["IDF1"]),
        "IDR": float(ident["IDR"]),
        "IDP": float(ident["IDP"]),
        "GT_IDs": int(count["GT_IDs"]),
        "IDs": int(count["IDs"]),
        "GT_Dets": int(count["GT_Dets"]),
        "Dets": int(count["Dets"]),
    }


def main(argv):
    root, out_json = argv[0], argv[3]
    with open(argv[1], encoding="utf-8") as handle:
        seq_info = {str(k): int(v) for k, v in json.load(handle).items()}
    trackers_dir = os.path.join(root, "trackers", "GSR-eval")
    trackers = sorted(os.listdir(trackers_dir))
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": os.path.join(root, "gt"),
            "TRACKERS_FOLDER": os.path.join(root, "trackers"),
            "OUTPUT_FOLDER": os.path.join(root, "output"),
            "BENCHMARK": "GSR",
            "SPLIT_TO_EVAL": "eval",
            "SEQ_INFO": seq_info,
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
            "TRACKERS_TO_EVAL": trackers,
        }
    )
    metric_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }
    evaluator = trackeval.Evaluator(eval_config)
    datasets = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    # the evaluator adds the Count metric itself
    metrics = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]
    results, messages = evaluator.evaluate(datasets, metrics)
    summary = {
        "tool": "sn-trackeval",
        "version": metadata.version("sn-trackeval"),
        "python": platform.python_version(),
        "seqs": sorted(seq_info),
        "trackers": {},
    }
    for tracker in trackers:
        by_seq = results["MotChallenge2DBox"][tracker]
        summary["trackers"][tracker] = {
            "message": messages["MotChallenge2DBox"][tracker],
            "per_seq": {seq: fields(by_seq[seq]["pedestrian"]) for seq in sorted(seq_info)},
            "combined": fields(by_seq["COMBINED_SEQ"]["pedestrian"]),
        }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
