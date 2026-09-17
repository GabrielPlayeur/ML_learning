import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import json

DATASET_PATH = "data/genres_original"
JSON_PATH = "data/data.json"

SAMPLE_RATE = 22050
DURATION = 30 # in second
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [], # list of all output values
        "mfcc": [], #list of mfcc
        "labels": [], #list of the keys values in the mapping list
        "track_ids": [] #id of the track come from
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    exp_num_mfcc_vec_per_segment = math.ceil(num_samples_per_segment/hop_length)

    label_mapping = {}

    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is dataset_path:
            continue

        semantic_label = dirpath.split("/")[-1]
        data["mapping"].append(semantic_label)
        print(f"\n- Processing {semantic_label}")

        for f in filenames:
            file_path = os.path.join(dirpath, f)

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"  ❗️ Skipping corrupted file {file_path}\n")
                continue

            # process segment
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                semantic_label = os.path.basename(dirpath)
                if semantic_label not in label_mapping:
                    label_mapping[semantic_label] = len(label_mapping)
                    data["mapping"].append(semantic_label)
                label = label_mapping[semantic_label]

                sample = signal[start_sample: finish_sample]
                mfcc = librosa.feature.mfcc(y=sample,
                                            sr=SAMPLE_RATE,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length
                                            )
                mfcc = mfcc.T

                track_id = os.path.relpath(file_path, dataset_path)

                if len(mfcc) == exp_num_mfcc_vec_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(label)
                    data["track_ids"].append(track_id)
                    print(f"{file_path}: segment {s+1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)