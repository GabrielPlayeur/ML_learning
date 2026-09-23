import os
import math
import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_features(dataset_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    mfcc_list, label_list, track_id_list = [], [], []
    label_mapping = {}

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    exp_frames = math.ceil(num_samples_per_segment / hop_length)

    for dirpath, _, filenames in os.walk(dataset_path):
        if dirpath == dataset_path:
            continue

        genre = os.path.basename(dirpath)
        if genre not in label_mapping:
            label_mapping[genre] = len(label_mapping)
        label = label_mapping[genre]
        print(f"\nProcessing: {genre}")

        for fname in filenames:
            file_path = os.path.join(dirpath, fname)
            try:
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue

            track_id = os.path.relpath(file_path, dataset_path)

            for s in range(num_segments):
                start = num_samples_per_segment * s
                segment = signal[start:start + num_samples_per_segment]

                mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                features = np.concatenate([mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2),], axis=0).T
                # shape: (frames, 3*n_mfcc)
                # features = mfcc.T

                if len(features) == exp_frames:
                    mfcc_list.append(features)
                    label_list.append(label)
                    track_id_list.append(track_id)

    return (
        np.array(mfcc_list, dtype=np.float32),
        np.array(label_list, dtype=np.int64),
        np.array(track_id_list),
        label_mapping,
    )


def save_npz(output_path, mfcc, labels, track_ids, label_mapping):
    if not os.path.isfile(output_path):
        with open(output_path, 'a'):
            os.utime(output_path, None)
    mapping_array = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    np.savez_compressed(
        output_path,
        mfcc=mfcc,
        labels=labels,
        track_ids=track_ids,
        mapping=np.array(mapping_array),
    )

if __name__ == "__main__":
    mfcc, labels, track_ids, mapping = extract_features("dataset/genres_original", num_segments=10)
    save_npz("dataset/mfcc.npz", mfcc, labels, track_ids, mapping)