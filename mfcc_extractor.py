import json
import math
import os
import librosa
import soundfile as sf

DATASET_PATH = "D:\Projects\CNN_genre_classificator\dataset\Data\genres_original"
JSON_PATH = "dataset/Data/data.json"

DURATION = 30  # in seconds, knowing the dataset
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def is_valid_wav(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            return f.format == 'WAV'  # Ensures it's a real WAV file
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return False


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segments = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at root level
        if dirpath is not DATASET_PATH:

            # save the semantic label
            dirpath_components = dirpath.split("/")  # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                if not f.lower().endswith(".wav"):
                    print(f"Skipping {f}: Not a WAV file")
                    continue

                file_path = os.path.join(dirpath, f)

                # Check if it's a valid WAV
                if not is_valid_wav(file_path):
                    continue  # Skip invalid files

                # Load the file safely
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segments:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment: {}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
