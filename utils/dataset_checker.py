import os


def check_dataset(dataset_path="D:/Projects/CNN_genre_classificator/dataset"):
    """
    Checks if the dataset directory exists and prints its contents.
    """
    if os.path.exists(dataset_path):
        print(f"✅  Dataset directory found: {dataset_path}")

        # List contents of the dataset directory
        dataset_contents = os.listdir(dataset_path)
        print("📂 Contents:", dataset_contents)

        # Check if the 'data' folder exists inside
        data_path = os.path.join(dataset_path, "data")
        if os.path.exists(data_path):
            print(f"✅ 'data' folder found inside: {data_path}")

            # Show first 5 WAV files
            # wav_files = [f for f in os.listdir(data_path) if f.endswith(".wav")]
            # print(f"🎵 Found {len(wav_files)} WAV files. Listing first 5:", wav_files[:5])
        else:
            print("⚠️ 'data' folder is missing inside the dataset directory.")
    else:
        print("❌ Dataset directory not found. Please check the path.")

