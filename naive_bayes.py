import os
import math
from PIL import Image
import random

# === STEP 1: Ekstraksi fitur dari gambar ===
def extract_features(image_path, size=(8, 8)):
    image = Image.open(image_path).convert('L')  # ubah ke grayscale
    image = image.resize(size)
    pixels = list(image.getdata())  # list of pixel values
    return pixels

# === STEP 2: Load dataset dari folder ===
def load_dataset(dataset_path):
    data = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path): continue
        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(label_path, filename)
                features = extract_features(image_path)
                data.append({'label': label, 'features': features})
    return data

# === STEP 3: Split manual train/test ===
def split_data(data, split_ratio=0.8):
    random.shuffle(data)
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]

# === STEP 4: Hitung mean dan stddev ===
def mean(values):
    return sum(values) / len(values)

def stddev(values, mean_val):
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

# === STEP 5: Training - rangkum per kelas ===
def summarize_by_class(dataset):
    summaries = {}
    separated = {}

    for row in dataset:
        label = row['label']
        if label not in separated:
            separated[label] = []
        separated[label].append(row['features'])

    for label, features_list in separated.items():
        summaries[label] = []
        for i in range(len(features_list[0])):
            col = [row[i] for row in features_list]
            mean_val = mean(col)
            std_val = stddev(col, mean_val)
            summaries[label].append((mean_val, std_val))
    return summaries

# === STEP 6: Gaussian probability ===
def gaussian_prob(x, mean, std):
    if std == 0:
        return 1 if x == mean else 0
    exp = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp

# === STEP 7: Prediksi satu input ===
def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for label, class_summaries in summaries.items():
        probabilities[label] = 1
        for i in range(len(class_summaries)):
            mean, std = class_summaries[i]
            x = input_vector[i]
            prob = gaussian_prob(x, mean, std)
            probabilities[label] *= prob
    return probabilities

def predict(summaries, input_vector):
    probs = calculate_class_probabilities(summaries, input_vector)
    return max(probs, key=probs.get)

# === STEP 8: Jalankan semua ===
if __name__ == "__main__":
    dataset_path = "dataset"
    data = load_dataset(dataset_path)

    if len(data) < 2:
        print("Dataset terlalu sedikit ngab. Tambahin beberapa gambar lagi.")
        exit()

    train_data, test_data = split_data(data)
    model = summarize_by_class(train_data)

    print(f"Jumlah data train: {len(train_data)}, test: {len(test_data)}\n")
    benar = 0
    log_lines = []

    for i, test in enumerate(test_data):
        pred = predict(model, test['features'])
        actual = test['label']
        if pred == actual:
            benar += 1
        line = f"[{i+1}] Asli: {actual:10s} → Prediksi: {pred:10s}"
        print(line)
        log_lines.append(line)

    akurasi = benar / len(test_data) * 100
    akurasi_line = f"\n🎯 Akurasi: {akurasi:.2f}%"
    print(akurasi_line)
    log_lines.append(akurasi_line)

    # Simpan ke file hasil_prediksi.txt
    with open("hasil_prediksi.txt", "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")