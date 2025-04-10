import os
import random
import math
from PIL import Image
import numpy as np

def extract_binary_features(image_path, size=(16, 16), threshold=128):
    image = Image.open(image_path).convert("L").resize(size)
    data = np.array(image)
    binary = (data > threshold).astype(int)
    return binary.flatten().tolist()

def extract_gaussian_features(image_path, size=(16, 16)):
    image = Image.open(image_path).convert("L").resize(size)
    data = np.array(image)
    return data.flatten().tolist()

def load_dataset(dataset_path, mode='binary'):
    data = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path): continue
        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(label_path, filename)
                if mode == 'binary':
                    features = extract_binary_features(path)
                else:
                    features = extract_gaussian_features(path)
                data.append({'label': label, 'features': features, 'filename': filename})
    return data

def train_naive_bayes_binary(data):
    model = {}
    total_per_class = {}
    for sample in data:
        label = sample['label']
        if label not in model:
            model[label] = {'counts': [ [0, 0] for _ in range(len(sample['features'])) ]}
            total_per_class[label] = 0
        total_per_class[label] += 1
        for i, val in enumerate(sample['features']):
            model[label]['counts'][i][val] += 1
    model['total'] = sum(total_per_class.values())
    model['per_class'] = total_per_class
    return model

def predict_binary(model, features):
    max_prob = -math.inf
    best_label = None
    for label, data in model.items():
        if label in ('total', 'per_class'): continue
        log_prob = math.log(model['per_class'][label] / model['total'])
        for i, val in enumerate(features):
            count_0, count_1 = data['counts'][i]
            total = count_0 + count_1
            prob = (count_1 if val == 1 else count_0) / total if total != 0 else 0.5
            log_prob += math.log(prob + 1e-6)
        if log_prob > max_prob:
            max_prob = log_prob
            best_label = label
    return best_label

def train_naive_bayes_gaussian(data):
    model = {}
    class_counts = {}
    for sample in data:
        label = sample['label']
        if label not in model:
            model[label] = {'features': []}
            class_counts[label] = 0
        class_counts[label] += 1
    num_features = len(data[0]['features'])
    for label in model:
        model[label]['features'] = [[0.0, 0.0] for _ in range(num_features)]
    for sample in data:
        label = sample['label']
        for i, val in enumerate(sample['features']):
            model[label]['features'][i][0] += val
            model[label]['features'][i][1] += val**2
    for label in model:
        count = class_counts[label]
        for i in range(num_features):
            mean = model[label]['features'][i][0] / count
            sq_sum = model[label]['features'][i][1]
            std = math.sqrt((sq_sum / count) - (mean ** 2))
            if std == 0:
                std = 1e-6
            model[label]['features'][i] = (mean, std)
    model['total'] = sum(class_counts.values())
    model['per_class'] = class_counts
    return model

def gaussian_prob(x, mean, std):
    exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

def predict_gaussian(model, features):
    max_prob = -math.inf
    best_label = None
    for label, data in model.items():
        if label in ('total', 'per_class'): continue
        log_prob = math.log(model['per_class'][label] / model['total'])
        for i, val in enumerate(features):
            mean, std = data['features'][i]
            prob = gaussian_prob(val, mean, std)
            log_prob += math.log(prob + 1e-6)
        if log_prob > max_prob:
            max_prob = log_prob
            best_label = label
    return best_label

def evaluate_model(model, test_data, predict_func):
    benar = 0
    for i, sample in enumerate(test_data):
        pred = predict_func(model, sample['features'])
        if pred == sample['label']:
            benar += 1
    return benar / len(test_data) * 100

if __name__ == "__main__":
    dataset_path = "dataset"
    data_binary = load_dataset(dataset_path, mode='binary')
    data_gaussian = load_dataset(dataset_path, mode='gaussian')

    random.shuffle(data_binary)
    random.shuffle(data_gaussian)

    split_binary = int(0.8 * len(data_binary))
    split_gaussian = int(0.8 * len(data_gaussian))

    train_binary = data_binary[:split_binary]
    test_binary = data_binary[split_binary:]

    train_gaussian = data_gaussian[:split_gaussian]
    test_gaussian = data_gaussian[split_gaussian:]

    print("Training Binary Naive Bayes...")
    model_binary = train_naive_bayes_binary(train_binary)
    acc_binary = evaluate_model(model_binary, test_binary, predict_binary)
    print(f"Akurasi Binary Naive Bayes: {acc_binary:.2f}%\n")

    print("Training Gaussian Naive Bayes...")
    model_gaussian = train_naive_bayes_gaussian(train_gaussian)
    acc_gaussian = evaluate_model(model_gaussian, test_gaussian, predict_gaussian)
    print(f"Akurasi Gaussian Naive Bayes: {acc_gaussian:.2f}%")
