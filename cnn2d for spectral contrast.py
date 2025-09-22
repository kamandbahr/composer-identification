import os
import numpy as np
import random
import tensorflow as tf
import keras
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers


# Check for GPU availability
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU is available and will be used for training.")
    else:
        print("❌ No GPU is available. Training will be slow on the CPU.")
except Exception as e:
    print(f"Error checking for GPU: {e}")
    print("Assuming no GPU is available.")

MFCC_DIR = '/Users/kamand/preprocessing/mfcc_features'
SC_DIR = '/Users/kamand/preprocessing/spectral_contrast_features'
max_time = 216
num_mfcc_features = 20
num_sc_features = 7
combined_features = num_mfcc_features + num_sc_features

# Composers to remove for clean data
COMPOSERS_TO_REMOVE = [
    'Balakirev', 'Bendel', 'Cui', 'Donizetti', 'Dussek',
    'Ferrari', 'Fries', 'Giuliani', 'Glazunov', 'Glinka'
]

# Function to apply the new grouping logic
def get_composer_group(composer):
    if composer in ['Beethoven', "Alkan"]:
        return 'Beethoven_Alkan'
    if composer in ['Mozart', 'Liszt']:
        return 'Mozart_Liszt'
    if composer in ['Chopin', 'Rachmaninoff']:
        return 'Chopin_Rachmaninoff'
    return composer

x_data = []
y_data_str = []

print("Loading and combining MFCC and Spectral Contrast data...")

file_names = [f for f in os.listdir(MFCC_DIR) if f.endswith('.npy')]

for filename in file_names:
    composer_name = filename.split('_')[0]
    
    if composer_name in COMPOSERS_TO_REMOVE:
        continue
    
    group_label = get_composer_group(composer_name)
    
    mfcc_path = os.path.join(MFCC_DIR, filename)
    sc_path = os.path.join(SC_DIR, filename)
    
    if not os.path.exists(mfcc_path) or not os.path.exists(sc_path):
        continue
        
    mfcc_source = np.load(mfcc_path).T
    sc_source = np.load(sc_path).T
    
    mfcc_padded = np.zeros((max_time, num_mfcc_features))
    sc_padded = np.zeros((max_time, num_sc_features))
    
    length_mfcc = min(mfcc_source.shape[0], max_time)
    length_sc = min(sc_source.shape[0], max_time)
    
    mfcc_padded[:length_mfcc] = mfcc_source[:length_mfcc]
    sc_padded[:length_sc] = sc_source[:length_sc]
    
    combined_padded = np.hstack((mfcc_padded, sc_padded))
    
    x_data.append(combined_padded)
    y_data_str.append(group_label)

X = np.array(x_data)[..., np.newaxis]
y_str = np.array(y_data_str)

counts = Counter(y_str)
valid_labels = {label for label, count in counts.items() if count >= 10}

filtered = [(x, label) for x, label in zip(X, y_str) if label in valid_labels]
if filtered:
    X, y_str = zip(*filtered)
    X = np.array(X)
    y_str = np.array(y_str)

print("\nFiltered label distribution:")
counts = Counter(y_str)
for label, count in sorted(counts.items()):
    print(f"  {label}: {count}")

label_encoder = LabelEncoder()
if len(y_str) > 0:
    y = label_encoder.fit_transform(y_str)
    num_classes = len(np.unique(y))
else:
    num_classes = 0
    print("⚠️ No valid classes left to train a model.")
    exit()

if num_classes > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.30, random_state=42, stratify=y_train
    )
else:
    print("⚠️ Not enough classes to split the dataset for training.")
    exit()

# === STEP 5: Normalize and add Positional Encoding ===
def add_positional_encoding(data_tensor, max_time, num_features):
    """Adds positional encoding to a data tensor."""
    pe = np.zeros((max_time, num_features, 1))
    for i in range(max_time):
        for j in range(num_features):
            pe[i, j, 0] = np.sin(i / 10000**(2 * j / num_features))
    
    pe = (pe - np.mean(pe)) / np.std(pe)
    pe_tiled = np.tile(pe[np.newaxis, ...], [data_tensor.shape[0], 1, 1, 1])
    
    return data_tensor + pe_tiled

if len(X_train) > 0:
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std

    X_train = add_positional_encoding(X_train, max_time, combined_features)
    X_valid = add_positional_encoding(X_valid, max_time, combined_features)
    X_test = add_positional_encoding(X_test, max_time, combined_features)
else:
    mean = std = 0

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_valid_cat = keras.utils.to_categorical(y_valid, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# === STEP 7: Hybrid SeparableConv2D Model (Corrected) ===
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def build_hybrid_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.SeparableConv2D(128, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.SeparableConv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling to create the embedding vector
    # This output is a 1D vector of shape (None, 512)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Connect the global pooling output directly to the final classification head
    # The intermediate dense layer is also removed for simplicity and stability
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='hybrid_model')
    return model

model = build_hybrid_model(input_shape=(max_time, combined_features, 1), num_classes=num_classes)

model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_hybrid_features_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_valid, y_valid_cat),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

embedding_model = keras.Model(inputs=model.input, outputs=model.layers[-3].output)
embeddings = embedding_model.predict(X_test)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=tsne_embeddings[:, 0],
    y=tsne_embeddings[:, 1],
    hue=y_test,
    palette=sns.color_palette("hsv", num_classes),
    legend="full"
)
plt.title("t-SNE Embeddings of Hybrid Model")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("tsne_embeddings_hybrid_model.png")
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

print("\nClassification Report:")
present_labels = np.unique(y_test)
present_class_names = [label_encoder.classes_[i] for i in present_labels]
print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_class_names, zero_division=0))

conf_matrix = confusion_matrix(y_test, y_pred, labels=present_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=present_class_names,
            yticklabels=present_class_names,
            cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Hybrid Features, Advanced CNN)")
plt.tight_layout()
plt.savefig("confusion_matrix_hybrid_advanced.png")
plt.show()

model.save("composer_hybrid_advanced_model.keras")
