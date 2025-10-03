import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from tensorflow.keras.applications import ResNet50, resnet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
# how to extract and use composer's fingerprint ? 
# Kamand

DATA_DIR = '/Users/kamand/preprocessing/midi-features_2_images/mel_spectrograms'
OUTPUT_DIR = '/Users/kamand/preprocessing'
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs_melspectrogram')
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'best_model_melspectrogram')
LAST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'overfitted_model_melspectrogram')

BATCH_SIZE = 16          
INITIAL_EPOCHS = 50      
FINE_TUNE_EPOCHS = 50    
PATIENCE = 7
IMG_SIZE = (224, 224)
DROPOUT_RATE = 0.5
L2_REG = 0.01
UNFREEZE_AT = -30        
MIXED_PRECISION = True   

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(LAST_MODEL_DIR, exist_ok=True)

# conda deactive
# conda activate tf-new-final ---> my environment
# pip install tensorflow-metal

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print("Could not set memory growth:", e)
else:
    print("No GPUs found or visible to TensorFlow.")

if MIXED_PRECISION:
    try:
        mixed_precision.set_global_policy('mixed_float16')
        print('Mixed precision policy set to:', mixed_precision.global_policy())
    except Exception as e:
        print('Could not enable mixed precision:', e)


def load_image_paths(data_path):
    """
    Finds all .png file paths from composer subdirectories and extracts labels.
    Returns a list of file paths and a list of integer-encoded labels.
    """
    all_image_paths = []
    all_labels = []

    composer_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if not composer_dirs:
        print("No composer directories found. Please check the data path.")
        return None, None, None
    for composer in composer_dirs:
        composer_dir = os.path.join(data_path, composer)
        image_files = [f for f in os.listdir(composer_dir) if f.endswith('.png')]

        for image_file in image_files:
            all_image_paths.append(os.path.join(composer_dir, image_file))
            all_labels.append(composer)

    if not all_image_paths:
        print("No .png files found in the composer directories. Please check your data.")
        return None, None, None
    
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(all_labels)
    int_to_label = {i: label for i, label in enumerate(encoder.classes_)}
    
    print(f"Found {len(all_image_paths)} images across {len(int_to_label)} composers.")
    print(f"Composers found: {list(int_to_label.values())}")
    
    return all_image_paths, y_int, int_to_label


def make_datasets(file_paths, y, batch_size=BATCH_SIZE, validation_split=0.2, test_split=0.1):
    """
    Creates TensorFlow datasets from image file paths to load data on-the-fly.
    """

    paths_tmp, paths_test, y_tmp, y_test = train_test_split(file_paths, y, test_size=test_split, random_state=42, stratify=y)
    val_rel = validation_split / (1.0 - test_split)
    paths_train, paths_val, y_train, y_val = train_test_split(paths_tmp, y_tmp, test_size=val_rel, random_state=42, stratify=y_tmp)

    print(f"Samples -> train: {len(paths_train)}, val: {len(paths_val)}, test: {len(paths_test)}")

    AUTOTUNE = tf.data.AUTOTUNE

    def _load_and_preprocess(file_path, label):
        img = tf.io.read_file(file_path)

        img = tf.io.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32)

        img = tf.image.resize(img, IMG_SIZE)
     
        img = resnet50.preprocess_input(img)
        
        return img, label
    train_ds = tf.data.Dataset.from_tensor_slices((paths_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=min(len(paths_train), 1000))
    train_ds = train_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((paths_val, y_val))
    val_ds = val_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((paths_test, y_test))
    test_ds = test_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, y_train, y_test



def build_model(num_classes, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG):

    base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    for layer in base.layers:
        layer.trainable = False

    x = base.output
    # here we can add the trainable layers for the first phase, first phase can have multiple layers but we should take care of the dropouts
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model, base


def unfreeze_model(base_model, unfreeze_at=UNFREEZE_AT):
    if unfreeze_at is None:
        for layer in base_model.layers:
            layer.trainable = True
    else:
        for layer in base_model.layers[:unfreeze_at]:
            layer.trainable = False
        for layer in base_model.layers[unfreeze_at:]:
            layer.trainable = True



def plot_and_save_confusion(y_true, y_pred, labels_map, out_path):
    class_names = [labels_map[i] for i in sorted(labels_map.keys())]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved confusion matrix to', out_path)


def calculate_class_weights(y):
    counts = Counter(y)
    total = len(y)
    num_classes = len(counts)
    class_weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    print('Class distribution:')
    for k, v in counts.items():
        print(f'  - {k}: {v}')
    return class_weights


def main():
    file_paths, y, int_to_label = load_image_paths(DATA_DIR)
    
    if file_paths is None or y is None:
        return

    num_classes = len(np.unique(y))

    train_ds, val_ds, test_ds, y_train, y_test = make_datasets(file_paths, y, batch_size=BATCH_SIZE)
    
    class_weights = calculate_class_weights(y_train)

    model, base = build_model(num_classes)
    
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_cb = callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, timestamp), histogram_freq=1)
    best_ckpt = callbacks.ModelCheckpoint(os.path.join(BEST_MODEL_DIR, 'best_model.weights.h5'), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    last_ckpt = callbacks.ModelCheckpoint(os.path.join(LAST_MODEL_DIR, 'last_model_epoch_{epoch:02d}.weights.h5'), save_best_only=False, save_weights_only=True, verbose=0)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

    print('Stage 1: training head (frozen ResNet backbone)')
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, class_weight=class_weights, callbacks=[tb_cb, best_ckpt, last_ckpt, es])

    print('Stage 2: fine-tuning')
    unfreeze_model(base, unfreeze_at=UNFREEZE_AT)
    opt_finetune = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
    model.compile(optimizer=opt_finetune, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, class_weight=class_weights, callbacks=[tb_cb, best_ckpt, last_ckpt, es])

    best_path = os.path.join(BEST_MODEL_DIR, 'best_model.weights.h5')
    if os.path.exists(best_path):
        model.load_weights(best_path)
        print('Loaded best model weights from', best_path)

    print('Evaluating on test set...')
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=[int_to_label[i] for i in sorted(int_to_label.keys())]))

    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plot_and_save_confusion(y_test, y_pred, int_to_label, cm_path)

    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.keras')
    model.save(final_model_path)
    print('Saved final model to', final_model_path)

    print('Done. Use TensorBoard to inspect training logs:')
    print(f"tensorboard --logdir={LOGS_DIR}")


if __name__ == '__main__':
    main()
