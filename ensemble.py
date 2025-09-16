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
from typing import Dict, List, Tuple



DATA_DIRS = {
    'mel': '/Users/kamand/preprocessing/midi-features_2_images/mel_spectrograms',
    'wavelet': '/Users/kamand/preprocessing/midi-features_2_images/wavelet_scalograms',
    'speccon': '/Users/kamand/preprocessing/midi-features_2_images/spectral_contrast',
}

OUTPUT_DIR = '/Users/kamand/preprocessing'
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs_ensemble')
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'best_model_ensemble')
LAST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'overfitted_model_ensemble')

BATCH_SIZE = 16
INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 50
PATIENCE = 7
IMG_SIZE = (224, 224)
DROPOUT_RATE = 0.5
L2_REG = 0.01
UNFREEZE_AT = -30   
MIXED_PRECISION = True
SEED = 42

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(LAST_MODEL_DIR, exist_ok=True)

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


def index_dataset(root: str) -> Dict[str, str]:
    """Return dict mapping relative key (composer/filename) -> absolute path for a root."""
    out = {}
    if not os.path.isdir(root):
        print(f"[WARN] Root not found: {root}")
        return out
    for composer in sorted(os.listdir(root)):
        comp_dir = os.path.join(root, composer)
        if not os.path.isdir(comp_dir):
            continue
        for f in os.listdir(comp_dir):
            if f.endswith('.png'):
                key = os.path.join(composer, f)
                out[key] = os.path.join(comp_dir, f)
    return out


def load_paired_paths(data_dirs: Dict[str, str]):
    """
    Build aligned lists of (mel_path, wavelet_path, speccon_path) with shared labels from directory names.
    Skips any sample missing in any modality. Returns: list_of_triplets, labels_int, int_to_label.
    """
    mel_idx = index_dataset(data_dirs['mel'])
    wav_idx = index_dataset(data_dirs['wavelet'])
    spc_idx = index_dataset(data_dirs['speccon'])

    keys_intersection = set(mel_idx.keys()) & set(wav_idx.keys()) & set(spc_idx.keys())
    missing = {
        'mel_only': len(set(mel_idx) - keys_intersection),
        'wavelet_only': len(set(wav_idx) - keys_intersection),
        'speccon_only': len(set(spc_idx) - keys_intersection),
    }
    if any(missing.values()):
        print('[INFO] Skipping unmatched samples ->', missing)

    triplets: List[Tuple[str, str, str]] = []
    labels: List[str] = []
    for key in sorted(keys_intersection):
        composer = os.path.dirname(key)
        triplets.append((mel_idx[key], wav_idx[key], spc_idx[key]))
        labels.append(composer)

    if not triplets:
        print('[ERROR] No paired samples found. Check directory structures and filenames.')
        return None, None, None

    encoder = LabelEncoder()
    y_int = encoder.fit_transform(labels)
    int_to_label = {i: label for i, label in enumerate(encoder.classes_)}

    print(f"Found {len(triplets)} paired triplets across {len(int_to_label)} composers.")
    print(f"Composers: {list(int_to_label.values())}")
    return triplets, y_int, int_to_label


def make_datasets(triplets, y, batch_size=BATCH_SIZE, validation_split=0.2, test_split=0.1):
    paths_tmp, paths_test, y_tmp, y_test = train_test_split(
        triplets, y, test_size=test_split, random_state=SEED, stratify=y
    )
    val_rel = validation_split / (1.0 - test_split)
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_tmp, y_tmp, test_size=val_rel, random_state=SEED, stratify=y_tmp
    )

    print(f"Samples -> train: {len(paths_train)}, val: {len(paths_val)}, test: {len(paths_test)}")

    AUTOTUNE = tf.data.AUTOTUNE

    def _load_img(fp: tf.Tensor):
        img = tf.io.read_file(fp)
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.resize(tf.cast(img, tf.float32), IMG_SIZE)
        return img

    def _load_and_preprocess(mel_fp, wav_fp, spc_fp, label):
        mel = _load_img(mel_fp)
        wav = _load_img(wav_fp)
        spc = _load_img(spc_fp)

        mel = resnet50.preprocess_input(mel)
        wav = resnet50.preprocess_input(wav)
        spc = (spc / 127.5) - 1.0
        return (mel, wav, spc), label

    def build_ds(paths, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(paths), 2048), seed=SEED)
        ds = ds.map(lambda t, y: (_load_and_preprocess(t[0], t[1], t[2], y)), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = build_ds(paths_train, y_train, shuffle=True)
    val_ds = build_ds(paths_val, y_val, shuffle=False)
    test_ds = build_ds(paths_test, y_test, shuffle=False)

    return train_ds, val_ds, test_ds, y_train, y_test



def resnet_branch(name: str, l2_reg=L2_REG, dropout=DROPOUT_RATE):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name=f"{name}_resnet50")
    for layer in base.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(base.output)
    x = layers.Dropout(dropout, name=f"{name}_do1")(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name=f"{name}_fc1")(x)
    x = layers.Dropout(dropout, name=f"{name}_do2")(x)
    return base.input, x, base


def sepconv_block(x, filters, kernel, strides=1, name_prefix="sc"):  
    x = layers.SeparableConv2D(filters, kernel, strides=strides, padding='same', use_bias=False, name=f"{name_prefix}_sepconv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu")(x)
    return x


def spectral_contrast_branch(l2_reg=L2_REG, dropout=DROPOUT_RATE):
    """Custom 2D-CNN tuned for time–frequency images (lightweight, strong regularization)."""
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='speccon_input')
    x = inp
    x = sepconv_block(x, 64,  (7, 7), strides=2, name_prefix='spc_b1')
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='spc_pool1')(x)

    x = sepconv_block(x, 128, (3, 3), name_prefix='spc_b2_1')
    x = sepconv_block(x, 128, (3, 3), name_prefix='spc_b2_2')
    x = layers.MaxPooling2D((2, 2), name='spc_pool2')(x)

    x = sepconv_block(x, 256, (3, 3), name_prefix='spc_b3_1')
    x = sepconv_block(x, 256, (3, 3), name_prefix='spc_b3_2')
    x = layers.SpatialDropout2D(0.2, name='spc_sd2')(x)
    x = layers.MaxPooling2D((2, 2), name='spc_pool3')(x)

    x = sepconv_block(x, 384, (3, 3), name_prefix='spc_b4_1')
    x = layers.GlobalAveragePooling2D(name='spc_gap')(x)
    x = layers.Dropout(dropout, name='spc_do1')(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='spc_fc1')(x)
    x = layers.Dropout(dropout, name='spc_do2')(x)
    return inp, x


def build_ensemble(num_classes):
    mel_inp, mel_feat, mel_base = resnet_branch('mel')
    wav_inp, wav_feat, wav_base = resnet_branch('wav')
    spc_inp, spc_feat = spectral_contrast_branch()

    fused = layers.Concatenate(name='fusion_concat')([mel_feat, wav_feat, spc_feat])
    fused = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='fusion_fc1')(fused)
    fused = layers.Dropout(DROPOUT_RATE, name='fusion_do1')(fused)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='classifier')(fused)

    model = models.Model(inputs=[mel_inp, wav_inp, spc_inp], outputs=outputs, name='Ensemble_Mel_Wavelet_SpecCon')
    bases = {'mel': mel_base, 'wav': wav_base}
    return model, bases


def unfreeze_backbones(bases: Dict[str, tf.keras.Model], unfreeze_at=UNFREEZE_AT):
    for name, base in bases.items():
        if unfreeze_at is None:
            for layer in base.layers:
                layer.trainable = True
        else:
            for layer in base.layers[:unfreeze_at]:
                layer.trainable = False
            for layer in base.layers[unfreeze_at:]:
                layer.trainable = True
        print(f"[INFO] Unfreeze policy applied to {name} backbone (unfreeze_at={unfreeze_at})")



def plot_and_save_confusion(y_true, y_pred, labels_map, out_path):
    class_names = [labels_map[i] for i in sorted(labels_map.keys())]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (Ensemble)')
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
    triplets, y, int_to_label = load_paired_paths(DATA_DIRS)
    if triplets is None:
        return

    num_classes = len(np.unique(y))
    train_ds, val_ds, test_ds, y_train, y_test = make_datasets(triplets, y, batch_size=BATCH_SIZE)
    class_weights = calculate_class_weights(y_train)

    model, bases = build_ensemble(num_classes)

#   BackBone stage 1 : 
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_cb = callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, timestamp), histogram_freq=1)
    best_ckpt = callbacks.ModelCheckpoint(os.path.join(BEST_MODEL_DIR, 'best_ensemble.weights.h5'), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    last_ckpt = callbacks.ModelCheckpoint(os.path.join(LAST_MODEL_DIR, 'last_ensemble_epoch_{epoch:02d}.weights.h5'), save_best_only=False, save_weights_only=True, verbose=0)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

    print('Stage 1: training fusion head (frozen ResNet backbones)')
    model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, class_weight=class_weights, callbacks=[tb_cb, best_ckpt, last_ckpt, es])

#   Fine-Tune stage 2 : 
    print('Stage 2: fine-tuning backbones')
    unfreeze_backbones(bases, unfreeze_at=UNFREEZE_AT)
    opt_finetune = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
    model.compile(optimizer=opt_finetune, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, class_weight=class_weights, callbacks=[tb_cb, best_ckpt, last_ckpt, es])

#   two ways of getting the data, you can train the model from scratch
#   or with this part you can load the weights
    best_path = os.path.join(BEST_MODEL_DIR, 'best_ensemble.weights.h5')
    if os.path.exists(best_path):
        model.load_weights(best_path)
        print('Loaded best ensemble weights from', best_path)

#   evaluatation stage 3 : 
    print('Evaluating on test set...')
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    print('Classification Report (Ensemble):')
    print(classification_report(y_test, y_pred, target_names=[int_to_label[i] for i in sorted(int_to_label.keys())]))

    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_ensemble.png')
    plot_and_save_confusion(y_test, y_pred, int_to_label, cm_path)

#   save final model stage 4 : 
    final_model_path = os.path.join(OUTPUT_DIR, 'final_ensemble_model.keras')
    model.save(final_model_path)
    print('Saved final ensemble model to', final_model_path)
    print('Done. Use TensorBoard to inspect training logs:')
    print(f"tensorboard --logdir={LOGS_DIR}")


if __name__ == '__main__':
    main()
