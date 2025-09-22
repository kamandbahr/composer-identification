import os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import pandas as pd

tf.keras.utils.enable_interactive_logging()

try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except Exception:
    HAS_TFA = False

DATA_DIR = '/Users/kamand/preprocessing/midi-features_2_images/mel_spectrograms'
OUTPUT_DIR = '/Users/kamand/preprocessing'
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'best_model_3')
LAST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'overfitted_model_3')

BATCH_SIZE = 16
INITIAL_EPOCHS = 100
FINE_TUNE_EPOCHS = 100
PATIENCE = 10

BACKBONE = "efficientnetv2s"
IMG_SIZE = (300, 300)

DROPOUT_RATE = 0.5
L2_REG = 0.01
UNFREEZE_AT = -40
MIXED_PRECISION = False
CONTRASTIVE_WEIGHT = 0.0

FEATURE_DIRS = {
    "mel_spectrograms": "/Users/kamand/preprocessing/midi-features_2_images/mel_spectrograms",
    "wavelet_scalograms": "/Users/kamand/preprocessing/midi-features_2_images/wavelet_scalograms",
    "spectral_contrast": "/Users/kamand/preprocessing/midi-features_2_images/spectral_contrast",
}

BACKBONES = [
    "efficientnetv2s",
    "efficientnetb3",
    "efficientnetb4",
    "inceptionresnetv2",
    "xception",
    "densenet201",
    "mobilenetv3large",
]
BACKBONE_IMG = {
    "efficientnetv2s":   (300, 300),
    "efficientnetb3":    (300, 300),
    "efficientnetb4":    (380, 380),
    "inceptionresnetv2": (299, 299),
    "xception":          (299, 299),
    "densenet201":       (224, 224),
    "mobilenetv3large":  (224, 224),
}

OUTPUT_ROOT = os.path.join(OUTPUT_DIR, "experiments_multi")

def _nowstamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class SparseLabelSmoothingCE(tf.keras.losses.Loss):
    def __init__(self, num_classes, smoothing=0.1, from_logits=True, name="sparse_label_smoothing_ce"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=from_logits, label_smoothing=smoothing
        )
    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes)
        return self.cce(y_true_oh, y_pred)

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
else:
    mixed_precision.set_global_policy('float32')
    print('Mixed precision disabled; policy set to float32.')

def plot_and_save_confusion(y_true, y_pred, labels_map, out_path):
    class_names = [labels_map[i] for i in sorted(labels_map.keys())]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()
    print('Saved confusion matrix to', out_path)

def compute_class_weights(y):
    counts = Counter(y)
    total = len(y)
    num_classes = len(counts)
    class_weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    print('Class distribution:')
    for k, v in counts.items():
        print(f'  - {k}: {v}')
    print('Class weights:', class_weights)
    return class_weights

#data loader with name of the folders
def load_image_paths(data_path):
    all_image_paths, all_labels = [], []
    composer_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if not composer_dirs:
        print("No composer directories found. Please check the data path."); return None, None, None
    for composer in composer_dirs:
        cdir = os.path.join(data_path, composer)
        for f in os.listdir(cdir):
            if f.lower().endswith('.png'):
                all_image_paths.append(os.path.join(cdir, f))
                all_labels.append(composer)
    if not all_image_paths:
        print("No .png files found in the composer directories. Please check your data.")
        return None, None, None
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(all_labels)
    int_to_label = {i: label for i, label in enumerate(encoder.classes_)}
    print(f"Found {len(all_image_paths)} images across {len(int_to_label)} composers.")
    print(f"Composers found: {list(int_to_label.values())}")
    return np.array(all_image_paths), y_int, int_to_label

def make_datasets(file_paths, y, img_size, batch_size=BATCH_SIZE, validation_split=0.2, test_split=0.1):
    paths_tmp, paths_test, y_tmp, y_test = train_test_split(
        file_paths, y, test_size=test_split, random_state=42, stratify=y
    )
    val_rel = validation_split / (1.0 - test_split)
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_tmp, y_tmp, test_size=val_rel, random_state=42, stratify=y_tmp
    )
    print(f"Samples -> train: {len(paths_train)}, val: {len(paths_val)}, test: {len(paths_test)}")

    AUTOTUNE = tf.data.AUTOTUNE
    def _load_and_preprocess(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size) 
        img = tf.cast(img, tf.float32)        
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((paths_train, y_train)).shuffle(
        buffer_size=min(len(paths_train), 2000)
    ).map(_load_and_preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((paths_val, y_val)).map(
        _load_and_preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((paths_test, y_test)).map(
        _load_and_preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, y_train, y_test

from tensorflow.keras.applications import (
    EfficientNetV2S, efficientnet_v2,
    EfficientNetB3, EfficientNetB4, efficientnet,
    InceptionResNetV2, inception_resnet_v2,
    Xception, xception,
    DenseNet201, densenet,
    MobileNetV3Large, mobilenet_v3
)

def get_backbone_and_preprocess(backbone_name, input_shape):
    if backbone_name == "efficientnetv2s":
        return EfficientNetV2S(weights="imagenet", include_top=False, input_shape=input_shape), efficientnet_v2.preprocess_input
    elif backbone_name == "efficientnetb3":
        return EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape), efficientnet.preprocess_input
    elif backbone_name == "efficientnetb4":
        return EfficientNetB4(weights="imagenet", include_top=False, input_shape=input_shape), efficientnet.preprocess_input
    elif backbone_name == "inceptionresnetv2":
        return InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape), inception_resnet_v2.preprocess_input
    elif backbone_name == "xception":
        return Xception(weights="imagenet", include_top=False, input_shape=input_shape), xception.preprocess_input
    elif backbone_name == "densenet201":
        return DenseNet201(weights="imagenet", include_top=False, input_shape=input_shape), densenet.preprocess_input
    elif backbone_name == "mobilenetv3large":
        return MobileNetV3Large(weights="imagenet", include_top=False, input_shape=input_shape), mobilenet_v3.preprocess_input
    else:
        raise ValueError("Unknown backbone")

def unfreeze_model(base_model, unfreeze_at=None):
    if unfreeze_at is None:
        for l in base_model.layers: l.trainable = True; return
    for l in base_model.layers[:unfreeze_at]:
        l.trainable = False
    for l in base_model.layers[unfreeze_at:]:
        l.trainable = True

def build_model(num_classes, backbone_name, img_size):
    base, preprocess = get_backbone_and_preprocess(backbone_name, (img_size[0], img_size[1], 3))
    inp = layers.Input(shape=(img_size[0], img_size[1], 3), name="image")
    x = preprocess(inp)                       
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    emb_pre = layers.Dense(256, activation=None, name="embeddings_dense")(x)
    emb_pre = layers.LayerNormalization(name="embeddings_ln")(emb_pre)
    emb = layers.Lambda(lambda z: tf.nn.l2_normalize(z, axis=-1), name="embeddings")(emb_pre)
    logits = layers.Dense(num_classes, activation=None, dtype='float32', name="logits")(emb)
    model = models.Model(inputs=inp, outputs=[logits, emb], name=f"{backbone_name}_composer_style")
    return model, base

def supervised_contrastive_loss(y_true, emb, temperature=0.1):
    y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.int32)
    sim = tf.matmul(emb, emb, transpose_b=True) / temperature
    matches = tf.cast(tf.equal(y_true, tf.transpose(y_true)), tf.float32)
    logits_mask = 1. - tf.eye(tf.shape(emb)[0])
    matches = matches * logits_mask
    log_prob = sim - tf.reduce_logsumexp(sim * logits_mask - 1e9*(1. - logits_mask), axis=1, keepdims=True)
    pos_count = tf.reduce_sum(matches, axis=1)
    loss_vec = -tf.reduce_sum(matches * log_prob, axis=1) / (pos_count + 1e-9)
    return tf.reduce_mean(loss_vec)

class SupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name="supcon_loss"):
        super().__init__(name=name); self.temperature = temperature
    def call(self, y_true, y_pred):
        return supervised_contrastive_loss(y_true, y_pred, self.temperature)

def run_one_feature_backbone(feature_name, data_dir, backbone_name, img_size):
    timestamp = _nowstamp()
    run_root = os.path.join(OUTPUT_ROOT, feature_name, backbone_name, timestamp)
    logs_dir = os.path.join(run_root, "logs")
    best_dir = os.path.join(run_root, "checkpoints", "best")
    last_dir = os.path.join(run_root, "checkpoints", "last")
    os.makedirs(logs_dir, exist_ok=True); os.makedirs(best_dir, exist_ok=True); os.makedirs(last_dir, exist_ok=True)

    print(f"\n=== {feature_name} / {backbone_name} @ {img_size[0]} -> {run_root}")

    file_paths, y, int_to_label = load_image_paths(data_dir)
    if file_paths is None or y is None:
        return None
    num_classes = len(np.unique(y))
    _save_json(int_to_label, os.path.join(run_root, "labels_map.json"))

    train_ds, val_ds, test_ds, y_train, y_test = make_datasets(file_paths, y, img_size, batch_size=BATCH_SIZE)

    class_weights_dict = compute_class_weights(y_train)
    cw_array = np.ones(num_classes, dtype=np.float32)
    for cls, w in class_weights_dict.items(): cw_array[int(cls)] = float(w)
    cw_tensor = tf.constant(cw_array, dtype=tf.float32)

    model, base = build_model(num_classes, backbone_name, img_size)

    if HAS_TFA:
        opt_head = tfa.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4)
        opt_ft   = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
    else:
        opt_head = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        opt_ft   = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)

    acc = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
    ce_loss = SparseLabelSmoothingCE(num_classes=num_classes, smoothing=0.1, from_logits=True)
    supcon = SupConLoss(temperature=0.1)

    for l in base.layers: l.trainable = False
    model.compile(
        optimizer=opt_head,
        loss={"logits": ce_loss, "embeddings": supcon},
        loss_weights={"logits": 1.0, "embeddings": CONTRASTIVE_WEIGHT},
        metrics={"logits": [acc]}
    )

    def add_labels_and_weights(img, label):
        w = tf.gather(cw_tensor, label)
        sw = {"logits": w, "embeddings": w}
        return (img, (label, label), sw)
    train_multi = train_ds.map(add_labels_and_weights)

    def add_labels_val(img, label):
        sw = {"logits": tf.ones((), tf.float32), "embeddings": tf.ones((), tf.float32)}
        return (img, (label, label), sw)
    val_multi = val_ds.map(add_labels_val)

    tb_cb = callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
    best_ckpt = callbacks.ModelCheckpoint(
        os.path.join(best_dir, 'best_model.weights.h5'),
        monitor='val_logits_loss', save_best_only=True, save_weights_only=True, verbose=1
    )
    last_ckpt = callbacks.ModelCheckpoint(
        os.path.join(last_dir, 'last_model_epoch_{epoch:02d}.weights.h5'),
        save_best_only=False, save_weights_only=True, verbose=0
    )
    es = callbacks.EarlyStopping(monitor='val_logits_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

    print('Stage 1: training head (frozen backbone)')
    hist1 = model.fit(train_multi, validation_data=val_multi, epochs=INITIAL_EPOCHS,
                      callbacks=[tb_cb, best_ckpt, last_ckpt, es], verbose=1)

    print('Stage 2: fine-tuning')
    if UNFREEZE_AT is None:
        for l in base.layers: l.trainable = True
    else:
        for l in base.layers[:UNFREEZE_AT]: l.trainable = False
        for l in base.layers[UNFREEZE_AT:]: l.trainable = True

    model.compile(optimizer=opt_ft,
                  loss={"logits": ce_loss, "embeddings": supcon},
                  loss_weights={"logits": 1.0, "embeddings": CONTRASTIVE_WEIGHT},
                  metrics={"logits": [acc]})

    hist2 = model.fit(train_multi, validation_data=val_multi, epochs=FINE_TUNE_EPOCHS,
                      callbacks=[tb_cb, best_ckpt, last_ckpt, es], verbose=1)

    best_path = os.path.join(best_dir, 'best_model.weights.h5')
    if os.path.exists(best_path):
        model.load_weights(best_path)
        print('Loaded best model weights from', best_path)

    def images_only(ds): return ds.map(lambda *args: args[0])
    print('Evaluating on test set (mixed clean + augmented)...')
    y_logits, _ = model.predict(images_only(test_ds), verbose=0)
    y_pred = np.argmax(y_logits, axis=1)

    print('Classification Report:')
    target_names = [int_to_label[i] for i in range(len(int_to_label))]
    report_txt = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report_txt)
    with open(os.path.join(run_root, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    cm_path = os.path.join(run_root, 'confusion_matrix.png')
    plot_and_save_confusion(y_test, y_pred, int_to_label, cm_path)

    final_model_path = os.path.join(run_root, 'final_model.keras')
    model.save(final_model_path); print('Saved final model to', final_model_path)

    embed_model = models.Model(model.input, model.get_layer("embeddings").output)
    embed_model_path = os.path.join(run_root, 'embedding_model.keras')
    embed_model.save(embed_model_path); print('Saved embedding model to', embed_model_path)

    hist_merged = {}
    for h in [hist1.history, hist2.history]:
        for k, v in h.items(): hist_merged.setdefault(k, []).extend(v)
    pd.DataFrame(hist_merged).to_csv(os.path.join(run_root, "history.csv"), index=False)

    top1 = float((y_pred == y_test).mean())
    return {
        "feature": feature_name,
        "backbone": backbone_name,
        "img_size": f"{img_size[0]}x{img_size[1]}",
        "test_top1_acc": top1,
        "run_dir": run_root
    }

def main():
    summaries = []
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for feat, feat_dir in FEATURE_DIRS.items():
        if not feat_dir or not os.path.isdir(feat_dir):
            print(f"⚠️  Skipping {feat} (missing dir: {feat_dir})")
            continue

        for bb in BACKBONES:
            img_size = BACKBONE_IMG[bb]
            try:
                row = run_one_feature_backbone(feat, feat_dir, bb, img_size)
                if row: summaries.append(row)
            except Exception as e:
                err_dir = os.path.join(OUTPUT_ROOT, "errors"); os.makedirs(err_dir, exist_ok=True)
                err_path = os.path.join(err_dir, f"error_{feat}_{bb}_{_nowstamp()}.txt")
                with open(err_path, "w") as f: f.write(str(e))
                print(f"❌ Failed {feat}/{bb}: {e}. Logged to {err_path}")

    if summaries:
        sum_csv = os.path.join(OUTPUT_ROOT, f"summary_{_nowstamp()}.csv")
        pd.DataFrame(summaries).to_csv(sum_csv, index=False)
        print(f"\n=== Summary saved to {sum_csv} ===")
        for r in summaries:
            print(f"{r['feature']:>18} | {r['backbone']:<16} | acc={r['test_top1_acc']:.4f} | {r['run_dir']}")
    else:
        print("No successful runs to summarize.")

if __name__ == '__main__':
    main()
