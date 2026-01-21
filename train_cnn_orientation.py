import os
import numpy as np
import tensorflow as tf
import cv2

# SETTINGS 

DATA_DIR = r"C:\\Users\\anakh\\Desktop\\MINI PROJECT\\muscle-bmode-images\\B-Bilder"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# 1) LOAD FILE PATHS + LABELS
paths, labels = [], []

for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(".png"):
        continue

    name = fname.lower()
    if "longitudinal" in name:
        label = 0
    elif "transversal" in name:
        label = 1
    else:
        continue

    paths.append(os.path.join(DATA_DIR, fname))
    labels.append(label)

paths = np.array(paths)
labels = np.array(labels, dtype=np.int32)

assert len(paths) > 0, "❌ No images found. Check DATA_DIR path."

print("Total images:", len(paths))
print("Longitudinal:", int(np.sum(labels == 0)))
print("Transversal:", int(np.sum(labels == 1)))

# 2) TRAIN / VAL / TEST SPLIT
idx = np.random.permutation(len(paths))
paths, labels = paths[idx], labels[idx]

n_total = len(paths)
n_train = max(1, int(0.7 * n_total))
n_val = max(1, int(0.1 * n_total))

train_paths = paths[:n_train]
train_labels = labels[:n_train]

val_paths = paths[n_train:n_train + n_val]
val_labels = labels[n_train:n_train + n_val]

test_paths = paths[n_train + n_val:]
test_labels = labels[n_train + n_val:]

print("\nSplit sizes:")
print("Train:", len(train_paths))
print("Val  :", len(val_paths))
print("Test :", len(test_paths))

# 3) CLAHE FUNCTION 

def clahe_numpy(gray_img_2d: np.ndarray) -> np.ndarray:
    gray_img_2d = np.clip(gray_img_2d, 0.0, 1.0)
    img_uint8 = (gray_img_2d * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(img_uint8)

    return out.astype(np.float32) / 255.0


# 4) PREPROCESSING FUNCTION (WITH CLAHE)
def load_preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_png(img_bytes, channels=1)
    img = tf.cast(img, tf.float32)

    # keep aspect ratio 
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1], method="nearest")

    # scale to [0,1]
    img = img / 255.0
    img = tf.clip_by_value(img, 0.0, 1.0)

    # apply CLAHE 
    img_2d = img[..., 0]  # (H,W)
    img_2d = tf.numpy_function(func=clahe_numpy, inp=[img_2d], Tout=tf.float32)

    # restore shape info 
    img_2d.set_shape([IMG_SIZE[0], IMG_SIZE[1]])
    img = tf.expand_dims(img_2d, axis=-1)  # back to (H,W,1)

 

    return img, tf.cast(label, tf.float32)

# 5) DATASETS
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(load_preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 6) SIMPLE CNN MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),   
    tf.keras.layers.Dropout(0.3),                 
    tf.keras.layers.Dense(1)
])


# 7) LOSS + OPTIMIZER + METRICS

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(LR)

train_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
val_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
test_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)


# 8) TRAINING LOOP (WITH VALIDATION)

print("\n Training CNN (with validation)...\n")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_acc.reset_state()
    val_acc.reset_state()

    # TRAIN 
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        probs = tf.sigmoid(logits)
        train_acc.update_state(y_batch, probs)

    # VALIDATION
    for x_batch, y_batch in val_ds:
        logits_val = model(x_batch, training=False)
        probs_val = tf.sigmoid(logits_val)
        val_acc.update_state(y_batch, probs_val)

    best_val_acc = max(best_val_acc, float(val_acc.result().numpy()))

    print(
        f"Epoch {epoch:02d} | "
        f"loss={loss.numpy():.4f} | "
        f"train_acc={train_acc.result().numpy():.4f} | "
        f"val_acc={val_acc.result().numpy():.4f}"
    )

print("\nBest validation accuracy:", best_val_acc)

# 9) TEST EVALUATION + CONFUSION MATRIX

y_true, y_pred = [], []
test_acc.reset_state()

for x_batch, y_batch in test_ds:
    logits = model(x_batch, training=False)
    probs = tf.sigmoid(logits)
    preds = tf.cast(probs >= 0.5, tf.int32)

    test_acc.update_state(y_batch, probs)

    y_true.append(tf.cast(y_batch, tf.int32))
    y_pred.append(preds)

y_true = tf.concat(y_true, axis=0).numpy().reshape(-1)
y_pred = tf.concat(y_pred, axis=0).numpy().reshape(-1)

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()

print("\n TEST RESULTS")
print("Test accuracy:", float(test_acc.result().numpy()))
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("Labels: 0 = Longitudinal, 1 = Transversal")

#These are the results i got after running the code

#Best validation accuracy: 0.9069767594337463
# TEST RESULTS
#Test accuracy: 0.483146071434021
#Confusion matrix (rows=true, cols=pred):
#[[43  0]
 #[46  0]]

#Labels: 0 = Longitudinal, 1 = Transversal
