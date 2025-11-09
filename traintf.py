#!/usr/bin/env python3
# train_tf.py — TensorFlow detector training on YOLO-format labels (VRAM-friendly, long-run)

import os, math, yaml, glob
import tensorflow as tf
import numpy as np
from keras.saving import register_keras_serializable

# ---------------- Runtime safety on 4 GB VRAM ----------------
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
tf.keras.mixed_precision.set_global_policy("float32")  # keep stable

# ---------------- Config ----------------
IMG_SIZE = 512
GRID_SIZE = 13
BATCH_SIZE = 2
ACCUM_STEPS = 8
EPOCHS = 200
WARMUP_EPOCHS = 5
AUGMENT = True
MIXED_PRECISION = False

# ---------------- Utilities ----------------
def load_names(classes_path):
    with open(classes_path, "r") as f:
        names = [l.strip() for l in f.readlines() if l.strip()]
    return names

def load_paths(root):
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    imgs = sorted([p for p in glob.glob(os.path.join(img_dir, "*"))
                   if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]])
    def lbl_for(img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        p = os.path.join(lbl_dir, stem + ".txt")
        return p if os.path.exists(p) else None
    return [(i, lbl_for(i)) for i in imgs]

def parse_yolo_label(txt, num_classes):
    objs = []
    if txt is None or not tf.io.gfile.exists(txt):
        return objs
    with tf.io.gfile.GFile(txt, "r") as f:
        for line in f.read().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:  continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            if 0 <= cls < num_classes:
                objs.append((cls, x, y, w, h))
    return objs

def assign_to_grid(objects, S, num_classes):
    t = np.zeros((S, S, 5 + num_classes), dtype=np.float32)
    for cls, x, y, w, h in objects:
        cx = min(S - 1, max(0, int(x * S)))
        cy = min(S - 1, max(0, int(y * S)))
        tx = x * S - cx
        ty = y * S - cy
        if t[cy, cx, 4] == 0:
            t[cy, cx, 0:4] = [tx, ty, w, h]
            t[cy, cx, 4] = 1.0
            t[cy, cx, 5 + cls] = 1.0
    return t

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)
    return img

def build_dataset(pairs, class_count, shuffle=True):
    img_paths = [p[0] for p in pairs]
    lbl_paths = [p[1] if p[1] is not None else "" for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))

    def _load(img_path, lbl_path):
        img = decode_image(img_path)
        if AUGMENT:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.08)
            img = tf.image.random_contrast(img, 0.9, 1.1)

        def _py_parse(lbl_tensor):
            b = lbl_tensor.numpy() if hasattr(lbl_tensor, "numpy") else lbl_tensor
            path = b.decode("utf-8") if isinstance(b, bytes) else str(b)
            path = path.strip()
            objs = parse_yolo_label(path if path else None, class_count)
            tgt = assign_to_grid(objs, GRID_SIZE, class_count).astype(np.float32)
            return tgt

        target = tf.py_function(_py_parse, [lbl_path], Tout=tf.float32)
        target.set_shape((GRID_SIZE, GRID_SIZE, 5 + class_count))
        return img, target

    if shuffle:
        ds = ds.shuffle(min(2048, len(pairs)), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=1).batch(BATCH_SIZE).prefetch(1)
    return ds

# --------------- Model ----------------
@register_keras_serializable()
def split_activation(t):
    box  = tf.keras.activations.sigmoid(t[..., 0:2])  # tx, ty
    size = tf.keras.activations.sigmoid(t[..., 2:4])  # tw, th
    obj  = tf.keras.activations.sigmoid(t[..., 4:5])  # objectness
    cls  = t[..., 5:]                                 # logits
    return tf.concat([box, size, obj, cls], axis=-1)

def build_model(num_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")
    x = backbone(inputs)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Resizing(GRID_SIZE, GRID_SIZE, interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    out = tf.keras.layers.Conv2D(5 + num_classes, 1, padding="same", activation=None)(x)
    out = tf.keras.layers.Activation(split_activation)(out)
    model = tf.keras.Model(inputs, out, name="mini_yolo_tf")
    model.backbone = backbone
    return model

# --------------- Loss ----------------
def yolo_v1_loss(y_true, y_pred, num_classes,
                 lambda_box=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    obj_mask = y_true[..., 4:5]
    noobj_mask = 1.0 - obj_mask

    box_true = y_true[..., 0:4]
    box_pred = y_pred[..., 0:4]
    box_loss = tf.reduce_sum(tf.abs(box_true - box_pred) * obj_mask)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
    obj_true = y_true[..., 4:5]
    obj_pred = y_pred[..., 4:5]
    obj_bce = bce(obj_true, obj_pred)
    obj_bce = tf.expand_dims(obj_bce, -1) if obj_bce.shape.rank == 3 else obj_bce
    obj_loss   = tf.reduce_sum(obj_bce * obj_mask)
    noobj_loss = tf.reduce_sum(obj_bce * noobj_mask)

    cls_true = y_true[..., 5:]
    cls_logits = y_pred[..., 5:]
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")
    cls_map = ce(cls_true, cls_logits)
    cls_map = tf.expand_dims(cls_map, -1)
    cls_loss = tf.reduce_sum(cls_map * obj_mask)

    total = (lambda_box * box_loss
             + lambda_obj * obj_loss
             + lambda_noobj * noobj_loss
             + lambda_cls * cls_loss)

    B = tf.cast(tf.shape(y_true)[0], tf.float32)
    S = tf.cast(tf.shape(y_true)[1], tf.float32)
    return total / (B * S * S)

class DetectorLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes): super().__init__(reduction="sum"); self.num_classes = num_classes
    def call(self, y_true, y_pred):  return yolo_v1_loss(y_true, y_pred, self.num_classes)

# -------- LR schedule (now serializable ✅) --------
@register_keras_serializable(package="lr_schedules")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_epochs, total_epochs, steps_per_epoch, min_lr=1e-6):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(1, warmup_epochs * steps_per_epoch))
        self.total_steps  = int(max(self.warmup_steps + 1, total_epochs * steps_per_epoch))
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / float(self.warmup_steps))
        prog = tf.clip_by_value(
            (step - float(self.warmup_steps)) / max(1.0, float(self.total_steps - self.warmup_steps)),
            0.0, 1.0
        )
        cos = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(np.pi * prog))
        return tf.where(step < float(self.warmup_steps), tf.minimum(warm, self.base_lr), cos)

    def get_config(self):
        # must return JSON-serializable values
        return {
            "base_lr": self.base_lr,
            "warmup_epochs": int(round(self.warmup_steps)),  # kept for readability; not used on reload
            "total_epochs": int(round(self.total_steps)),
            "steps_per_epoch": 1,  # placeholder; not used on reload
            "min_lr": self.min_lr,
        }

# --------------- Gradient Accumulation model ---------------
class AccumModel(tf.keras.Model):
    def __init__(self, accum_steps, **kw):
        super().__init__(**kw)
        self.accum_steps = int(accum_steps)
        self._accum = None
        self._accum_count = tf.Variable(0, trainable=False, dtype=tf.int64)

    def _ensure_accum(self, grads):
        if self._accum is None:
            self._accum = [tf.Variable(tf.zeros_like(g), trainable=False) for g in grads]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self._ensure_accum(grads)
        for v, g in zip(self._accum, grads):
            v.assign_add(g)
        self._accum_count.assign_add(1)
        if int(self._accum_count.numpy()) % self.accum_steps == 0:
            mean_grads = [g / float(self.accum_steps) for g in self._accum]
            self.optimizer.apply_gradients(zip(mean_grads, self.trainable_variables))
            for v in self._accum:
                v.assign(tf.zeros_like(v))
        self.compiled_metrics.update_state(y, y_pred)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

# --------------- Callbacks ---------------
class UnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch): super().__init__(); self.unfreeze_epoch = unfreeze_epoch
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch and hasattr(self.model, "backbone"):
            for l in self.model.backbone.layers:
                l.trainable = True
            print(f"\n✅ Unfroze backbone at epoch {epoch}\n")

def main():
    data_root = "./dataset"
    classes_path = "./classes.txt"
    if os.path.exists("yolo_params.yaml"):
        try:
            with open("yolo_params.yaml","r") as f:
                y = yaml.safe_load(f)
                if isinstance(y.get("train"), str) and isinstance(y.get("val"), str):
                    data_root = os.path.commonpath([y["train"], y["val"]])
                if "names" in y and isinstance(y["names"], dict) and len(y["names"])>0:
                    with open("classes.txt","w") as f2:
                        for i in range(len(y["names"])):
                            f2.write(str(y["names"][i])+"\n")
        except Exception:
            pass

    names = load_names(classes_path); NUM_CLASSES = len(names)
    if MIXED_PRECISION: tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_pairs = load_paths(os.path.join(data_root, "train"))
    val_pairs   = load_paths(os.path.join(data_root, "val"))
    assert len(train_pairs), "No training images found"
    assert len(val_pairs),   "No validation images found"

    train_ds = build_dataset(train_pairs, NUM_CLASSES, shuffle=True)
    val_ds   = build_dataset(val_pairs,   NUM_CLASSES, shuffle=False)

    steps_per_epoch = max(1, math.ceil(len(train_pairs) / BATCH_SIZE))
    lr = WarmupCosine(base_lr=1e-4, warmup_epochs=WARMUP_EPOCHS,
                      total_epochs=EPOCHS, steps_per_epoch=steps_per_epoch, min_lr=1e-6)

    base_model = build_model(NUM_CLASSES)
    for l in base_model.backbone.layers: l.trainable = False

    model = AccumModel(accum_steps=ACCUM_STEPS, inputs=base_model.input, outputs=base_model.output)
    model.backbone = base_model.backbone
    model.summary()

    loss_fn = DetectorLoss(NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=10.0)

    os.makedirs("tf_runs/weights", exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath="tf_runs/weights/best.keras",
        monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1
    )
    tb = tf.keras.callbacks.TensorBoard(log_dir="tf_runs", write_graph=False)
    unfreeze = UnfreezeCallback(unfreeze_epoch=WARMUP_EPOCHS)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1)
    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[unfreeze, ckpt, tb, early, reduce],
        verbose=1
    )

    model.save("tf_runs/weights/final.keras")
    print("\nTraining complete.\n- Best:  tf_runs/weights/best.keras"
          "\n- Final: tf_runs/weights/final.keras"
          "\nTensorBoard: tensorboard --logdir tf_runs\n")

if __name__ == "__main__":
    main()
