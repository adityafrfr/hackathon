import os, cv2, numpy as np, tensorflow as tf
from glob import glob
from pathlib import Path
from keras.saving import register_keras_serializable

# ==== GPU-friendly settings (avoid XLA/VRAM spikes) ====
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

# --- Register the custom activation used in the saved model ---
@register_keras_serializable()
def split_activation(t):
    box  = tf.keras.activations.sigmoid(t[..., 0:2])   # tx, ty
    size = tf.keras.activations.sigmoid(t[..., 2:4])   # tw, th
    obj  = tf.keras.activations.sigmoid(t[..., 4:5])   # obj
    cls  = t[..., 5:]                                  # logits (leave as-is)
    return tf.concat([box, size, obj, cls], axis=-1)

# ===== CONFIG =====
MODEL_PATH = "tf_runs/weights/best.keras"   # or "tf_runs/weights/final.keras"
TEST_DIR   = "test/images"          # <- adjust if your test path differs
CLASSES_FILE = "classes.txt"
IMG_SIZE   = 512
GRID_SIZE  = 13
CONF_THRES = 0.5

# ===== LOAD MODEL =====
model = tf.keras.models.load_model(
    MODEL_PATH, compile=False, custom_objects={"split_activation": split_activation}
)
print(f"✅ Loaded {MODEL_PATH}")

# ===== CLASSES =====
with open(CLASSES_FILE, "r") as f:
    NAMES = [l.strip() for l in f if l.strip()]

# ===== HELPERS =====
def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

def decode(pred, conf=CONF_THRES):
    """pred: [S,S,5+C] -> list[x1,y1,x2,y2,cls,score] in 0..1 coords"""
    S = pred.shape[0]
    out = []
    for r in range(S):
        for c in range(S):
            cell = pred[r, c]
            obj = float(cell[4])
            if obj < conf:
                continue
            cls_probs = softmax(cell[5:])
            cls_id = int(np.argmax(cls_probs))
            cls_score = float(cls_probs[cls_id])
            score = obj * cls_score

            cx, cy, w, h = map(float, cell[0:4])
            cx = (c + cx) / S
            cy = (r + cy) / S
            x1, y1 = max(0.0, cx - w/2), max(0.0, cy - h/2)
            x2, y2 = min(1.0, cx + w/2), min(1.0, cy + h/2)
            out.append([x1, y1, x2, y2, cls_id, score])
    return out

def draw(img_rgb, boxes):
    h, w = img_rgb.shape[:2]
    for x1, y1, x2, y2, cls, sc in boxes:
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{NAMES[cls]} {sc:.2f}"
        cv2.putText(img_rgb, label, (x1, max(12, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img_rgb

# ===== RUN =====
paths = sorted(glob(os.path.join(TEST_DIR, "*.png")) +
               glob(os.path.join(TEST_DIR, "*.jpg")) +
               glob(os.path.join(TEST_DIR, "*.jpeg")))
print(f"Found {len(paths)} test images")

out_dir = Path("predictions"); out_dir.mkdir(exist_ok=True)

for p in paths[:50]:  # change slice to run more/all
    bgr = cv2.imread(p)
    if bgr is None:
        print(f"skip (unreadable): {p}")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    pred = model.predict(np.expand_dims(inp, 0), verbose=0)[0]  # [S,S,5+C]
    boxes = decode(pred)
    vis = draw(rgb.copy(), boxes)

    stem = os.path.splitext(os.path.basename(p))[0]
    out_path = out_dir / f"{stem}_pred.png"
    # save in RGB (cv2 expects BGR, so use matplotlib writer or convert back)
    import matplotlib.pyplot as plt
    plt.imsave(out_path, vis)  # vis is RGB
    print(f"wrote {out_path}")
