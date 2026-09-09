import io
import base64

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
try:
    _JET_CMAP = matplotlib.colormaps["jet"]  # matplotlib >= 3.9
except AttributeError:
    import matplotlib.cm as cm
    _JET_CMAP = cm.get_cmap("jet")  # older matplotlib fallback

from config import ZONE_ROWS, ZONE_COLS


def compute_gradcam(model, img_tensor, target_class_idx):
    """Real Grad-CAM on DenseNet121's final feature block (model.features).

    Returns a 224x224 float32 array in [0, 1]. No mocked or interpolated values —
    if the forward/backward hooks fail to capture anything, this raises instead of
    silently returning a placeholder.
    """
    feature_maps = []

    def forward_hook(module, inputs, output):
        feature_maps.append(output)

    target_layer = model.features
    handle = target_layer.register_forward_hook(forward_hook)

    try:
        output = model(img_tensor)
        score = output[0, target_class_idx]

        if not feature_maps:
            raise RuntimeError("Grad-CAM forward hook captured no feature maps.")

        f_map_tensor = feature_maps[0]
        grads_tensor = torch.autograd.grad(score, f_map_tensor, retain_graph=True)[0]
    finally:
        handle.remove()

    f_map = f_map_tensor.detach().cpu().numpy()[0]   # [C, H, W]
    grads = grads_tensor.detach().cpu().numpy()[0]    # [C, H, W]

    weights = np.mean(grads, axis=(1, 2))              # [C]
    cam = np.zeros(f_map.shape[1:], dtype=np.float32)  # [H, W]
    for i, w in enumerate(weights):
        cam += w * f_map[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_224 = F.interpolate(cam_tensor, size=(224, 224), mode="bilinear", align_corners=False)
    return cam_224.squeeze().numpy()


def zone_breakdown(cam_224):
    """Splits the 224x224 CAM into a ZONE_ROWS x ZONE_COLS grid and returns the
    mean activation for each zone as a percentage. This is what powers the
    zone-wise analysis panel — every number here comes directly from the same
    Grad-CAM array used for the heatmap, nothing is invented separately.
    """
    h, w = cam_224.shape
    n_rows, n_cols = len(ZONE_ROWS), len(ZONE_COLS)
    row_edges = np.linspace(0, h, n_rows + 1).astype(int)
    col_edges = np.linspace(0, w, n_cols + 1).astype(int)

    zones = []
    for ri, row_name in enumerate(ZONE_ROWS):
        for ci, col_name in enumerate(ZONE_COLS):
            patch = cam_224[row_edges[ri]:row_edges[ri + 1], col_edges[ci]:col_edges[ci + 1]]
            score = float(np.mean(patch) * 100.0)
            zones.append({
                "zone": f"{col_name} {row_name}",
                "row": row_name,
                "col": col_name,
                "activation": round(score, 1),
            })
    return zones


def make_overlay_base64(pil_img_224, cam_224, alpha=0.45):
    """Blends the Grad-CAM heatmap onto the (already 224x224) input image and
    returns a base64 PNG data URI, ready for the frontend to drop straight into
    an <img> tag — no client-side canvas compositing required, which removes an
    entire category of rendering bugs from the previous build.
    """
    heatmap = _JET_CMAP(cam_224)[:, :, :3]  # [H, W, 3] in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert("RGBA")

    base = pil_img_224.convert("RGBA")
    blended = Image.blend(base, heatmap_img, alpha=alpha)

    buf = io.BytesIO()
    blended.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
