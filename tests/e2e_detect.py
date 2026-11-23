"""Run GroundingDINO end to end on one frame and dump boxes. Usage: e2e_detect.py <frame.jpg> <out.json>"""
import sys, json, time, torch, warnings
warnings.filterwarnings("ignore")
import groundingdino, groundingdino.models.GroundingDINO.ms_deform_attn as m
from groundingdino.util.inference import load_model, load_image, predict
R = "/isis/home/hasana3/vlmtest/GroundingDINO"
print("groundingdino from", groundingdino.__file__, "| torch", torch.__version__, "| _C loaded:", hasattr(m, "_C"))
model = load_model(f"{R}/groundingdino/config/GroundingDINO_SwinT_OGC.py", f"{R}/weights/groundingdino_swint_ogc.pth")
_, img = load_image(sys.argv[1])
cap = "car . person . traffic light . tree ."
for _ in range(2): predict(model, img, cap, 0.35, 0.25, device="cuda")
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): boxes, logits, phrases = predict(model, img, cap, 0.35, 0.25, device="cuda")
torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 10 * 1e3
print(f"{len(boxes)} detections, {ms:.1f} ms/frame")
json.dump({"boxes": boxes.tolist(), "logits": logits.tolist(), "phrases": phrases, "ms": ms}, open(sys.argv[2], "w"))
