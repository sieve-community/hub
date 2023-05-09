import numpy as np
from PIL import Image
import torch
import sieve
from typing import List

def image_transform_grounding(init_image):
    import groundingdinolocal.datasets.transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

@sieve.Model(
    name="grounding-dino",
    gpu=True,
    python_packages=[
        "torch==1.13.1",
        "torchvision==0.14.1",
        "transformers==4.25.1",
        "addict",
        "yapf",
        "timm",
        "numpy",
        "opencv-python",
        "supervision==0.3.2",
        "pycocotools",
        "packaging==21.3"
    ],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "python3-setuptools"
    ],
    python_version="3.8",
    run_commands=[
        'mkdir -p /root/.cache/dino/models/',
        'wget -c https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -P /root/.cache/dino/models/',
        'wget -c https://storage.googleapis.com/sieve-public-data/groundingdino-0.1.0-cp38-cp38-linux_x86_64.whl -P /root/.cache/',
        'pip3 install /root/.cache/groundingdino-0.1.0-cp38-cp38-linux_x86_64.whl',
    ],
    environment_variables=[
        sieve.Env(name="prompt", default="person"),
        sieve.Env(name="retain_prompt", default="person"),
        sieve.Env(name="box_confidence_threshold", default="0.25"),
        sieve.Env(name="phrase_confidence_threshold", default="0.25"),
    ]
)
class GroundingDINO:
    def __setup__(self):
        import warnings
        import os
        from groundingdinolocal.models import build_model
        from groundingdinolocal.util.slconfig import SLConfig
        from groundingdinolocal.util.utils import clean_state_dict

        warnings.filterwarnings("ignore")
        
        checkpoint_file = '/root/.cache/dino/models/groundingdino_swint_ogc.pth'
        config_file = 'groundingdinolocal/config/GroundingDINO_SwinT_OGC.py'
        
        args = SLConfig.fromfile(config_file)
        self.model = build_model(args)

        checkpoint = torch.load(checkpoint_file, map_location='cuda')
        log = self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(checkpoint_file, log))

    def __predict__(self, image: sieve.Image) -> List:
        from torchvision.ops import box_convert
        from groundingdinolocal.util.inference import annotate, load_image, predict
        import cv2
        import os
        prompt, box_confidence_threshold, phrase_confidence_threshold = os.environ["prompt"], float(os.environ["box_confidence_threshold"]), float(os.environ["phrase_confidence_threshold"])
        input_image = Image.open(image.path)

        init_image = input_image.convert("RGB")
        _, image_tensor = image_transform_grounding(init_image)

        print("Running inference...")
        boxes, logits, phrases = predict(self.model, image_tensor, prompt, box_confidence_threshold, phrase_confidence_threshold)
        
        h, w, _ = image.array.shape
        boxes = boxes * torch.Tensor([w, h, w, h])

        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        xyxy = xyxy.tolist()
        logits = logits.tolist()
        print("Boxes found: ", len(xyxy))

        outputs = []
        for box, cls_name, score in zip(xyxy, phrases, logits):
            if "retain_prompt" in os.environ and cls_name in os.environ["retain_prompt"]:
                outputs.append({
                    "box": box,
                    "class_name": cls_name,
                    "score": score,
                    "frame_number": None if not hasattr(image, "frame_number") else image.frame_number
                })

        return outputs

@sieve.workflow(name="test_grounding-dino")
def test_grounding_dino(image: sieve.Image) -> List:
    return GroundingDINO()(image)
