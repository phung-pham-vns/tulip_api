import torch
from PIL import Image
import open_clip

"""Load TULIP model, transforms, and tokenizer."""
model, _, preprocess = open_clip.create_model_and_transforms(
    "TULIP-B-16-224",
    pretrained="/Users/mac/Documents/PHUNGPX/tulip_api/models/open_clip/tulip-B-16-224.ckpt",
)
model.eval()
tokenizer = open_clip.get_tokenizer("TULIP-B-16-224")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

image = preprocess(
    Image.open(
        "/Users/mac/Documents/PHUNGPX/tulip_api/models/open_clip/images/iStock-1052880600-1024x683.jpg"
    )
).unsqueeze(0)
text = tokenizer(["a cat", "a dog", "a bird"])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probabilities:", similarities)
