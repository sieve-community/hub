import sieve

@sieve.workflow(name="image_captioning_vitgpt2")
def image_captioning(image: sieve.Image) -> str:
    captioner = sieve.reference("sieve/vit_gpt2_image_captioner")
    return captioner(image)
