import sieve

@sieve.workflow(name="audio_noise_reduction_fullsubnet")
def audio_enhance(audio: sieve.Audio) -> sieve.Audio:
    return sieve.reference("sieve/fullsubnet_plus")(audio)
