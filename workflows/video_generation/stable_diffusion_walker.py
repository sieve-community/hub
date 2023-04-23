import sieve

@sieve.workflow(name="stable_diffusion_walker")
def stable_diffusion_walker(from_prompt: str, to_prompt: str) -> sieve.Video:
    return sieve.reference("sieve/stable_diffusion_walker")(from_prompt, to_prompt)
