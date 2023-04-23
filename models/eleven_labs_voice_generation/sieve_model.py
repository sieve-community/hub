import sieve

@sieve.function(
    name="eleven_labs_voice_generation",
    python_packages=[
        "requests==2.25.1"
    ],
    environment_variables=[
        sieve.Env(
            name="ELEVEN_LABS_KEY",
            description="Eleven Labs API Key"
        )
    ]
)
def eleven_labs_voice_generation(args: dict) -> sieve.Audio:
    import os
    if 'script' not in args:
        raise ValueError('Missing script argument')
    else:
        script = args['script']
    if 'voice_settings' not in args:
        args['voice_settings'] = {}
    if 'stability' not in args['voice_settings']:
        stability = 0
    else:
        stability = args['voice_settings']['stability']
    if 'similarity_boost' not in args['voice_settings']:
        similarity_boost = 0
    else:
        similarity_boost = args['voice_settings']['similarity_boost']

    if 'voice_id' not in args:
        raise ValueError('Missing voice_id argument')
    else:
        voice_id = args['voice_id']

    import requests

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    payload = {
        "text": script,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'accept': 'audio/mpeg',
        'xi-api-key': os.environ['ELEVEN_LABS_KEY']
    }

    print("Sending request to Eleven Labs API")
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise ValueError(response.text)
    
    # save the audio file from the response to a audio.mpeg file
    with open('audio.mpeg', 'wb') as f:
        f.write(response.content)

    print("Returning audio file")
    return sieve.Audio(path='audio.mpeg')
