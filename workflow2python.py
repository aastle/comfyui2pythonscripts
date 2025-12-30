import os
import random
import torch
import sys
from nodes import NODE_CLASS_MAPPINGS
from typing import Sequence, Mapping, Any, Union

sys.path.append("../")
from nodes import (
    VAEDecode,
    KSamplerAdvanced,
    EmptyLatentImage,
    SaveImage,
    CheckpointLoaderSimple,
    CLIPTextEncode,
    LoraLoader,
    CLIPSetLastLayer,
)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

checkpointName = "juggernautXL_ragnarokBy.safetensors" #"juggernautXL_ragnarokBy.safetensors" 
loraName = "EasterEggHome_IXL.safetensors"
clipSkip = -2
positivePrompt = "masterpiece, best quality, cows, no humans, <lora:EasterEggHome_IXL:1.0>, outdoors, sky, day, cloud, water, tree, blue sky, window, grass, cherry blossoms, rock, stairs, mountain, road, bridge, river, path, moss, barrel, pond, stream, chimney, watchtower, a herd of cows"
negativePrompt = "low quality, bad anatomy, bad proportions, extra limbs, missing limbs, floating limbs, disconnected limbs, malformed limbs, extra digits, fewer digits, hands in inappropriate places, text in inappropriate places, bad composition, low resolution, jpeg artifacts, signature, watermark, username, blurry"
ksamplerSteps = 30
ksamplerCFG = 7
loraModelStrength = 0.5
loraClipStrength = 0.5

sampler = "dpmpp_2m"
schedulerName = "karras"

def main():
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name = checkpointName
        )
   
        loraLoader = LoraLoader()
        loraloader_12 = loraLoader.load_lora(
            lora_name = loraName,
            strength_model = loraModelStrength,
            strength_clip = loraClipStrength,
            model = checkpointloadersimple_4[0],
            clip = checkpointloadersimple_4[1],
        )
 
        clipSetLastLayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()  #CLIPSetLastLayer()
        clipSetLastLayer_13 = clipSetLastLayer.set_last_layer(
            #clip = loraloader_12[0],
            clip=get_value_at_index(loraloader_12, 1),
            stop_at_clip_layer = clipSkip,
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text = positivePrompt,
            clip = clipSetLastLayer_13[0],
        )

        cliptextencode_7 = cliptextencode.encode(
            text=negativePrompt,
            clip=clipSetLastLayer_13[0]
        )
     
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        for q in range(1):
            ksampleradvanced_10 = ksampleradvanced.sample(
                add_noise = "enable",
                noise_seed = random.randint(1, 2**64),
                steps = ksamplerSteps,
                cfg = ksamplerCFG,
                sampler_name = sampler,
                scheduler = schedulerName,
                start_at_step = 0,
                end_at_step = 1000,
                return_with_leftover_noise = "disable",
                model=loraloader_12[0],
                positive=cliptextencode_6[0],
                negative=cliptextencode_7[0],
                latent_image=emptylatentimage_5[0],
            )
 
            vaedecode_17 = vaedecode.decode(
                samples=ksampleradvanced_10[0], 
                vae=checkpointloadersimple_4[2]
            )
        
            saveimage_19 = saveimage.save_images(
                filename_prefix = "Workflow2Python_", 
                images = vaedecode_17[0]
            )


if __name__ == "__main__":
    main()