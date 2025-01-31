# LlavaGuard
*LLAVAGUARD: VLM-based Safeguard for Vision Dataset Curation and Safety Assessment*

This is the anonymous repository for LlavaGuard, a versatile framework for evaluating the safety compliance of visual content. It is designed for both dataset annotation and safeguarding of generative models. In the following we provide access to the private & anonymized repositories of the dataset and model weights on Hugging Face. To access them, please use the following token: "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".

HF Repositories:
- LG-Anonym/LlavaGuard-Dataset
- LG-Anonym/LlavaGuard-v1.2-0.5B-OV
- LG-Anonym/LlavaGuard-v1.2-7B-OV

<div align="center">
  <img src="figs/llavaguard_pipe.png" width="400"  alt="">
</div>

## Table of Contents
- [Usage](#usage)
- [Infernce via SGLang](#infernce-via-sglang)
  - 1 [Install the requirements](#1-install-the-requirements)
  - 2 [Launch LlavaGuard Server](#1-launch-llavaguard-server)
  - 3 [Perform model inference](#3-perform-model-inference)
  - 4 [Response](#response)
- [Methodology](#methodology)
- [Safety Taxonomy](#safety-taxonomy)
- [Evaluation](#evaluation)



## Usage
The provided LlavaGuard weights are compatible for inference via [SGLang](https://github.com/sgl-project/sglang) as described below. 
The models are pre-trained on our dataset and can be used for further tuning either via LoRAs or full training. We also provide scripts for data preparation/augmentation and training in our repository (see scripts).
You can use the following [docker file for infernce](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile). For training, a working LLaVA installation is required, e.g., see our docker file for tuning.

### Infernce via SGLang

For inference, you use the above-mentioned [sglang docker](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile) and proceed with step 1.
Otherwise, you can also install sglang via pip or from source [see here](https://github.com/sgl-project/sglang).

#### 1. Launch LlavaGuard Server
Select one of the three checkpoints provided and launch the server. The following code snippet demonstrates how to load the 7B, 13B, and 34B checkpoints, respectively.
   ```sh
   CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path LG-Anonym/LlavaGuard-v1.2-0.5B-OV --tokenizer-path LG-Anonym/LlavaGuard-v1.2-0.5B-OV --port 10000
   CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path LG-Anonym/LlavaGuard-v1.2-7B-OV --tokenizer-path LG-Anonym/LlavaGuard-v1.2-7B-OV --port 10000
   ```
#### 2. Perform model inference
After setting up the server, you can run a Python script to perform inference with LlavaGuard. E.g. 
   ```sh
   python my_script.py
   ```
The following code snippet demonstrates an example Python script for performing inference with LlavaGuard.
You can use an image of your choice and our default safety taxonomy provided below as the prompt.
```python
import sglang as sgl
from sglang import RuntimeEndpoint

@sgl.function
def guard_gen(s, image_path, prompt):
    s += sgl.user(sgl.image(image_path) + prompt)
    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
    }
    s += sgl.assistant(sgl.gen("json_output", **hyperparameters))

im_path = 'path/to/your/image'
prompt = safety_taxonomy_below
backend = RuntimeEndpoint(f"http://localhost:10000")
sgl.set_default_backend(backend)
out = guard_gen.run(image_path=im_path, prompt=prompt)
print(out['json_output'])
```
#### 4. Response
The server will respond with the safety assessment for the content provided by the user. The response will include a safety **rating**, a safety **categories**, and an in-depth **rationale**.


**Important Notes:**
* This version is not compatible with the HF transformer implementation and must be used with SGLang or LLaVA implementation.
* An HF implementation is available aswell.
## Methodology

This paper introduces Llavaguard, a suite of VLM-based vision safeguards that address the critical need for reliable tools in the era of large-scale data and models. To this end, we establish a novel open framework, describing a customizable safety taxonomy, data preprocessing, augmentation, and training setup. For teaching a VLM safeguard on safety, we further create a multimodal safety dataset with high-quality human expert annotations, where each image is labeled with a safety rating, category and rationale. We also employ advanced augmentations to support context-specific assessments. The resulting Llavaguard models, ranging from 0.5B to 7B, serve as a versatile tool for evaluating the safety compliance of visual content against flexible policies. In comprehensive experiments, Llavaguard outperforms both state-of-the-art safeguards and VLMs in accuracy and in flexibly handling different policies. Additionally, we demonstrate Llavaguard's performance in two real-world applications: large-scale dataset annotation and moderation of text-to-image models. We make our entire framework publicly available, including the dataset and model weights.


<div align="center">
  <img src="figs/fig1.png" width="400"  alt="">
</div>


## Safety Taxonomy
Our default policy prompt looks like this:

```
prompt = """
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content: 
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}
"""
```
