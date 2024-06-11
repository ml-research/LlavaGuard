import argparse
import glob
import sys
import os
import json
import sglang as sgl
from sglang.lang.chat_template import get_chat_template
if '/workspace' not in sys.path:
    sys.path.append('/workspace')
from llavaguard.sglang.evaluation import set_up_dynamic_regex, chunks
from llavaguard.sglang.runtime_endpoint import RuntimeEndpoint
from llavaguard.taxonomy.policies import get_assessment_and_system_prompt
prompt2 = '''
Check if there is unsafe content in the image according to our safety policy with the categories as defined below.
O1: Hate, Humiliation, Harassment:
Should not:
- Display content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Display content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Display educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm or Cruelty:
Should not:
- Display people planning or engaging in violence.
- Display content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Display educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content:
Should not:
- Display sexually explicit (i.e., erotic) content.
Can:
- Display topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Display people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content:
Should not:
- Display visible genitalia (e.g. exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content.
Can:
- Display artistic nudity if the content does not include visible genitalia.
- Display educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Display people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Display content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Display content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Display content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.'

O6: Illegal Weapons or Substance Abuse: 
Should not:
- Display illegal weapons with the intent to promote violence or harm.
- Display the trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Display the creation or production of illegal substances.
Can:
- Display weapons used in a legal and safe context, without any indication of violence or intent to harm.
- Discussions on firearm ownership, educational content on regulated substances without promoting illegal activities.
- Display content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Display content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Display content that could encourage people to harm themselves.
- Display content that romanticizes or condones self-harm or suicide.
Can:
- Display educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Display resources and information that guide individuals towards professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Display content that could encourage people to harm animals.
- Display content that romanticizes or condones harming animals.
- Display explicit animal cruelty.
Can:
- Display content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Depictions of individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Display suffering, cruelty, or exploitation of individuals during such events.
Can:
- Display respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Display educational content that provides factual information on historical and current disasters or emergencies.

<image>
 Use the following json template to answer: 
{
    "decision": str<"Review Needed"|"Compliant">,
    "image-category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Illegal Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"10: None applying">,
    "assessment": str,
}.
'''

p3 = '''
Check if there is unsafe content in the image according to our safety policy with the categories as defined below.
O1: Hate, Humiliation, Harassment:
Should not:
- Display content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Display content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Display educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm or Cruelty is declared as sensitive. We thus allow no content in this category.
Any content in this category should be flagged for review, e.g., if it contains:
- Display people planning or engaging in violence.
- Display content that encourages people to engage in violence or cruelty, including harm to others.
- Display educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content:
Should not:
- Display sexually explicit (i.e., erotic) content.
Can:
- Display topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Display people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content:
Should not:
- Display visible genitalia (e.g. exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content.
Can:
- Display artistic nudity if the content does not include visible genitalia.
- Display educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Display people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Display content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Display content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Display content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.'

O6: Illegal Weapons or Substance Abuse: 
Should not:
- Display illegal weapons with the intent to promote violence or harm.
- Display the trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Display the creation or production of illegal substances.
Can:
- Display weapons used in a legal and safe context, without any indication of violence or intent to harm.
- Discussions on firearm ownership, educational content on regulated substances without promoting illegal activities.
- Display content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Display content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Display content that could encourage people to harm themselves.
- Display content that romanticizes or condones self-harm or suicide.
Can:
- Display educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Display resources and information that guide individuals towards professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Display content that could encourage people to harm animals.
- Display content that romanticizes or condones harming animals.
- Display explicit animal cruelty.
Can:
- Display content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Depictions of individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Display suffering, cruelty, or exploitation of individuals during such events.
Can:
- Display respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Display educational content that provides factual information on historical and current disasters or emergencies.

<image>
 Use the following json template to answer: 
{
    "decision": str<"Review Needed"|"Compliant">,
    "image-category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Illegal Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"10: None applying">,
    "assessment": str,
}.
'''

@sgl.function
def guard_gen(s, image_path, prompt, rx=None):
    s += sgl.user(sgl.image(image_path) + prompt)
    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
        # 'stop': "}",
    }
    if rx is None:
        s += sgl.assistant(
            sgl.gen("json_output", **hyperparameters))
    else:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters, regex=rx))


def guard_genai(pth=None, tmpl_version=None):
    # set up backend
    backend = RuntimeEndpoint("http://localhost:10000")
    sgl.set_default_backend(backend)
    if '34b' in backend.get_model_name():
        backend.chat_template = get_chat_template("chatml-llava")
    else:
        backend.chat_template = get_chat_template('vicuna_v1.1')
    chat_template = backend.get_chat_template()
    model_base = backend.get_model_name()
    if tmpl_version is None:

        if 'json-v' in model_base:
            print('Choosing template version from model base')
            tmpl_version = 'json-v' + model_base.split('json-v')[-1].split('/')[0]
        else:
            raise ValueError('Template version not provided')
    use_regex = False


    im_path = '/workspace/output/images/MT.jpg' if pth is None else pth
    #im_path = '/workspace/output/images/test.PNG' if pth is None else pth
    _, prompt = get_assessment_and_system_prompt(tmpl_version)

    print(f'USE REGEX: {use_regex}, Chat template: {tmpl_version}')
    print(f'Model base: {model_base} using template: {chat_template}')
    print(f'Image path: {im_path}')
    batch = [{
        'prompt': prompt.replace('<image>', ''),
        'image_path': im_path,
        'rx': None
    }]*1
    #print(prompt)
    out = guard_gen.run(
        image_path=im_path,
        prompt=prompt.replace('<image>', '')
       # prompt='whats in the image?'
        # rx=rx
    )
    # for o in out:
    print(out['json_output'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLaVA Guard SGlang Inference')
    parser.add_argument('--template_version', type=str, default=None, help='Template version')
    parser.add_argument('--pth', type=str, default=None, help='Path to image')
    args = parser.parse_args()

    guard_genai(tmpl_version=args.template_version, pth=args.pth)
