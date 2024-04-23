import json

from llavaguard.taxonomy.default_policy import policy_system_prompt, policy_system_prompt_v6, \
    policy_system_prompt_v8

# categories from human feedback labels
inappropriate_categories = ['Humiliation, Harassment or Hate',
                            'Violence, Harm or Cruelty',
                            'Nudity',
                            'Animal Cruelty',
                            'Substance Abuse or Weapons',
                            'Illegal Activities',
                            'Explicit Sexual Content',
                            'Disasters or Emergencies',
                            'Suicide or Self Harm']
appropriate_categories = ['Polite Communication and Gestures', 'Nature', 'Respectful or Appropriate Content',
                          'Peace or Care', 'NaN']

# policy categories 1
categories1 = [
    'O1: Violence, Hate, Humiliation, Harassment',
    'O2: Sexual Content',
    'O3: Nudity Content',
    'O4: Criminal Planning',
    'O5: Guns and Illegal Weapons',
    'O6: Substance Abuse',
    'O7: Self-Harm',
    'O8: Animal Cruelty'
]
# policy categories 2
categories2 = [
    'O1: Hate, Humiliation, Harassment',
    'O2: Violence, Harm or Cruelty',
    'O3: Sexual Content',
    'O4: Nudity Content',
    'O5: Criminal Planning',
    'O6: Illegal Weapons or Substance Abuse',
    'O7: Self-Harm',
    'O8: Animal Cruelty',
    'O9: Disasters or Emergencies',
    '10: None applying'
]

cat_prompt1 = '|'.join([f'"{x}"' for x in categories1])
cat_prompt2 = '|'.join([f'"{x}"' for x in categories2])

mapping = {
    'Humiliation, Harassment or Hate': 'O1: Hate, Humiliation, Harassment',
    'Violence, Harm or Cruelty': 'O2: Violence, Harm or Cruelty',
    'Explicit Sexual Content': 'O3: Sexual Content',
    'Nudity': 'O4: Nudity Content',
    'Illegal Activities': 'O5: Criminal Planning',
    'Substance Abuse or Weapons': 'O6: Illegal Weapons or Substance Abuse',
    'Suicide or Self Harm': 'O7: Self-Harm',
    'Animal Cruelty': 'O8: Animal Cruelty',
    'Disasters or Emergencies': 'O9: Disasters or Emergencies',
    'Polite Communication and Gestures': '10: None applying',
    'Nature': '10: None applying',
    'Respectful or Appropriate Content': '10: None applying',
    'Peace or Care': '10: None applying',
    'nan': '10: None applying',
    'NaN': '10: None applying',
}


def convert_category(category):
    if category in mapping.keys():
        category = mapping[category]
        category = category[4:]
    if ': ' in category:
        category = category.split(': ')[1]
    return category


json_template_v0 = f''' Use the following json template to answer:
{{
    "final-assessment": str<"Review Needed"|"Compliant">, '
    "policy-category": str<{cat_prompt1}>,
}}.
If the image content violates the safety policy, the final-assessment should be "Review Needed", otherwise it should be "Compliant".
Select a policy category that is or could be violated by the image.
'''

json_template_v1 = (
        ('Use the following json template to answer: {"final-assessment": str<"Review Needed", "Compliant">, '
         '"policy-category": ') + f'str<{cat_prompt1}>,' +
        ' "explanation": str}. A review is needed as soon the safety policies are violated. '
        'Provide the policy category and explain the reason for your final assessment.')

json_template_v2 = \
    f''' Use the following json template to answer:
{{
    "final-assessment": str<"Review Needed"|"Compliant">, '
    "policy-category": str<{cat_prompt1}>,
    "explanation": str
}}.
The final assessment should determine whether the content is "compliant" to the safety policy.
If the content violates the safety policy, the final assessment should be "Review Needed".
Provide the policy category that is or could be violated and explain the reason for your final assessment.
'''

json_template_v3 = \
    f''' Use the following json template to answer: 
{{
    "policy-category": str<{cat_prompt1}>,
    "explanation": str,
    "final-assessment": str<"Review Needed"|"Compliant">,
}}.
Select the policy-category that is or could be violated and explain the image content and reason for your final assessment.
If the content violates the safety policy, the final-assessment should be "Review Needed", otherwise it should be "Compliant".
'''

json_template_v4 = \
    f''' Use the following json template to answer: 
{{
    "image-category": str<{cat_prompt1}>,
    "assessment": str,
    "decision": str<"Review Needed"|"Compliant">,
}}.
Select and assign an image-category from the list. Explain the image content and assess whether it violates the safety policy.
If the content violates the safety policy, the decision should be "Review Needed", otherwise it should be "Compliant".
'''

json_template_v5 = f''' Use the following json template to answer:
{{
    "final-assessment": str<"Review Needed"|"Compliant">, '
}}.
If the image content violates the safety policy, the final-assessment should be "Review Needed", otherwise it should be "Compliant".
'''

json_template_v6 = \
    f''' Use the following json template to answer: 
{{
    "assessment": str<"Review Needed"|"Compliant">,
    "category": str<{cat_prompt2}>,
    "explanation": str,
}}.
If the image content violates the safety policy, the assessment should be "Review Needed", otherwise it should be "Compliant".
Select a category from the policy that is or could be violated, if none of the categories apply, select "10: None applying".
Explain the image content and the reason for your assessment.
'''

json_template_v7 = \
    f''' Use the following json template to answer: 
{{
    "image-category": str<{cat_prompt2}>,
    "assessment": str,
    "decision": str<"Review Needed"|"Compliant">,
}}.
Select and assign an image-category from the list. If none of the policy-categories are matching, select "10: None applying".
Explain the image content and assess whether it violates the safety policy.
If the content violates the safety policy, the decision should be "Review Needed", otherwise it should be "Compliant".
'''

json_template_v8 = \
    f''' Use the following json template to answer: 
{{
    "image-category": str<{cat_prompt2}>,
    "assessment": str,
    "decision": str<"Review Needed"|"Compliant">,
}}.
Select and assign an image-category from the list. Explain the image content and assess whether it violates the safety policy.
If the content violates the safety policy, the decision should be "Review Needed", otherwise it should be "Compliant".
'''

json_templates = {
    'nl': " If unsafe, the answer must include a comma-separated explanation with a list of violated categories.",
    'json-v0': json_template_v0,
    'json-v1': json_template_v1,
    'json-v2': json_template_v2,
    'json-v3': json_template_v3,
    'json-v4': json_template_v4,
    'json-v5': json_template_v5,
    'json-v6': json_template_v6,
    'json-v7': json_template_v7,
    'json-v8': json_template_v8,
    'json-v9': json_template_v7,
    'json-v10': json_template_v8,
}

policy_system_prompts = {
    'nl': policy_system_prompt,
    'json-v0': policy_system_prompt,
    'json-v1': policy_system_prompt,
    'json-v2': policy_system_prompt,
    'json-v3': policy_system_prompt,
    'json-v4': policy_system_prompt,
    'json-v5': policy_system_prompt,
    'json-v6': policy_system_prompt_v6,
    'json-v7': policy_system_prompt_v6,
    'json-v8': policy_system_prompt_v8,
    'json-v9': policy_system_prompt_v6,
    'json-v10': policy_system_prompt_v6,
}


def get_assessment(score, category, explanation=None):
    final_assessment1 = ['Compliant', 'Review Needed']
    final_assessment1 = ['Safe', 'Unsafe']
    scores_label = ["Highly Inappropriate", "Moderately Inappropriate", "Barely Acceptable", "Generally Acceptable"]
    c = ['policy violation', 'approved', 'compliant']
    assessment = 'Compliant' if 'Acceptable' in score else 'Review Needed'

    policy_assessment = f"""final-assessment: {assessment}
    policy-category: {category}"""
    # policy_assesment += f', Explanation: The content is deemed {score} and is compliant to the provided policy under the category: {category}'

    if assessment == 'Compliant':
        policy_assessment += f'\nExplanation: The image is compliant to the provided safety policy.'
        if explanation is not None and explanation != '':
            policy_assessment += f' {explanation}'
    elif explanation is not None and explanation != '':
        policy_assessment += f'\nExplanation: {explanation}'
    return policy_assessment


def get_assessment_json_v0(score, category, explanation=None):
    assessment = 'Compliant' if 'Acceptable' in score else 'Review Needed'
    policy_assessment = {
        "final-assessment": assessment,
        "policy-category": category,
    }
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json(score, category, explanation=None):
    assessment = 'Compliant' if 'Acceptable' in score else 'Review Needed'
    policy_assessment = {
        "final-assessment": assessment,
        "policy-category": category,
    }
    if assessment == 'Compliant':
        policy_assessment['explanation'] = f'The image is compliant to the provided safety policy.'
    else:
        policy_assessment['explanation'] = 'The image violates the provided safety policy.'
    if explanation is not None and explanation != '':
        policy_assessment['explanation'] += f' {explanation}'
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v3(score, category, explanation=None):
    assessment, statement = (
        'Compliant', 'The image is compliant to the provided safety policy.') if 'Acceptable' in score else (
        'Review Needed', 'The image violates the provided safety policy.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.':
        ex = explanation + '. ' + statement
    else:
        ex = explanation + ' ' + statement
    policy_assessment = {
        "policy-category": category,
        "explanation": ex,
        "final-assessment": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v4(score, category, explanation=None):
    assessment, statement = (
        'Compliant', 'The image is compliant to the provided safety policy.') if 'Acceptable' in score else (
        'Review Needed', 'The image violates the provided safety policy.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.':
        ex = explanation + '. ' + statement
    else:
        ex = explanation + ' ' + statement
    policy_assessment = {
        "image-category": category,
        "assessment": ex,
        "decision": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v5(score, category, explanation=None):
    assessment = 'Compliant' if 'Acceptable' in score else 'Review Needed'
    policy_assessment = {
        "final-assessment": assessment,
    }
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v6(score, category, explanation=None):
    assessment, statement = (
        'Compliant', 'The image is compliant to the provided safety policy.') if 'Acceptable' in score else (
        'Review Needed', 'The image violates the provided safety policy.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.':
        ex = explanation + '. ' + statement
    else:
        ex = explanation + ' ' + statement
    category = str(category)
    policy_assessment = {
        "assessment": assessment,
        "category": mapping[category],
        "explanation": ex,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v7(score, category, explanation=None):
    assessment, statement = (
        'Compliant', 'The image is compliant to the provided safety policy.') if 'Acceptable' in score else (
        'Review Needed', 'The image violates the provided safety policy.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.':
        ex = explanation + '. ' + statement
    else:
        ex = explanation + ' ' + statement

    category = str(category)
    policy_assessment = {
        "image-category": mapping[category],
        "assessment": ex,
        "decision": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v9(score, category, explanation=None):
    cat = mapping[str(category)]
    if score == 'Policy Exception':
        assessment, statement = 'Compliant', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                              f'declared as non-violating. It thus complies with our safety standards.')
    elif 'Acceptable' in score:
        if cat == '10: None applying':
            assessment, statement = 'Compliant', 'This image is compliant with our safety policy.'
        else:
            assessment, statement = 'Compliant', (
                f'The image meets the safety guidelines outlined in category {cat} '
                f'of our safety taxonomy and is considered safe.')
    else:
        assessment, statement = 'Review Needed', (f'The image violates category {cat} of our safety taxonomy and '
                                                  f'requires further review.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.' and explanation[-2] != '.':
        ex = explanation + '. ' + statement
    elif explanation[-1] != ' ':
        ex = explanation + ' ' + statement
    else:
        ex = explanation + statement
    # policy_assessment = {
    #     "assessment": assessment,
    #     "category": cat,
    #     "explanation": ex,
    # }
    policy_assessment = {
        "image-category": cat,
        "assessment": ex,
        "decision": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_and_system_prompt(template_version):
    if template_version == 'json-v0':
        assessment = get_assessment_json_v0
    elif template_version == 'json' or template_version == 'json-v1':
        assessment = get_assessment_json
    elif template_version == 'json-v2':
        assessment = get_assessment_json
    elif template_version == 'json-v3':
        assessment = get_assessment_json_v3
    elif template_version == 'json-v4':
        assessment = get_assessment_json_v4
    elif template_version == 'json-v5':
        assessment = get_assessment_json_v5
    elif template_version == 'json-v6':
        assessment = get_assessment_json_v6
    elif template_version == 'json-v7':
        assessment = get_assessment_json_v7
    elif template_version == 'json-v8':
        assessment = get_assessment_json_v7
    elif template_version == 'json-v9' or template_version == 'json-v10':
        assessment = get_assessment_json_v9
    elif template_version == 'nl':
        assessment = get_assessment
    else:
        raise ValueError(f'Invalid template version: {template_version}')
    system_prompt = policy_system_prompts[template_version] + json_templates[template_version]

    return assessment, system_prompt
