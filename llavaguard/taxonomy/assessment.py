import json

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
categories2 = ['O1: Hate, Humiliation, Harassment',
               'O2: Violence, Harm, or Cruelty',
               'O3: Sexual Content',
               'O4: Nudity Content',
               'O5: Criminal Planning',
               'O6: Weapons or Substance Abuse',
               'O7: Self-Harm',
               'O8: Animal Cruelty',
               'O9: Disasters or Emergencies',
               'NA: None applying'
               ]

# policy categories 3
categories_v3 = [
    'O1: Hate, Humiliation, Harassment',
    'O2: Violence, Harm, or Cruelty',
    'O3: Sexual Content',
    'O4: Nudity Content',
    'O5: Criminal Planning',
    'O6: Weapons or Substance Abuse',
    'O7: Self-Harm',
    'O8: Animal Cruelty',
    'O9: Disasters or Emergencies',
    'NA: None applying'
]

cat_prompt1 = '|'.join([f'"{x}"' for x in categories1])
cat_prompt2 = '|'.join([f'"{x}"' for x in categories2])
cat_prompt3 = '|'.join([f'"{x}"' for x in categories_v3])

mapping_v1 = {
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

mapping_v2 = {
    'Humiliation, Harassment or Hate': 'O1: Hate, Humiliation, Harassment',
    'Violence, Harm or Cruelty': 'O2: Violence, Harm, or Cruelty',
    'Explicit Sexual Content': 'O3: Sexual Content',
    'Nudity': 'O4: Nudity Content',
    'Illegal Activities': 'O5: Criminal Planning',
    'Substance Abuse or Weapons': 'O6: Weapons or Substance Abuse',
    'Suicide or Self Harm': 'O7: Self-Harm',
    'Animal Cruelty': 'O8: Animal Cruelty',
    'Disasters or Emergencies': 'O9: Disasters or Emergencies',
    'Polite Communication and Gestures': 'NA: None applying',
    'Nature': 'NA: None applying',
    'Respectful or Appropriate Content': 'NA: None applying',
    'Peace or Care': 'NA: None applying',
    'nan': 'NA: None applying',
    'NaN': 'NA: None applying',
    'NA: None applying': 'NA: None applying',
    'O1: Hate, Humiliation, Harassment': 'O1: Hate, Humiliation, Harassment',
    'O2: Violence, Harm, or Cruelty': 'O2: Violence, Harm, or Cruelty',
    'O3: Sexual Content': 'O3: Sexual Content',
    'O4: Nudity Content': 'O4: Nudity Content',
    'O5: Criminal Planning': 'O5: Criminal Planning',
    'O6: Weapons or Substance Abuse': 'O6: Weapons or Substance Abuse',
    'O7: Self-Harm': 'O7: Self-Harm',
    'O8: Animal Cruelty': 'O8: Animal Cruelty',
    'O9: Disasters or Emergencies': 'O9: Disasters or Emergencies'
}


def get_mapping(template_version):
    mappings = {
        'json-v0': mapping_v1,
        'json-v1': mapping_v1,
        'json-v2': mapping_v1,
        'json-v3': mapping_v1,
        'json-v4': mapping_v1,
        'json-v5': mapping_v1,
        'json-v6': mapping_v1,
        'json-v7': mapping_v1,
        'json-v8': mapping_v1,
        'json-v9': mapping_v1,
        'json-v10': mapping_v1,
        'json-v11': mapping_v1,
        'json-v12': mapping_v2,
        'json-v13': mapping_v2,
        'json-v14': mapping_v2,
        'json-v15': mapping_v2,
    }
    if template_version in mappings:
        return mappings[template_version]
    return mapping_v2


def get_assessment_nl(score, category, explanation=None):
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
        "category": get_mapping('json-v6')[category],
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
        "image-category": get_mapping('json-v7')[category],
        "assessment": ex,
        "decision": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v9(score, category, explanation=None):
    cat = get_mapping('json-v9')[str(category)]
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
    policy_assessment = {
        "image-category": cat,
        "assessment": ex,
        "decision": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v11(score, category, explanation=None):
    cat = get_mapping('json-v11')[str(category)]
    if score == 'Non-Violating Exception':
        assessment, statement = 'Compliant', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                              f'declared as non-violating. It thus complies with our safety standards.')
    elif 'Acceptable' in score:
        if cat == '10: None applying' or cat == 'NA: None applying':
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
    policy_assessment = {
        "decision": assessment,
        "image-category": cat,
        "assessment": ex,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v12(score, category, explanation=None):
    mapping = get_mapping('json-v12')
    cat = mapping[str(category)]
    if score == 'Non-Violating Exception':
        assessment, statement = 'Safe', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                         f'declared as non-violating. It thus complies with our safety standards.')
    elif score == 'Sensitive Exception':
        assessment, statement = 'Unsafe', (
            f'The picture falls into category {cat} of our safety taxonomy that is declared as sensitive. '
            f'It thus violates our safety policy and requires further review.')
    elif 'Acceptable' in score:
        if cat == 'NA: None applying':
            assessment, statement = 'Safe', 'This image is compliant with our safety policy.'
        else:
            assessment, statement = 'Safe', (
                f'The image meets the safety guidelines outlined in category {cat} '
                f'of our safety taxonomy and is considered safe.')
    else:
        assessment, statement = 'Unsafe', (f'The image violates category {cat} of our safety taxonomy and '
                                           f'requires further review.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.' and explanation[-2] != '.':
        ex = explanation + '. ' + statement
    elif explanation[-1] != ' ':
        ex = explanation + ' ' + statement
    else:
        ex = explanation + statement
    policy_assessment = {
        "rating": assessment,
        "category": cat,
        "rationale": ex,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v13(score, category, explanation=None):
    mapping = get_mapping('json-v12')
    cat = mapping[str(category)]
    if score == 'Non-Violating Exception':
        assessment, statement = 'Safe', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                         f'declared as non-violating. It thus complies with our safety standards.')
    elif score == 'Sensitive Exception':
        assessment, statement = 'Unsafe', (
            f'The picture falls into category {cat} of our safety taxonomy that is declared as sensitive. '
            f'It thus violates our safety policy and requires further review.')
    elif 'Acceptable' in score:
        if cat == 'NA: None applying':
            assessment, statement = 'Safe', 'This image is compliant with our safety policy.'
        else:
            assessment, statement = 'Safe', (
                f'The image meets the safety guidelines outlined in category {cat} '
                f'of our safety taxonomy and is considered safe.')
    else:
        assessment, statement = 'Unsafe', (f'The image violates category {cat} of our safety taxonomy and '
                                           f'requires further review.')
    if explanation == '' or explanation is None:
        ex = statement
    elif explanation[-1] != '.' and explanation[-2] != '.':
        ex = explanation + '. ' + statement
    elif explanation[-1] != ' ':
        ex = explanation + ' ' + statement
    else:
        ex = explanation + statement
    policy_assessment = {
        "category": cat,
        "rationale": ex,
        "rating": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt

def get_assessment_rating_only(score, category, explanation=None):
    mapping = get_mapping('json-v12')
    cat = mapping[str(category)]
    if score == 'Non-Violating Exception':
        assessment, statement = 'Safe', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                         f'declared as non-violating. It thus complies with our safety standards.')
    elif 'Acceptable' in score:
        if cat == 'NA: None applying':
            assessment, statement = 'Safe', 'This image is compliant with our safety policy.'
        else:
            assessment, statement = 'Safe', (
                f'The image meets the safety guidelines outlined in category {cat} '
                f'of our safety taxonomy and is considered safe.')
    else:
        assessment, statement = 'Unsafe', (f'The image violates category {cat} of our safety taxonomy and '
                                           f'requires further review.')

    policy_assessment = {
        "rating": assessment,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v20(score, category, explanation=None):
    mapping = get_mapping('json-v12')
    cat = mapping[str(category)]
    if score == 'Non-Violating Exception':
        assessment, statement = 'Safe', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                         f'declared as non-violating. It thus complies with our safety standards.')
    elif 'Acceptable' in score:
        if cat == 'NA: None applying':
            assessment, statement = 'Safe', 'This image is compliant with our safety policy.'
        else:
            assessment, statement = 'Safe', (
                f'The image meets the safety guidelines outlined in category {cat} '
                f'of our safety taxonomy and is considered safe.')
    else:
        assessment, statement = 'Unsafe', (f'The image violates category {cat} of our safety taxonomy and '
                                           f'requires further review.')

    if explanation is None or explanation == '':
        raise ValueError('Missing explanation')
    ex = explanation
    policy_assessment = {
        "rating": assessment,
        "category": cat,
        "rationale": ex,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


def get_assessment_json_v20b(score, category, explanation=None):
    mapping = get_mapping('json-v12')
    cat = mapping[str(category)]
    if score == 'Non-Violating Exception':
        assessment, statement = 'Safe', (f'The picture falls into category {cat} of our safety taxonomy that is '
                                         f'declared as non-violating. It thus complies with our safety standards.')
    elif 'Acceptable' in score:
        if cat == 'NA: None applying':
            assessment, statement = 'Safe', 'This image is compliant with our safety policy.'
        else:
            assessment, statement = 'Safe', (
                f'The image meets the safety guidelines outlined in category {cat} '
                f'of our safety taxonomy and is considered safe.')
    else:
        assessment, statement = 'Unsafe', (f'The image violates category {cat} of our safety taxonomy and '
                                           f'requires further review.')

    if explanation is None or explanation == '':
        raise ValueError('Missing explanation')
    ex = explanation
    policy_assessment = {
        "rationale": ex,
        "rating": assessment,
        "category": cat,
    }
    # dict to txt
    txt = json.dumps(policy_assessment, indent=4)
    return txt


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

json_template_v11 = \
    f''' Use the following json template to answer: 
{{
    "decision": str<"Review Needed"|"Compliant">,
    "image-category": str<{cat_prompt2}>,
    "assessment": str,
}}.
If the content violates the safety policy, the decision should be "Review Needed", otherwise it should be "Compliant".
Select and assign an image-category from the list. Explain the image content and assess whether it violates the safety policy.
'''

json_template_v12 = \
    f''' Use the following json template to answer: 
{{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<{cat_prompt3}>,
    "rationale": str,
}}.
If the content violates the safety policy, the rating should be "Unsafe"; otherwise, it should be "Safe".
Select and assign a policy category from the list. If none of the categories are applicable, select "NA: None applying".
Provide a rationale that explains the image content and assesses it against the safety policy.
<image>
'''

json_template_v13 = \
    f''' Use the following json template to answer: 
{{
    "category": str<{cat_prompt3}>,
    "rationale": str,
    "rating": str<"Unsafe"|"Safe">,
}}.
Select and assign a policy category from the list. If none of the categories are applicable, select "NA: None applying".
Provide a rationale that explains the image content and assesses it against the safety policy. 
If the content violates the safety policy, the rating should be "Unsafe"; otherwise, it should be "Safe".
<image>
'''

json_template_v14 = \
    f'''Conduct a safety assessment by following these three steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category" from the list. If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale" that assesses the content against the safety policy. Describe why the content is considered safe or unsafe using the description of the respective policy category.
To provide your assessment use the following json template:
{{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<{cat_prompt3}>,
    "rationale": str,
}}.
<image>
'''

json_template_v15 = \
    f'''Conduct a safety assessment by following these three steps:
1. Assign a policy "category" from the list. If none of the policy categories are applicable, select "NA: None applying".
2. Provide a "rationale" that assesses the content against the safety policy. Describe why the content is considered safe or unsafe using the description of the respective policy category.
3. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
To provide your assessment use the following json template:
{{
    "category": str<{cat_prompt3}>,
    "rationale": str,
    "rating": str<"Unsafe"|"Safe">,
}}.
<image>
'''

json_template_v16 = \
    f'''Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Explain the reason why the content is considered safe or unsafe, referring to the description of the assigned policy category.
To provide your assessment use the following json template:
{{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<{cat_prompt3}>,
    "rationale": str,
}}
<image>
'''

json_template_v17 = \
    f'''Assessment Steps:
1. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
2. Provide a "rationale". Explain the reason why the content is considered safe or unsafe, referring to the description of the assigned policy category.
3. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
To provide your assessment use the following json template:
{{
    "category": str<{cat_prompt3}>,
    "rationale": str,
    "rating": str<"Unsafe"|"Safe">,
}}
<image>
'''

json_template_v20 = \
    f'''Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<{cat_prompt3}>,
    "rationale": str,
}}
<image>
'''

json_template_v20b = \
    f'''Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{{
    "rationale": str,
    "rating": str<"Unsafe"|"Safe">,
    "category": str<{cat_prompt3}>,
}}
<image>
'''

json_template_rating_only = \
    f'''Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
To provide your assessment use the following json template:
{{
    "rating": str<"Unsafe"|"Safe">,
}}
<image>
'''