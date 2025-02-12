from llavaguard_config import llava_guard_config

def get_default_policy(template_version):
    '''
    This function will return a custom system prompt.
    We drop the violation category from the model prompt changing the safety label to “Compliant”.
    '''
    template_version = f'json-{template_version}' if 'json' not in template_version else template_version
    policy = llava_guard_config[template_version]['policy_start_sentence']
    c_pol_dict = llava_guard_config[template_version]['policy_dict']
    if c_pol_dict == {}:
        print('llavaguard policy version is not flexible')
    
    for key, value in c_pol_dict.items():
        policy += key + ': \n'
        policy += value[0] + '\n'
        
    if '<image>' not in policy and '<image>' not in llava_guard_config[template_version]['response_template']:
        policy += '<image>\n' + llava_guard_config[template_version]['response_template']
    else:
        policy += llava_guard_config[template_version]['response_template']
    return policy



def get_safety_categories(template_version):
    '''
    This function will return the safety_categories given a template version
    '''
    if template_version in llava_guard_config.keys():
        return llava_guard_config[template_version]['safety_categories']
    else:
        raise ValueError(f'Invalid template version: {template_version}')


def get_assessment_and_system_prompt(template_version):
    '''
    This function will return the assessment function and construct the corresponding prompt
    template_version: template version for which we get assessment function and construct the corresponding prompt
    returns: (assessment_funct, policy_prompt)
    policy_prompt: test
    assessment_funct: funct(score, category, explanation) - return assessmentstext for provided values
    '''
    template_version = f'json-{template_version}' if 'json' not in template_version else template_version

    
    if template_version in llava_guard_config.keys():
        assessment = llava_guard_config[template_version]['assessment_function']
    else:
        raise ValueError(f'Invalid template version: {template_version}')
    policy = get_default_policy(template_version)
    return assessment, policy
