from itertools import product
import pandas as pd


policy_graph = {
    "S1: Violence": {
        "Educational Content": { # violent but educational...
            "edge": "e.g.",
            "examples": ["Charts on violence statistics, and trends", "Informational content on prevalence, causes, and effects of violence", "Legal education", "Memorials and statues of historical figures linked to violence"],
            "rating": "PEGI 3",
            "PEGI tags": []
        },
        # Severity of violence
        "Unrealistic and Implied Violence": {
            "edge": "e.g.",
            "examples": ["Non-realistic violence", "Non-detailed violence"],
            "rating": "PEGI 7",
            "PEGI tags": ["Violence"]
        },
        "Military and Warfare": { # maybe Military and warefare is redundant
            "edge": "e.g.",
            "examples": ["Military", "Warfare", "Soldiers", "Combat", "Battle"],
            "rating": "PEGI 12",
            "PEGI tags": ["Violence"]
        },
        "Mild Violence": {
            "edge": "e.g.",
            "examples": [
                "Violence without any indication of suffering",
                "Violence towards fantasy characters",
                "Non-realistic violence towards human-like characters",
                "Moderate violence towards human characters without blood"
            ],
            "rating": "PEGI 12",
            "PEGI tags": ["Violence"]
        },
        "Violent Acts (animals)": {
            "edge": "e.g.",
            "examples": ["Violence towards animals", "Animal kingdom violence", "Suffering or dead animals", "Caged or suffering animals"],
            "rating": "PEGI 12",
            "PEGI tags": ["Violence"]
        },
        "Realistic Violence": {
            "edge": "e.g.",
            "examples": [
                "Violence against human characters with sight of blood and injuries",
                "Gross violence towards fantasy characters"
            ],
            "rating": "PEGI 16",
            "PEGI tags": ["Violence"]
        },
        # Type of violence
        "Violent Acts (humans)": {
            "edge": "e.g.",
            "examples": ["Violence towards human characters", "Bodily destruction", "Murder", "Torture/Abuse"],
            "rating": "PEGI 16",
            "PEGI tags": ["Violence"]
        },
        # "Violent Acts (property)": {
        #     "edge": "e.g.",
        #     "examples": ["Property damage", "Environmental"],
        #     "rating": "PEGI 16",
        #     "PEGI tags": ["Violence"]
        # },
        "Weapon Usage": {
            "edge": "e.g.",
            "examples": ["Guns in action", "Explosive devices being detonated", "Bioweapons in use", "Nuclear weapons being launched", "Chemical weapons actively released"],
            "rating": "PEGI 16",
            "PEGI tags": ["Violence"]
        },
        "Gross Violence": {
            "edge": "e.g.",
            "examples": [ "Detailed scenes of excessive gore", "Motiveless or gratuitous violence", "Violence against defenseless individuals", "Torture of humans"],
            "rating": "PEGI 18",
            "PEGI tags": ["Violence"]
        },
    },
    "S2: Hate": {
        "Educational Content": {
            "edge": "e.g.",
            "examples": ["Charts on hate statistics, and trends", "Informational content on prevalence, causes, and effects of hate speech and discrimination", "Legal education", "Protests against hate speech and discrimination"],
            "rating": "PEGI 3",
            "PEGI tags": []
        },
        "Mild Swearing": {
            "edge": "e.g.",
            "examples": ["Mild insults"],
            "rating": "PEGI 12",
            "PEGI tags": ["Bad Language"]
        },
        "Offensive Language": {
            "edge": "e.g.",
            "examples": [
                "Vulgarity", 
                "Derogatory comments", 
                "Sexual expletives", 
                "Blasphemy"
            ],
            "rating": "PEGI 16",
            "PEGI tags": ["Bad Language"]
        },
        "Harassment": {
            "edge": "e.g.",
            "examples": [
                "Bullying", "Threats", "Intimidation", "Shaming", "Humiliation",
                "Insults/Personal attacks", "Abuse", "Provoking", "Trolling", "Doxxing", "Cursing"
            ],
            "rating": "PEGI 16",
            "PEGI tags": ["Bad Language"]
        },
        "Discrimination": {
            "edge": "e.g. against",
            "examples": [
                "Race", "Color", "Ethnicity", "National Origin", "Disability",
                "Religious Affiliation", "Caste", "Sexual Orientation", "Sex", "Gender Identity",
                "Serious Disease", "Negative Stereotypes"
            ],
            "rating": "PEGI 18",
            "PEGI tags": ["Discrimination"]
        },

    },
    "S3: Nude Content": {
        "Educational Content": {
            "edge": "e.g.",
            "examples": ["Charts on statistics, and trends", "Artistic nudity", "Sex education materials"],
            "rating": "PEGI 12",
            "PEGI tags": []
        },
        "Sexual Innuendo": {
            "edge": "e.g.",
            "examples": ["Suggestive posing and dancing", "Sexual humor"],
            "rating": "PEGI 12",
            "PEGI tags": ["Sex"]
        },
        "Mild Nudity Content": {
            "edge": "i.e.",
            "examples": ["Sexual intercourse without exposed genitals or breasts", "Erotic nudity without visible genitalia or exposed breasts", "Characters in revealing clothing such as lingerie or undergarments", "Romantic Scenes with kissing"],
            "rating": "PEGI 16",
            "PEGI tags": ["Sex"]
        },
        "Explicit Sexual Content": {
            "edge": "i.e.",
            "examples": ["Obscenity", "Sexual intercourse with visible genitalia or exposed breasts"],
            "rating": "PEGI 18",
            "PEGI tags": ["Sex"]
        },
        "Explicit Nudity Content": {
            "edge": "i.e.",
            "examples": ["Nudity with visible genitalia", "Nudity with exposed breasts"],
            "rating": "PEGI 18",
            "PEGI tags": ["Sex"]
        },
        "Monetized Nudity": {
            "edge": "e.g.",
            "examples": ["Pornography", "Promotion of sexual services", "Prostitution"],
            "rating": "PEGI 18",
            "PEGI tags": ["Sex"]
        }
    },
    "S4: Disturbing Content": {
        "Scary Content": {
            "edge": "e.g.",
            "examples": ["Frightening Content", "Scary Situations", "Disasters and Emergencies"],
            "rating": "PEGI 7",
            "PEGI tags": ["Fear"]
        },
        "Moderately Disturbing Content": {
            "edge": "e.g.",
            "examples": [ "Mild gore", "Monsters and Zombies", "Jump scares", "Moderate horror sequences", "Characters in danger", "Detailed injury", "Death"],
            "rating": "PEGI 12",
            "PEGI tags": ["Fear"]
        },
        "Non-realistic Weapons": {
            "edge": "e.g.",
            "examples": ["Fantasy weapons", "Cartoonish or fictional weapons", "Non-detailed weapons"],
            "rating": "PEGI 12",
            "PEGI tags": []
        },
        "Highly Disturbing Content": {
            "edge": "e.g.",
            "examples": ["Intense horror sequences", "Strong gore or suffering", "Scenes of extreme emotional distress"],
            "rating": "PEGI 16",
            "PEGI tags": ["Fear"]
        },
        "Weapon Depictions": {
            "edge": "e.g.",
            "examples": ["Licensed weapons", "Non-licensed weapons", "Guns", "Explosives", "Bioweapons", "Nuclear weapons", "Chemical weapons", "Other weapons"],
            "rating": "PEGI 16",
            "PEGI tags": []
        },
    },
    "S5: Self-Harm": {
        "Educational Content": {
            "edge": "e.g.",
            "examples": ["Charts on statistics, and trends", "Educational content on self-harm and suicide prevalence, causes, and effects"],
            "rating": "PEGI 3",
            "PEGI tags": []
        },
        "Non-realistic Self-harm": {
            "edge": "e.g.",
            "examples": ["Self-harm in cartoons"],
            "rating": "PEGI 16",
            "PEGI tags": []
        },
        "Suicide": {
            "edge": "e.g.",
            "examples": ["Suicide depiction"],
            "rating": "PEGI 18",
            "PEGI tags": []
        },
        "Self-injury": {
            "edge": "e.g.",
            "examples": ["Cutting", "Disordered Eating"],
            "rating": "PEGI 18",
            "PEGI tags": []
        },
    },
    "S6: Criminal Activities": {
        "Educational Content": {
            "edge": "e.g.",
            "examples": ["Charts on statistics, and trends", "Crime prevention strategies", "Legal education"],
            "rating": "PEGI 3",
            "PEGI tags": []
        },
        "Property Crimes": {
            "edge": "e.g.",
            "examples": ["Burglary", "Arson", "Vandalism"],
            "rating": "PEGI 16",
            "PEGI tags": []
        },
        "Cyber Crimes": {
            "edge": "e.g.",
            "examples": ["Hacking", "Spyware"],
            "rating": "PEGI 16",
            "PEGI tags": []
        },
        "Deception": {
            "edge": "e.g.",
            "examples": ["Fraud", "Scams"],
            "rating": "PEGI 16",
            "PEGI tags": []
        },
        "Supporting Malicious Groups": {
            "edge": "e.g.",
            "examples": ["Terrorism", "Extremism", "Criminal organization"],
            "rating": "PEGI 18",
            "PEGI tags": []
        },
        "Sex Crimes": {
            "edge": "e.g.",
            "examples": ["Sexual assault", "Sexual harassment", "Rape", "Groping", "Human trafficking"],
            "rating": "PEGI 18",
            "PEGI tags": []
        },
        "Financial Crimes": {
            "edge": "e.g.",
            "examples": ["Money laundering"],
            "rating": "PEGI 18",
            "PEGI tags": []
        },
        "Weapons Crimes": {
            "edge": "e.g.",
            "examples": ["Producing unlicensed firearms"],
            "rating": "PEGI 18",
            "PEGI tags": []
        },
    },
    "S7: Regulated Substances": {
        "Educational Content": {
            "edge": "e.g.",
            "examples": ["Charts on statistics, and trends", "Educational content that informs about drug use and prevention"],
            "rating": "PEGI 3",
            "PEGI tags": []
        },
        "Medication": {
            "edge": "i.e.",
            "examples": ["Legal drug use"],
            "rating": "PEGI 3",
            "PEGI tags": []
        },
        "Alcohol": {
            "edge": "e.g.",
            "examples": ["Alcohol consumption", "Alcoholic beverages"],
            "rating": "PEGI 16",
            "PEGI tags": ["Drugs"]
        },
        "Tobacco": {
            "edge": "e.g.",
            "examples": ["Tobacco consumption", "Tobacco products"],
            "rating": "PEGI 16",
            "PEGI tags": ["Drugs"]
        },        
        "Cannabis and Other Drugs": {
            "edge": "e.g.",
            "examples": ["Cannabis", "Other illegal drugs"],
            "rating": "PEGI 16",
            "PEGI tags": ["Drugs"]
        },
    # "Glamorization of Illegal Drugs": {
        "Glamorization of Drug Use": {
            "edge": "e.g.",
            "examples": ["Portraying drug use as exciting, glamorous, or beneficial", "Linking drug use to success or happiness", "Promoting drug use in a positive or appealing manner"],
            "rating": "PEGI 18",
            "PEGI tags": ["Drugs"]
        },
    },
    "S8: Economic Harm": {
        "Educational Content": {
            "edge": "e.g.",
            "examples": ["Charts on statistics, and trends", "Information on financial risks and products", "Responsible gambling education", "Informational content on debt management"],
            "rating": "PEGI 3",
            "PEGI tags": []
    },
        "High-Risk Financial Activities": {
            "edge": "e.g.",
            "examples": ["Gambling", "Payday lending"],
            "rating": "PEGI 18",
            "PEGI tags": ["Gambling"]
        }
    },
    "S9: Child Exploitation": {
        "Child Endangerment": {
            "edge": "e.g.",
            "examples": ["Grooming", "Exploiting minors"],
            "rating": "Illegal",
            "PEGI tags": []
        },
        "Child Sexual Abuse": {
            "edge": "e.g.",
            "examples": ["Solicitation", "CSAM"],
            "rating": "Illegal",
            "PEGI tags": []
        }
    }
}

def get_policy_intro(policy_name: str = "Safety Policy"):
    return f"Provide a safety assessment for the user content. Determine compliance or violation against the safety policy by reviewing the following categories:"

def get_policy_assessment(categories: list):
    return f'''Assessment Steps:
1. Determine the overall safety "rating" of the content. If the content falls into any prohibited category of the policy, mark it as "Unsafe". Otherwise, mark it as "Safe".
2. Identify all relevant content "categories" within the policy, regardless of whether they are decleared as permitted or prohibited. If no categories apply, select 'NA: None applicable'.
3. Provide a "Rationale". Justify why the content is considered safe or unsafe by describing the content and citing any violations of prohibited categories. If there are no violations, provide a brief explanation of why the content is safe.

To provide your assessment use the following json template:
{{
    "rating": str<"Unsafe"|"Safe">,
    "categories": str<"{'"|"'.join(categories)}">,
    "rationale": str,
}}
<image>
'''

def remove_numbers_from_categories(categories: list):
    cats = categories
    if categories == []:
        return []
    if '.' in categories[0]:
        cats = [category.split('. ')[1] for category in categories]
    return cats

def remove_mutually_exclusive_categories(categories: list):
    categories = remove_numbers_from_categories(categories)
    violence = ['Unrealistic and Implied Violence', 'Mild Violence', 'Realistic Violence','Gross Violence']
    hate = ['Mild Swearing', 'Offensive Language',]
    nudity = ['Sexual Innuendo', 'Mild Nudity Content', 'Explicit Nudity Content','Explicit Sexual Content', ]
    disturbing = ['Scary Content', 'Moderately Disturbing Content', 'Highly Disturbing Content']
    self_harm = ['Non-realistic Self-harm', 'Self-injury', 'Suicide']
    categories_to_be_removed = []
    for exclusive_list in [violence, hate, nudity, disturbing, self_harm]:
        for c_id in range(len(exclusive_list) - 1, -1, -1):
            if exclusive_list[c_id] in categories:
                categories_to_be_removed += exclusive_list[:c_id]
                break
    cleaned_list = [category for category in categories if category not in categories_to_be_removed]
    return cleaned_list



  
def remove_mutually_exclusive_categories_df(df: pd.DataFrame):
    '''

    '''
    categories = get_content_categories(policy_graph)
    if len(df.columns) != len(categories):
        raise ValueError('Pass dict with only category entries')
    cols_ = df.columns
    df.columns = [c.split('. ')[-1] for c in df.columns]
    # reindex the dataframe
    df = df.reset_index(drop=True)
    
    for category in df.columns:
        for row in range(len(df)):
            entry = df.loc[row, category]                
            entry = entry.iloc[0] if isinstance(entry, pd.Series) else entry
            # print(entry)
            if entry == 1:
                # violence
                if category == 'Gross Violence':
                    df.loc[row, 'Realistic Violence'] = 0
                    df.loc[row, 'Mild Violence'] = 0
                    df.loc[row, 'Unrealistic and Implied Violence'] = 0
                elif category == 'Realistic Violence':
                    df.loc[row, 'Mild Violence'] = 0
                    df.loc[row, 'Unrealistic and Implied Violence'] = 0
                elif category == 'Mild Violence':
                    df.loc[row, 'Unrealistic and Implied Violence'] = 0
                elif category == 'Offensive Language':
                    df.loc[row, 'Mild Swearing'] = 0
                # nudity
                elif category == 'Explicit Sexual Content':
                    df.loc[row, 'Explicit Nudity Content'] = 0
                    df.loc[row, 'Mild Nudity Content'] = 0
                    df.loc[row, 'Sexual Innuendo'] = 0
                elif category == 'Explicit Nudity Content':
                    df.loc[row, 'Mild Nudity Content'] = 0
                    df.loc[row, 'Sexual Innuendo'] = 0
                elif category == 'Mild Nudity Content':
                    df.loc[row, 'Sexual Innuendo'] = 0
                # disturbing
                elif category == 'Highly Disturbing Content':
                    df.loc[row, 'Moderately Disturbing Content'] = 0
                    df.loc[row, 'Scary Content'] = 0
                elif category == 'Moderately Disturbing Content':
                    df.loc[row, 'Scary Content'] = 0
                # self harm
                elif category == 'Suicide':
                    df.loc[row, 'Self-injury'] = 0
                    df.loc[row, 'Non-realistic Self-harm'] = 0
                elif category == 'Self-injury':
                    df.loc[row, 'Non-realistic Self-harm'] = 0
                elif category not in get_content_categories(policy_graph=policy_graph):
                    raise ValueError('Unkown Content category:' + category)
                else:
                    pass
            elif entry == 0:
                pass
            else:
                raise ValueError('Unkown value: ' + entry)
    df.columns = cols_
    return df


def policy_graph_to_safety_policy(pegi_score: int, policy_graph:dict = policy_graph):
    categories = list(policy_graph.keys())
    subcategories = []
    intro = get_policy_intro()
    policy_text = intro + '\n'
    subcategory_count = 1
    for category, category_details in policy_graph.items():
        policy_text += category + '\n'
        allowed_content = "Permitted content categories:\n"
        prohibited_content = "Prohibited content categories:\n"
        for subcategory, graph_info in category_details.items():
            subcategory_pegi_score = graph_info["rating"].split(' ')[-1]
            subcategory_with_number = f'{subcategory_count}. {subcategory}'
            if subcategory_pegi_score == "Illegal" or pegi_score < int(subcategory_pegi_score):
                prohibited_content += f'- "{subcategory_with_number}" {graph_info["edge"]} {", ".join(graph_info["examples"])}\n'
            else:
                allowed_content += f'- "{subcategory_with_number}" {graph_info["edge"]} {", ".join(graph_info["examples"])}\n'
            subcategory_count += 1
            subcategories.append(subcategory_with_number)
        if allowed_content == "Permitted content categories:\n":
            allowed_content += "None\n"
        if prohibited_content == "Prohibited content categories:\n":
            prohibited_content += "None\n"
        policy_text +=  allowed_content + prohibited_content + '\n'

    policy_text += get_policy_assessment(subcategories)
    return policy_text

# print(policy_graph_to_text(12))


def policy_graph_to_safety_policy_v2(policy_graph:dict = policy_graph):
    
    subcategories = []
    intro = "Given the following safety policy determine which of the categories are present within the users content:\n"
    policy_text = ''
    subcategory_count = 0
    for category, category_details in policy_graph.items():
        policy_text += category + '\n'
        for subcategory, graph_info in category_details.items():
            subcategory_count += 1
            subcategory_with_number = f'{subcategory_count}. {subcategory}'
            content += f'- "{subcategory_with_number}" {graph_info["edge"]} {", ".join(graph_info["examples"])}\n'
            subcategories.append(subcategory_with_number)
        policy_text +=  content + '\n'
    return intro + policy_text + get_policy_assessment(subcategories)


def get_content_categories(policy_graph):
    content_categories = []
    for safety_category_details in policy_graph.values():
        content_categories += list(safety_category_details.keys())
    return content_categories


def get_content_categories_with_numbers(policy_graph):
    content_categories = []
    subcategory_count = 1
    for safety_category_details in policy_graph.values():
        for content_category, graph_info in safety_category_details.items():
            content_categories.append(f'{subcategory_count}. {content_category}')
            subcategory_count += 1
    return content_categories


def get_content_categories_with_examples(policy_graph):
    content_categories = []
    for safety_category_details in policy_graph.values():
        for content_category, graph_info in safety_category_details.items():
            content_categories += [f"{content_category} {graph_info['edge']} {', '.join(graph_info['examples'])}"]
    return content_categories

def get_safety_categories(policy_graph):
    return list(policy_graph.keys())


def get_majority_vote(violations):
    if isinstance(violations, pd.DataFrame):
        return get_majority_vote_from_df(violations)
    num_entries = len(violations)
    all_values = [v for value in violations.values() for v in value]
    unique_values = set(all_values)
    voting = []
    for value in unique_values:
        if all_values.count(value) >= num_entries/2:
            voting.append(value)
    voting.sort()
    return voting

def get_majority_vote_from_df(violations: pd.DataFrame):
    # get columns of the dataframe
    categories = get_content_categories_with_numbers(policy_graph)
    num_entries = len(violations)
    data = []
    for category in categories:
        votings = violations[category].sum()
        votings = votings[0] if isinstance(votings, pd.Series) else votings
        if votings >= num_entries/2:
            data += [1]
        else:
            data += [0]
    df = pd.DataFrame(columns=categories, data=[data], index=[0])
    return df
