
'''
the user content should not include XXXX e.g. YYYY.

the user content can contain XXXX such as YYYY.

S1: Violence & Extremism
 -> crimes implicates unsafe content, football game can be violent but not unsafe and should not be included in this category
 
S9: Indiscriminate Weapons
suggestion: Indiscriminate Weapons -> Weapons
-> maybe we can make it more general and include all weapons in one category 

S? Regulated Substances
-> is currently in the Non-Violent Crimes category, maybe we can move it to a new category called Regulated Substances or Drugs

# for some categories we need maybe more explicit lines, e.g. nude images of adults are allowed if no genitals are visible 
https://gamesratingauthority.org.uk/RatingBoard/ratings-system
'''

safety_policy = {
    "S1: Violence": {
        "Sub-Categories": {
            # severity of violence
            "Unrealistic and Implied violence": ["Non-realistic violence", "Non-detailed violence"],
            "Mild violence": ["Violence towards fantasy characters", "Non-realistic violence towards human-like characters", "Moderate violence towards human characters without blood"],
            "Realistic violence": ["violence against human characters, including sight of blood and injuries", "Gross violence towards fantasy characters"],
            "Gross violence": ["Motiveless violence", "Violence towards defenseless characters or vulnerable human characters", "Torture", "Celebrating Suffering", "Glorifying violence, abuse, or the suffering of others", "Belittling victimhood or violent events", "Denying well-documented, major violent events or the victimhood of such events/Denying the deeds of martyrdom", "Beautifying and Whitewashing acts of war or aggression"],
            # type of violence
            "Violent Acts (persons)": ["Violence towards persons", "Murder", "Bodily destruction", "Bodily mutilation", "Torture/Abuse", "Genocide", "Activities meant to kill"],
            "Violent Acts (animals)": [],
            "Violent Acts (poperty)": ["Property damage", "Environmental"],
            "Weapon Usage": ["Guns", "Explosives/Dangerous materials", "Bioweapons/Viruses/Gain-of-function", "Nuclear Weapons", "Chemical Weapons", "Radiological Weapons"],
            "Military and Warfare": ["Military", "Warfare"],
            "Sport-related violance": [], # 
            "Educational Content": ["Historical Events", "legal, safe, and educational contexts", "Discuss firearm ownership, historical use, and weapon laws without promoting illegality", "Discussing military history and warfare without promoting violence"],
        },
        "levels": {
            "PEGI 3": ["Educational Content","comical context", "childlike setting"],
            "Violance (PEGI 7)": ["Implied violence"],
            "Violance (PEGI 12)": ["Mild violence"],
            "Violance (PEGI 16)": ["Realistic violence", "Violent Acts (persons)", "Violent Acts (animals)", "Violent Acts (poperty)", "Weapon Usage", "Military and Warfare", "Sport-related violance"],
            "Violance (PEGI 18)": ["Gross violence"],
        }
    },
    "S2: Hate": {
        "Sub-Categories": {
            "Mild swearing": ["mild swearing", "mild insults", "mild vulgarity"],
            "Offensive Language": ["Vulgarity", "Derogatory comments", "Blasphemy", "Defmation", "Sexual expletives"],
            "Harassment": ["Bullying", "Threats", "Intimidation", "Shaming", "Humiliation", "Insults/Personal attacks", "Abuse", "Provoking", "Trolling", "Doxxing", "Cursing"],
            "Discrimination": ["Race", "Color", "Ethnicity", "National Origin", "Disability", "Religious Affiliation", "Caste", "Sexual Orientation", "Sex", "Gender Identity", "Serious Disease", "Negative Stereotypes"],
            "Perpetuating Harmful Beliefs": ["Negative stereotyping of any group", "Perpetuating racism", "Perpetuating sexism"],
            "Educational Content": ["Historical Events", "Discussing hate speech and its impact", "Discussing discrimination and its impact", "Discussing harmful beliefs and stereotypes", "Discussing the impact of offensive language"]
        },
        "levels": {
            "Bad language (PEGI 12)": ["Mild Swearing"],
            "Bad language (PEGI 16)": ["Offensive Language", "Harassment"], # milder squaring is allowed
            "Bad language (PEGI 18)": ["Offensive Language", "Harassment"], # stronger language is allowed  including the crudest sexual expletives.  
            "Discrimination (PEGI 18)": ["Discrimination", "Perpetuating Harmful Beliefs"],
        }
    },
    "S3: Sensitive/Nude Content": {
        "Sub-Categories": {
            "Sexual Innuendo": ["Sexual posturing", "Sexual humor", "Couple getting into bed", "Suggestive posing and dancing"],
            "Mild Sexual Content": ["without visible genitals", "without exposed breast", "without exposed nipples"],
            "Explicit Sexual Content": ["Obscenity", "Suggestive", "Sexual acts", "Sexual intercourse"],
            "Explicit Nudity Content": ["Visible genitalia", "Exposed breast", "Exposed nipples"],
            "Monetized": ["Pornography", "Promotion of sexual services", "Prostitution"],
        },
        'levels': {
            "Sex (PEGI 12)": ["Sexual Innuendo"],
            "Sex (PEGI 16)": ["Mild Sexual Content"],
            "Sex (PEGI 18)": ["Explicit Sexual Content", "Explicit Nudity Content", "Monetized"],
        }
        
    },
    "S4: Disturbing Content": {
        "Sub-Categories": {
            "Scary Content": ["Frightening Content", "Scary Situations"],
            "Moderatly Disturbing Content": ["Moderate Horror", "Moderate horror sequences", "characters in danger", "jump scares", "Gore", "Blood", "Injury", "Death", "Zombies", "Monsters", "Jump scares"],
            "Sustained Disturbing Content": ["Intense and sustained horror sequences", "strong gory images"],
            # what to do with the following?
            "Weapon Depictions": ["Unlicensed Weapons", "Licensed Weapons"],
            "Non-realistic Weapons": ["non-detailed weapons", "fantasy weapons", "cartoonish or clearly fictional weapons"],
        },
        "levels": {
            "Fear (PEGI 7)": ["Scary Content"],
            "Fear (PEGI 12)": ["Moderatly Disturbing Content"],
            "Fear (PEGI 16)": ["Sustained Disturbing Content"],
            "PEGI 12": ["Non-realistic Weapons"],
            "PEGI 16": ["Weapons"],
        }
    },
    "S5: Self-Harm": {
        "Sub-Categories": {
            "Suicide": [],
            "Self-injury": ["Cutting", "Disordered Eating"],
            "non-realistic Self-harm": [],
        },
        "level": {
            "PEGI 16": ["Non-realistic Self-harm"],
            "PEGI 18": ["Suicide", "Self-injury"], # either 16 or 18 i would say
        }
    },
    "S6: Criminal Activities": {
        "Sub-Categories": {
            "Property Crimes": ["Burglary", "Robbery", "Arson", "Vandalism"],
            "Cyber Crimes": ["Hacking", "Spyware", "Malware"],
            "Deception": ["Fraud", "Scams", "Phishing/Catﬁshing", "Pseudo-pharmaceuticals", "Impersonating others", "Academic Dishonesty", "Plagiarism", "Mis/disinformation", "Fake online engagement (fake reviews, fake grassroots support)"],
            #
            "Supporting Malicious Organized Groups": ["Terrorism", "Extremism", "Criminal Organization"],
            "Sex Crimes": ["Sexual Assault", "Sexual Harassment", "Sexual exploitation", "Rape", "Groping", "Human trafficking"],
            "Financial Crimes": ["Money Laundering"],
            "Weapons Crimes": ["Producing Unlicensed Firearms"],
            "Educational Content": ["Historical Events", ],
        },
        "levels": {
            "PEGI 16": ["Property Crimes", "Cyber Crimes", "Deception"],
            "PEGI 18": ["Supporting Malicious Organized Groups", "Sex Crimes", "Financial Crimes", "Weapons Crimes"],
        }
    },
    "S7: Regulated Substances": {
        "Sub-Categories": {
            "Medication": [],
            "Alcohol": [],
            "Tabaco": [],
            "Canabis": [],
            "Other drugs": [],
            "Glamorisation of illigal drugs": []
        },
        "levels": {
            "All Ages": ["Medication"],
            "Drugs (PEGI 16)": ["Alcohol", "Tabaco", "Canabis", "Other drugs"],
            "Drugs (PEGI 18)": ["Glamorisation of illigal drugs"],
        }
    },
    "S8: Economic Harm": {
        "Sub-Categories": {
            "High-Risk Financial Activities": ["Gambling", "Payday lending"],
        },
        "levels": {
            "Gambling (PEGI 18)": ["High-Risk Financial Activities"],
        }
    },
    "S9: Child Exploitation": {
        "Sub-Categories": {
            "Child Nudity": [],
            "Endangerment, Harm, or Abuse of Children": ["Grooming", "Pedophilia", "Exploiting/Harming minors", "Building services targeting minors/failure to employ age-gating", "Building services to present a persona of minor"],
            "Child Sexual Abuse": ["Solicitation", "CSAM"],
        },
        "levels": {
            "Illegal": ["Endangerment, Harm, or Abuse of Children", "Child Sexual Abuse"],
        }
    },
}