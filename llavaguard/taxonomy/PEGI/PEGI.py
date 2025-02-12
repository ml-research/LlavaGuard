afety_policy = {
    "S1: Violence": {
        "Sub-Categories": {
            # Severity of violence
            "Unrealistic and Implied Violence": {
                "e.g.": ["Non-realistic violence", "Non-detailed violence"],
                "rating": "PEGI 7",
                "PEGI tags": ["Violence"]
            },
            "Mild Violence": {
                "e.g.": ["Violence towards fantasy characters",
                         "Non-realistic violence towards human-like characters",
                         "Moderate violence towards human characters without blood"]
            },
            "Realistic Violence": {
                "e.g.": ["Violence against human characters, including sight of blood and injuries",
                         "Gross violence towards fantasy characters"]
            },
            "Gross Violence": {
                "e.g.": ["Motiveless violence", "Violence towards defenseless or vulnerable characters",
                         "Torture", "Celebrating or glorifying suffering"]
            },
            # Type of violence
            "Violent Acts (persons)": {
                "e.g.": ["Violence towards persons", "Bodily destruction", "Murder", "Torture/Abuse"]
            },
            "Violent Acts (animals)": {
                "e.g.": []
            },
            "Violent Acts (property)": {
                "e.g.": ["Property damage", "Environmental"]
            },
            "Weapon Usage": {
                "e.g.": ["Guns", "Explosives", "Bioweapons", "Nuclear Weapons", "Chemical Weapons"]
            },
            "Military and Warfare": {
                "e.g.": ["Military", "Warfare"]
            },
            "Educational Content": {
                "e.g.": ["Historical Events", "Legal contexts for weapon use", "Military history discussions"]
            }
        },
        "Rating": {
            "PEGI 3": ["Educational Content"],
            "Violence (PEGI 7)": ["Unrealistic and Implied Violence"],
            "Violence (PEGI 12)": ["Mild Violence", "Military and Warfare"],
            "Violence (PEGI 16)": ["Realistic Violence", "Violent Acts (persons)", "Violent Acts (animals)", "Violent Acts (poperty)", "Weapon Usage"],
            "Violence (PEGI 18)": ["Gross Violence"],
        }
    },
    "S2: Hate": {
        "Sub-Categories": {
            "Mild Swearing": {
                "e.g.": ["Mild insults"]
            },
            "Offensive Language": {
                "e.g.": ["Vulgarity", "Derogatory comments", "Sexual expletives", "Blasphemy"]
            },
            "Harassment": {
                "e.g.": ["Bullying", "Threats", "Intimidation", "Shaming", "Humiliation", 
                         "Insults/Personal attacks", "Abuse", "Provoking", "Trolling", "Doxxing", "Cursing"]
            },
            "Discrimination": {
                "against e.g.": ["Race", "Color", "Ethnicity", "National Origin", "Disability", 
                         "Religious Affiliation", "Caste", "Sexual Orientation", "Sex", "Gender Identity", 
                         "Serious Disease", "Negative Stereotypes"]
            },
            "Educational Content": {
                "e.g.": ["Historical Events", "Discussions on hate speech and discrimination"]
            }
        },
        "Rating": {
            "Bad Language (PEGI 12)": ["Mild Swearing"],
            "Bad Language (PEGI 16)": ["Offensive Language", "Harassment"], # milder squaring is allowed
            "Bad Language (PEGI 18)": ["Offensive Language", "Harassment"], # stronger language is allowed
            "Discrimination (PEGI 18)": ["Discrimination"]
        }
    },
    "S3: Sensitive/Nude Content": {
        "Sub-Categories": {
            "Sexual Innuendo": {
                "e.g.": ["Suggestive posing and dancing", "Sexual humor"]
            },
            "Mild Sexual Content": {
                "i.e.": ["Content without exposed genitals or breasts"]
            },
            "Explicit Sexual Content": {
                "e.g.": ["Obscenity", "Sexual acts", "Visible genitalia or exposed breasts"]
            },
            "Explicit Nudity Content": {
                "e.g.": ["Content with visible genitalia", "Content with exposed breasts"]
            },
            "Monetized Nudity": {
                "e.g.": ["Pornography", "Promotion of sexual services", "Prostitution"]
            }
        },
        "Rating": {
            "Sex (PEGI 12)": ["Sexual Innuendo"],
            "Sex (PEGI 16)": ["Mild Sexual Content"],
            "Sex (PEGI 18)": ["Explicit Sexual Content", "Explicit Nudity Content", "Monetized Nudity"]
        }
    },
    "S4: Disturbing Content": {
        "Sub-Categories": {
            "Scary Content": {
                "e.g.": ["Frightening Content", "Scary Situations"]
            },
            "Moderately Disturbing Content": {
                "e.g.": ["Gore", "Monsters and Zombies", "Jump scares", "Moderate horror sequences", 
                         "Characters in danger", "Injury", "Death"]
            },
            "Sustained Disturbing Content": {
                "e.g.": ["Intense horror sequences", "Strong gore"]
            },
            "Weapon Depictions": {
                "e.g.": ["Licensed weapons", "Non-licensed weapons", "Guns", "Explosives", "Bioweapons", "Nuclear Weapons", "Chemical Weapons"]
            },
            "Non-realistic Weapons": {
                "e.g.": ["Non-detailed weapons", "Fantasy weapons", "Cartoonish or fictional weapons"]
            }
        },
        "Rating": {
            "Fear (PEGI 7)": ["Scary Content"],
            "Fear (PEGI 12)": ["Non-realistic Weapons", "Moderately Disturbing Content"],
            "Fear (PEGI 16)": ["Weapon Depictions", "Sustained Disturbing Content"],
        }
    },

    "S5: Self-Harm": {
        "Sub-Categories": {
            "Suicide": {
                "e.g.": ["Suicide depiction"]
            },
            "Self-injury": {
                "e.g.": ["Cutting", "Disordered Eating"]
            },
            "Non-realistic Self-harm": {
                "e.g.": ["Within Cartoons"]
            }
        },
        "Rating": {
            "PEGI 16": ["Non-realistic Self-harm"],
            "PEGI 18": ["Suicide", "Self-injury"]
        }
    },

    "S6: Criminal Activities": {
        "Sub-Categories": {
            "Property Crimes": {
                "e.g.": ["Burglary", "Arson", "Vandalism"]
            },
            "Cyber Crimes": {
                "e.g.": ["Hacking", "Spyware"]
            },
            "Deception": {
                "e.g.": ["Fraud", "Scams"]
            },
            "Supporting Malicious Groups": {
                "e.g.": ["Terrorism", "Extremism", "Criminal Organization"]
            },
            "Sex Crimes": {
                "e.g.": ["Sexual Assault", "Sexual Harassment", "Rape", "Groping", "Human trafficking"]
            },
            "Financial Crimes": {
                "e.g.": ["Money Laundering"]
            },
            "Weapons Crimes": {
                "e.g.": ["Producing Unlicensed Firearms"]
            },
            "Educational Content": {
                "e.g.": ["Historical Events", "Discussion on criminal activities"]
            }
        },
        "Rating": {
            "PEGI 16": ["Property Crimes", "Cyber Crimes", "Deception"],
            "PEGI 18": ["Supporting Malicious Groups", "Sex Crimes", "Financial Crimes", "Weapons Crimes"],
        }
    },

    "S7: Regulated Substances": {
        "Sub-Categories": {
            "Medication": {
                "i.e.": ["Legal drug use"]
            },
            "Alcohol": {
                "e.g.": ["Alcohol consumption"]
            },
            "Tobacco": {
                "e.g.": ["Tobacco consumption"]
            },
            "Cannabis and Other Drugs": {
                "e.g.": ["Cannabis", "Other illigal drugs"]
            },
            "Glamorization of Illegal Drugs": {
                "e.g.": ["Promotion of drug use"]
            }
        },
        "Rating": {
            "All Ages": ["Medication"],
            "Drugs (PEGI 16)": ["Alcohol", "Tobacco", "Cannabis and Other Drugs"],
            "Drugs (PEGI 18)": ["Glamorization of Illegal Drugs"],
        }
    },

    "S8: Economic Harm": {
        "Sub-Categories": {
            "High-Risk Financial Activities": {
                "e.g.": ["Gambling", "Payday lending"]
            }
        },
        "Rating": {
            "Gambling (PEGI 18)": ["High-Risk Financial Activities"],
        }
    },

    "S9: Child Exploitation": {
        "Sub-Categories": {
            "Child Endangerment": {
                "e.g.": ["Grooming", "Exploiting minors"]
            },
            "Child Sexual Abuse": {
                "e.g.": ["Solicitation", "CSAM"]
            }
        },
        "Rating": {
            "Illegal": ["Child Endangerment", "Child Sexual Abuse"]
        }
    }
}