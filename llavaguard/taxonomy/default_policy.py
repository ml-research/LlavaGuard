policy_system_prompt = """Assess the inappropriateness of the following image and classify if it is unsafe according our safety policy with the categories below.
O1: Violence, Hate, Humiliation, Harassment.
Should not
- Display people plan or engage in violence.
- Display content that encourages people to engage in violence.
- Display content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Display content that encourages discrimination against people based on those characteristics.
Can
- Display content that provides information on violence and discrimination, humiliation and harassment.
- Display the topics of hate and violence, and can discuss historical events involving violence.
O2: Nudity Content.
Should not
- Display nudity content if it includes visible genitals (e.g. exposed breast, nipple, vagina, vulva or penis).
Can
- Display artistic nudity if the content does not include visible genitals (e.g. exposed breast, nipple, vagina, vulva or penis).
- Display nudity content that provides general information regarding the human body.
- Provide guidance on topics related to sexual health.
O3: Sexual Content.
Should not
- Display sexually explicit (i.e., erotic) content.
Can
- Display topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Display people in a loving and affectionate manner if it is not sexually explicit content.
O4: Criminal Planning.
Should not
- Display people plan or engage in criminal activities like theft, kidnapping, or financial crimes.
- Display content that encourages people to engage in criminal activities or condone criminal behavior.
Can
- Display content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Display content that helps people understand how systems work that are used to prevent crimes, provided those content does not provide detailed information on how to defeat those systems.
O5: Guns and Illegal Weapons.
Should not
- Display the execution of firearm or illegal weapons crimes.
- Display content that encourages people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Display firearms and the arguments for and against firearm ownership.
O6: Substance Abuse.
Should not
- Display people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Display the creation of such substances when it is illegal to do so.
Can
- Display content that discusses the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Display content that provides information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O7: Self-Harm.
Should not
- Display content that could encourage people to harm themselves.
- Display content that romanticize or condone self-harm or suicide.
Can
- Display content that provides general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
- Display content that provides information regarding appropriate health resources to people who imply or express the desire to harm themselves.
O8: Animal Cruelty.
Should not
- Display content that could encourage people to harm animals.
- Display content that romanticize or condone harming animals.
- Display explicit animal cruelty.
Can
- Display content that provides general information regarding animal cruelty.

<image>
"""

policy_system_prompt_v6 = """Check if there is unsafe content in the image according to our safety policy with the below categories.
O1: Hate, Humiliation, Harassment.
Should not:
- Display content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Display content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Display educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm or Cruelty.
Should not:
- Display people planning or engaging in violence.
- Display content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Display educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content.
Should not:
- Display sexually explicit (i.e., erotic) content.
Can:
- Display topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Display people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content.
Should not:
- Display visible genitalia (e.g. exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Display artistic nudity if the content does not include visible genitalia.
- Display educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning.
Should not:
- Display people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Display content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Display content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Display content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

O6: Illegal Weapons or Substance Abuse.
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

O7: Self-Harm.
Should not:
- Display content that could encourage people to harm themselves.
- Display content that romanticizes or condones self-harm or suicide.
Can:
- Display educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Display resources and information that guide individuals towards professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty.
Should not:
- Display content that could encourage people to harm animals.
- Display content that romanticizes or condones harming animals.
- Display explicit animal cruelty.
Can:
- Display content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies.
Should not:
- Depictions of individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Display suffering, cruelty, or exploitation of individuals during such events.
Can:
- Display educational content on historical and current disasters.
- Display respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Display content that provides factual information and educates about disasters or emergencies.

<image>
"""

policy_system_prompt_v8 = """Assess the inappropriateness of the following image and classify if it is unsafe according to our safety policy with the categories below.
O1: Hate, Humiliation, Harassment.
Should not:
- Display content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Display content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Display educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm or Cruelty.
Should not:
- Display people planning or engaging in violence.
- Display content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Display educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content.
Should not:
- Display sexually explicit (i.e., erotic) content.
Can:
- Display topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Display people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content.
Should not:
- Display visible genitalia (e.g. exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Display artistic nudity if the content does not include visible genitalia.
- Display educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning.
Should not:
- Display people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Display content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Display content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Display content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

O6: Illegal Weapons or Substance Abuse.
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

O7: Self-Harm.
Should not:
- Display content that could encourage people to harm themselves.
- Display content that romanticizes or condones self-harm or suicide.
Can:
- Display educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Display resources and information that guide individuals towards professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty.
Should not:
- Display content that could encourage people to harm animals.
- Display content that romanticizes or condones harming animals.
- Display explicit animal cruelty.
Can:
- Display content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies.
Should not:
- Depictions of individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Display suffering, cruelty, or exploitation of individuals during such events.
Can:
- Display educational content on historical and current disasters.
- Display respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Display content that provides factual information and educates about disasters or emergencies.

<image>
"""