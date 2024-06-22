
label2index = {
    "label_ABDOMINAL": 0, 
    "label_ADVANCED-CAD": 1, 
    "label_ALCOHOL-ABUSE": 2, 
    "label_ASP-FOR-MI": 3, 
    "label_CREATININE": 4, 
    "label_DIETSUPP-2MOS": 5, 
    "label_DRUG-ABUSE": 6, 
    "label_ENGLISH": 7, 
    "label_HBA1C": 8, 
    "label_KETO-1YR": 9, 
    "label_MAJOR-DIABETES": 10, 
    "label_MAKES-DECISIONS": 11, 
    "label_MI-6MOS": 12
}

index2prompt = {
    0: "Does note above meet the criteria: History of intra abdominal surgery, small or large intestine resection or small bowel obstruction?",
    1: "Does note above meet the criteria: Advanced cardiovascular disease? We define 'advanced' as having two or more of the following: 1. Taking two or more medications to treat CAD; 2. History of myocardial infarction; 3. Currently experiencing angina; 4. Ischemia, past or present.",
    2: "Does note above meet the criteria: Current alcohol use over weekly recommended limits?",
    3: "Does note above meet the criteria: Use of aspirin to prevent myocardial infarction?",
    4: "Does note above meet the criteria: Serum creatinine > upper limit of normal?",
    5: "Does note above meet the criteria: Taken a dietary supplement (excluding Vitamin D) in the past 2 months?",
    6: "Does note above meet the criteria: Drug abuse, current or past?",
    7: "Does note above meet the criteria: Patient speak English?",
    8: "Does note above meet the criteria: Any HbA1c value between 6.5 and 9.5%?",
    9: "Does note above meet the criteria: Diagnosis of ketoacidosis in the past year?",
    10: "Does note above meet the criteria: Major diabetes-related complication? We define 'major complication' (as opposed to 'minor complication') as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: Amputation, Kidney damage, Skin conditions, Retinopathy, nephropathy, neuropathy.",
    11: "Does note above meet the criteria: Patient must make their own medical decisions?",
    12: "Does note above meet the criteria: Myocardial infarction in the past 6 months?"
}
