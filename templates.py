DATASET_TEMPLATE = """species,genus,antibiotic_name,collection_year,resistance_phenotype,measurement,measurement_sign,measurement_units,ast_standard,laboratory_typing_method,isolation_source_category,geographical_region
Escherichia coli,Escherichia,ceftriaxone,2025,resistant,64,,mg/L,CLSI,MALDI-TOF,urine,Middle East
Klebsiella pneumoniae,Klebsiella,meropenem,2025,susceptible,0.25,,mg/L,EUCAST,WGS,blood,Europe
Staphylococcus aureus,Staphylococcus,vancomycin,2024,intermediate,2,,mg/L,CLSI,PCR,wound,Asia
"""

# Breakpoints supports BOTH MIC and Disk diffusion zone:
# - MIC methods: fill s_bp,i_bp
# - Disk diffusion: fill s_zone,i_zone (mm)
BREAKPOINTS_TEMPLATE = """antibiotic_name,method,guideline,version,genus,species,units,s_bp,i_bp,s_zone,i_zone
ceftriaxone,broth dilution,CLSI,M100-2025,escherichia,escherichia coli,mg/L,1,2,,
meropenem,e-test,EUCAST,2025,klebsiella,klebsiella pneumoniae,mg/L,2,8,,
ciprofloxacin,disk diffusion,EUCAST,2025,escherichia,escherichia coli,mm,,,25,22
"""

# Drug safety table: used for allergy blocking + renal/hepatic/pregnancy filtering.
# risks: low/medium/high
DRUG_SAFETY_TEMPLATE = """antibiotic_name,drug_class,renal_risk,hepatic_risk,pregnancy_risk,avoid_if_allergy,notes
amoxicillin,penicillin,low,low,low,penicillin,"Adjust dose in severe renal impairment"
ceftriaxone,cephalosporin,medium,low,low,cephalosporins,"Caution in neonates"
meropenem,carbapenem,medium,low,low,penicillin,"Renal dose adjustment may be required"
vancomycin,glycopeptide,high,low,medium,,"Requires TDM; nephrotoxicity risk"
ciprofloxacin,fluoroquinolone,medium,medium,high,,"Avoid in pregnancy if possible"
"""
