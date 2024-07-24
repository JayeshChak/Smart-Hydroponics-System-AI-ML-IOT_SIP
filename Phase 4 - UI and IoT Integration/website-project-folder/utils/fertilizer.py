import pandas as pd

# Load crop data and fertilizer data
crop_data = pd.read_csv('Data/crop_data.csv')
fert_data = pd.read_csv('Data/fertilizer.csv')


def recommend_fertilizer(crop_name, crop_growth_stage, water_content):
    # Fetch crop details
    crop = crop_data[(crop_data['crop_name'] == crop_name) & 
                     (crop_data['crop_growth_stage'] == crop_growth_stage)]

    if crop.empty:
        return {"error": "Crop data not found"}

    # Check for nutrient deficiencies
    deficiencies = {}
    for nutrient in ['nitrogen', 'phosphorus', 'potassium', 'calcium', 'magnesium', 'sulfur', 'copper', 'chlorine', 'boron', 'iron', 'zinc', 'manganese', 'molybdenum', 'nickel', 'cobalt', 'sodium']:
        if water_content[nutrient] < crop[f'crop_{nutrient}'].values[0]:
            deficiencies[nutrient] = crop[f'crop_{nutrient}'].values[0] - water_content[nutrient]

    # Find matching fertilizers
    deficiency_fertilizers = []
    for nutrient, deficiency in deficiencies.items():
        fert = fert_data[fert_data[f'fert_{nutrient}'] > 0]
        if not fert.empty:
            deficiency_fertilizers.append({
                'fert_name': fert['fert_name'].values[0],
                'fert_type': fert['fert_type'].values[0],
                'fert_nitrogen': fert['fert_nitrogen'].values[0],
                'fert_phosphorus': fert['fert_phosphorus'].values[0],
                'fert_potassium': fert['fert_potassium'].values[0],
                'fert_calcium': fert['fert_calcium'].values[0],
                'fert_magnesium': fert['fert_magnesium'].values[0],
                'fert_sulfur': fert['fert_sulfur'].values[0],
                'fert_copper': fert['fert_copper'].values[0],
                'fert_chlorine': fert['fert_chlorine'].values[0],
                'fert_boron': fert['fert_boron'].values[0],
                'fert_iron': fert['fert_iron'].values[0],
                'fert_zinc': fert['fert_zinc'].values[0],
                'fert_manganese': fert['fert_manganese'].values[0],
                'fert_molybdenum': fert['fert_molybdenum'].values[0],
                'fert_nickel': fert['fert_nickel'].values[0],
                'fert_cobalt': fert['fert_cobalt'].values[0],
                'fert_sodium': fert['fert_sodium'].values[0],
            })

    # Check for pH stabilizers
    ph_stabilizer = fert_data[(fert_data['fert_ph'] == water_content['ph'])]
    ph_stabilizer = ph_stabilizer.iloc[0] if not ph_stabilizer.empty else None

    # Check for EC balancers
    ec_balancer = fert_data[(fert_data['fert_ec'] == water_content['ec'])]
    ec_balancer = ec_balancer.iloc[0] if not ec_balancer.empty else None

    # Check for growth regulators
    growth_regulator = fert_data[fert_data['fert_role'].str.contains('growth regulator', case=False)]
    growth_regulator = growth_regulator.iloc[0] if not growth_regulator.empty else None

    recommendations = {
        "deficiency_fertilizers": deficiency_fertilizers,
        "ph_stabilizer": ph_stabilizer,
        "ec_balancer": ec_balancer,
        "growth_regulator": growth_regulator,
    }

    return recommendations