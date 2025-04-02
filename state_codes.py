# state_codes = {
#     "AN": "Andaman and Nicobar Islands",
#     "AP": "Andhra Pradesh",
#     "AR": "Arunachal Pradesh",
#     "AS": "Assam",
#     "BR": "Bihar",
#     "CG": "Chhattisgarh",
#     "CH": "Chandigarh",
#     "DD": "Daman and Diu",
#     "DL": "Delhi",
#     "GA": "Goa",
#     "GJ": "Gujarat",
#     "HP": "Himachal Pradesh",
#     "HR": "Haryana",
#     "JH": "Jharkhand",
#     "JK": "Jammu and Kashmir",
#     "KA": "Karnataka",
#     "KL": "Kerala",
#     "LD": "Lakshadweep",
#     "MH": "Maharashtra",
#     "ML": "Meghalaya",
#     "MN": "Manipur",
#     "MP": "Madhya Pradesh",
#     "MZ": "Mizoram",
#     "NL": "Nagaland",
#     "OD": "Odisha",
#     "PB": "Punjab",
#     "PY": "Puducherry",
#     "RJ": "Rajasthan",
#     "SK": "Sikkim",
#     "TN": "Tamil Nadu",
#     "TR": "Tripura",
#     "TS": "Telangana",
#     "UK": "Uttarakhand",
#     "UP": "Uttar Pradesh",
#     "WB": "West Bengal"
# }

# import pandas as pd

# # Load RTO mapping
# rto_data = pd.read_csv("Indian_RTO.csv", delimiter=",", dtype=str)
# rto_data.columns = rto_data.columns.str.replace('"', '').str.strip()

# rto_mapping = {row["RTO"]: (row["Location"], row["state"]) for _, row in rto_data.iterrows()}

# def get_rto_details(plate_text):
#     """Extract state and district from a number plate."""
#     rto_code = plate_text[:4]  # Extract first 4 characters
#     return rto_mapping.get(rto_code, ("Unknown District", "Unknown State"))

import pandas as pd

# Load RTO mapping
rto_data = pd.read_csv("Indian_RTO.csv", delimiter=",", dtype=str)
rto_data.columns = rto_data.columns.str.replace('"', '').str.strip()

rto_mapping = {row["RTO"]: (row["Location"], row["state"]) for _, row in rto_data.iterrows()}

def get_rto_details(plate_text):
    """Extract state and district from a number plate."""
    plate_text = plate_text.upper().strip()  # Normalize text
    if len(plate_text) < 4:
        return "Unknown District", "Unknown State"
    
    rto_code = plate_text[:4]  # Extract first 4 characters
    return rto_mapping.get(rto_code, ("Unknown District", "Unknown State"))
