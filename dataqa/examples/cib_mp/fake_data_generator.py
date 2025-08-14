import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()

# Define companies with their IDs and categories
companies = [
    {"name": "Starbucks", "co_id": 456, "extl_id": 1001, "category": "Food & Beverage"},
    {"name": "Home Depot", "co_id": 789, "extl_id": 1002, "category": "Home Improvement"},
    {"name": "Costco", "co_id": 123, "extl_id": 1003, "category": "Wholesale"},
    {"name": "Barnes & Noble", "co_id": 321, "extl_id": 1004, "category": "Retail"},
    {"name": "ExxonMobil", "co_id": 654, "extl_id": 1005, "category": "Petroleum"}
]

# Define possible values for various fields
cust_types = ['BU', 'TD', 'CO', 'CU', 'CH']
states = ['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI', 'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT', 'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'WV', 'ID', 'HI', 'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
countries = ['US', 'CA']
cust_stats = ['I', 'A', 'R', 'N', 'S'] # A is the most common
mktseg_codes = ['SMBUS', 'NATNL', 'MOMKT', 'CAN-SMBUS', 'Associations']

# Define MCC descriptions based on company category
mcc_descriptions = {
    "Food & Beverage": "Eating Places, Restaurants",
    "Home Improvement": "Home Supply Warehouse Stores",
    "Wholesale": "Wholesale Clubs",
    "Retail": "Book Stores",
    "Petroleum": "Petroleum & Petroleum Products"
}

# Generate subsidiaries for each company
subsidiaries = []
num_subsidiaries_per_company = 10 # Number of subsidiaries per company

for company in companies:
    for _ in range(num_subsidiaries_per_company):
        cust_extl_id = fake.unique.random_int(min=100, max=999) # Unique identifier for each subsidiary
        mcc_cd = fake.unique.random_int(min=5000, max=5999) # Unique MCC code for each subsidiary

        subsidiary = {
            "CUST_KEY": fake.unique.random_int(min=100000, max=999999),
            "CUST_ID": fake.unique.random_int(min=100000, max=999999),
            "BANK_ENTERPRISE_CUST_ID": fake.unique.random_int(min=1000000, max=9999999),
            "CUST_NAME": f"{company['name']} - {fake.city()}",
            "CUST_TYPE_CD": random.choice(cust_types),
            "CUST_STATE_CD": random.choice(states),
            "CUST_COUNTRY_CD": random.choice(seq=countries), 
            "CUST_EXTL_ID": cust_extl_id,
            "CO_ORG_ID": company['co_id'],
            "CUST_STAT_CD": random.choice(cust_stats),
            "MCC_DESC": mcc_descriptions[company["category"]],
            "MKTSEG_CD": random.choice(mktseg_codes),
            "OWNRSHP_COMP_LVL_1_EXTL_ID": company["extl_id"],
            "OWNRSHP_COMP_LVL_1_NAME": company["name"]
        }
        subsidiaries.append(subsidiary)

# Create a DataFrame for subsidiaries
df_subsidiaries = pd.DataFrame(subsidiaries)
df_subsidiaries.to_csv("FAKE_ETS_D_CUST_PORTFOLIO.csv", index=False)


# --- FAKE DATA FOR PROD_BD_TH_FLAT_V3 ---
num_transactions = 10000
start_date = datetime(2024, 4, 17)
end_date = datetime(2025, 4, 17)

# Generate random dates within the specified range
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

mop_cd_ptendpoint_pairs = {
    ('CR', 'ChaseNet'),
    ('DB', 'Discover Settled'),
    ('DX', 'Discover Settled'),
    ('EB', 'Debit Tampa FE'),
    ('DD', 'Discover Settled'),
    ('VP', 'Visa Canada'),
    ('CZ', 'ChaseNet'),
    ('VI', 'Visa'),
    ('MC', 'MasterCard'),
    ('VR', 'Visa Canada'),
    ('VT', 'Visa'),
    ('CH', 'ChaseNet'),
    ('DI', 'Discover Conv'),
    ('AX', 'Amex US'),
    ('AI', 'Amex Intl'),
    ('MR', 'MasterCard')
 }

countries_currencies = {
    ('US', 'USD', 'USA', 'USD'),
    ('CA', 'CAD', 'CAN', 'CAD'),
    ('GB', 'GBP', 'GBR', 'GBP'),
    ('DE', 'EUR', 'DEU', 'EUR'),
    ('AU', 'AUD', 'AUS', 'AUD'),
    ('JP', 'JPY', 'JPN', 'JPY')
}

# Generate data for PROD_BD_TH_FLAT_V3
transactions = []

for _ in range(num_transactions):
    subsidiary = random.choice(subsidiaries)
    mop_cd, ptendpoint = random.choice(list(mop_cd_ptendpoint_pairs))
    acct_country, settled_currency, country, currency = random.choice(list(countries_currencies))
    gross_sales_units = np.random.choice([1, 0], p=[0.9, 0.1]) # 0=false, 1=true
    gross_sales_usd = 0.0 if gross_sales_units == 0 else float(random.randint(100, 12000))

    transaction = {
        "TRAN_DETAIL_ID": fake.unique.random_int(min=10000000, max=99999999),
        "SUBM_DT_YYYYMM": random_date(start_date, end_date).strftime("%Y%m"),
        "SUBM_DT": random_date(start_date, end_date).strftime("%Y-%m-%d"),
        "CO_ORG_ID": subsidiary["OWNRSHP_COMP_LVL_1_EXTL_ID"],
        "MBR_ENT": subsidiary["CUST_EXTL_ID"],
        "GROSS_SALES_USD": gross_sales_usd,
        "GROSS_SALES_UNITS": gross_sales_units,
        "MOP_CD": mop_cd,
        "TXN_TYPE": random.choice(['R', '8', '7', '6', '5', '1']),
        "ACCT_COUNTRY_CD": acct_country,
        "SETTLED_CURRENCY": settled_currency,
        "SUBM_PROCESS_DT": random.choice([100]),
        "PTENDPOINT": ptendpoint,
        "COUNTRY": country,
        "CURRENCY_CD": currency
    }
    transactions.append(transaction)

# Create a DataFrame for transactions
df_transactions = pd.DataFrame(transactions)
df_transactions.to_csv("FAKE_PROD_BD_TH_FLAT_V3.csv", index=False)