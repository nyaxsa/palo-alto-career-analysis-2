"""
Generates a synthetic HR extract (1470 employees) shaped like the classic
IBM HR Analytics Attrition dataset, rebranded as a Palo Alto Networks
extract for the Career Progression / Promotion Gap Analysis project.

Not real company data -- built to have realistic distributions and
correlations (tenure vs promotion gap, satisfaction vs attrition, etc.)
so the notebook's EDA/clustering/scoring logic runs meaningfully.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.RandomState(42)
N = 1470

departments = ["Research & Development", "Sales", "Human Resources"]
dept_p = [0.65, 0.30, 0.05]

job_roles_by_dept = {
    "Research & Development": [
        "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Research Director", "Manager",
    ],
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Human Resources": ["Human Resources", "Manager"],
}

education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"]

df = pd.DataFrame()
df["Age"] = rng.randint(18, 61, N)
df["Department"] = rng.choice(departments, N, p=dept_p)
df["JobRole"] = [rng.choice(job_roles_by_dept[d]) for d in df["Department"]]
df["EducationField"] = rng.choice(education_fields, N)
df["JobLevel"] = rng.choice([1, 2, 3, 4, 5], N, p=[0.35, 0.30, 0.17, 0.10, 0.08])

# TotalWorkingYears correlated with Age
df["TotalWorkingYears"] = np.clip(
    (df["Age"] - 20) * rng.uniform(0.3, 0.9, N) + rng.normal(0, 2, N), 0, None
).round().astype(int)

# YearsAtCompany bounded by TotalWorkingYears
df["YearsAtCompany"] = np.array([
    rng.randint(0, max(twy, 1) + 1) for twy in df["TotalWorkingYears"]
])

# YearsInCurrentRole and YearsWithCurrManager bounded by YearsAtCompany
df["YearsInCurrentRole"] = np.array([
    rng.randint(0, yac + 1) if yac > 0 else 0 for yac in df["YearsAtCompany"]
])
df["YearsWithCurrManager"] = np.array([
    rng.randint(0, yac + 1) if yac > 0 else 0 for yac in df["YearsAtCompany"]
])

# YearsSinceLastPromotion: higher JobLevel / longer role tenure -> more gap, with noise
promo_base = df["YearsInCurrentRole"] * rng.uniform(0.4, 1.0, N)
df["YearsSinceLastPromotion"] = np.clip(
    np.minimum(promo_base, df["YearsAtCompany"]).round(), 0, None
).astype(int)
df["YearsSinceLastPromotion"] = np.minimum(df["YearsSinceLastPromotion"], df["YearsAtCompany"])

df["TrainingTimesLastYear"] = rng.choice([0, 1, 2, 3, 4, 5, 6], N, p=[0.05, 0.10, 0.15, 0.35, 0.20, 0.10, 0.05])
df["PercentSalaryHike"] = rng.randint(11, 26, N)
df["PerformanceRating"] = rng.choice([3, 4], N, p=[0.85, 0.15])
df["JobSatisfaction"] = rng.choice([1, 2, 3, 4], N, p=[0.15, 0.20, 0.30, 0.35])
df["EnvironmentSatisfaction"] = rng.choice([1, 2, 3, 4], N, p=[0.15, 0.20, 0.30, 0.35])
df["WorkLifeBalance"] = rng.choice([1, 2, 3, 4], N, p=[0.10, 0.25, 0.45, 0.20])
df["MonthlyIncome"] = (df["JobLevel"] * rng.uniform(1800, 3200, N) + df["TotalWorkingYears"] * rng.uniform(40, 90, N)).round().astype(int)
df["DistanceFromHome"] = rng.randint(1, 30, N)
df["Gender"] = rng.choice(["Male", "Female"], N, p=[0.6, 0.4])
df["MaritalStatus"] = rng.choice(["Single", "Married", "Divorced"], N, p=[0.32, 0.46, 0.22])
df["OverTime"] = rng.choice(["Yes", "No"], N, p=[0.28, 0.72])
df["NumCompaniesWorked"] = rng.randint(0, 10, N)

# Attrition probability rises with promotion gap, low satisfaction, overtime, low WLB
gap_ratio = df["YearsSinceLastPromotion"] / df["YearsAtCompany"].clip(lower=1)
attr_logit = (
    -2.2
    + 1.8 * gap_ratio
    + 0.6 * (df["OverTime"] == "Yes")
    - 0.35 * (df["JobSatisfaction"] - 2.5)
    - 0.25 * (df["WorkLifeBalance"] - 2.5)
    + 0.15 * (df["JobLevel"] == 1)
)
attr_prob = 1 / (1 + np.exp(-attr_logit))
df["Attrition"] = (rng.uniform(0, 1, N) < attr_prob).astype(int)

df.insert(0, "EmployeeID", range(1001, 1001 + N))

out = Path(__file__).parent / "Palo_Alto_Networks.csv"
df.to_csv(out, index=False)
print(f"wrote {df.shape} -> {out}")
print(f"attrition rate: {df['Attrition'].mean()*100:.1f}%")
