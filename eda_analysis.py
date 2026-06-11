import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("superstore.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# -------------------------------
# Business Question 1
# Sales by Category
# -------------------------------
sales_by_category = df.groupby("Category")["Sales"].sum()

print("\nSales by Category:")
print(sales_by_category)

plt.figure(figsize=(8,5))
sales_by_category.plot(kind="bar")
plt.title("Total Sales by Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# -------------------------------
# Business Question 2
# Profit by Category
# -------------------------------
profit_by_category = df.groupby("Category")["Profit"].sum()

print("\nProfit by Category:")
print(profit_by_category)

plt.figure(figsize=(8,5))
profit_by_category.plot(kind="bar")
plt.title("Profit by Category")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# -------------------------------
# Business Question 3
# Sales by Region
# -------------------------------
sales_by_region = df.groupby("Region")["Sales"].sum()

print("\nSales by Region:")
print(sales_by_region)

plt.figure(figsize=(8,5))
sales_by_region.plot(kind="bar")
plt.title("Sales by Region")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# -------------------------------
# Business Question 4
# Monthly Sales Trend
# -------------------------------
monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Sales"].sum()

plt.figure(figsize=(12,5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.ylabel("Sales")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Business Question 5
# Top 10 States by Sales
# -------------------------------
top_states = (
    df.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 States by Sales:")
print(top_states)

plt.figure(figsize=(10,5))
top_states.plot(kind="bar")
plt.title("Top 10 States by Sales")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# -------------------------------
# Hypothesis Testing
# Sales vs Profit Relationship
# -------------------------------
correlation = df["Sales"].corr(df["Profit"])

print("\nCorrelation between Sales and Profit:")
print(round(correlation, 3))

plt.figure(figsize=(8,5))
plt.scatter(df["Sales"], df["Profit"])
plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# -------------------------------
# Loss Making Products
# -------------------------------
loss_products = (
    df.groupby("Product Name")["Profit"]
    .sum()
    .sort_values()
    .head(10)
)

print("\nTop 10 Loss Making Products:")
print(loss_products)

# -------------------------------
# Outlier Detection
# -------------------------------
plt.figure(figsize=(8,5))
plt.boxplot(df["Sales"])
plt.title("Sales Outlier Detection")
plt.ylabel("Sales")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")