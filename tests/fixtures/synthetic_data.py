"""
Synthetic data generation for testing DataQA framework.

This module provides utilities to generate realistic synthetic datasets
for various testing scenarios including performance testing, edge cases,
and integration testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    name: str
    size: int
    columns: Dict[str, Any]
    relationships: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


class SyntheticDataGenerator:
    """Generate synthetic datasets for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        self.seed = seed or 42
        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def generate_sales_data(self, size: int = 1000) -> pd.DataFrame:
        """Generate realistic sales transaction data."""
        np.random.seed(self.seed)
        
        # Product categories and names
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        products_by_category = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
            'Home & Garden': ['Chair', 'Table', 'Lamp', 'Plant', 'Vase'],
            'Sports': ['Basketball', 'Tennis Racket', 'Running Shoes', 'Yoga Mat', 'Bicycle'],
            'Books': ['Fiction Novel', 'Cookbook', 'Biography', 'Technical Manual', 'Children Book']
        }
        
        # Regions and sales reps
        regions = ['North', 'South', 'East', 'West', 'Central']
        sales_reps = [f'Rep_{i:03d}' for i in range(1, 21)]
        
        data = []
        for i in range(size):
            category = np.random.choice(categories)
            product = np.random.choice(products_by_category[category])
            region = np.random.choice(regions)
            sales_rep = np.random.choice(sales_reps)
            
            # Generate correlated data
            base_price = {
                'Electronics': 500, 'Clothing': 50, 'Home & Garden': 100,
                'Sports': 75, 'Books': 20
            }[category]
            
            quantity = np.random.poisson(3) + 1
            unit_price = base_price * (0.8 + 0.4 * np.random.random())
            revenue = quantity * unit_price
            cost = revenue * (0.6 + 0.2 * np.random.random())
            
            # Date with seasonal patterns
            start_date = datetime(2020, 1, 1)
            days_offset = np.random.randint(0, 1460)  # 4 years
            date = start_date + timedelta(days=days_offset)
            
            # Add seasonal boost for certain categories
            if category == 'Electronics' and date.month in [11, 12]:
                revenue *= 1.3
            elif category == 'Sports' and date.month in [4, 5, 6, 7, 8]:
                revenue *= 1.2
            
            data.append({
                'transaction_id': f'TXN_{i+1:06d}',
                'product': product,
                'category': category,
                'quantity': quantity,
                'unit_price': round(unit_price, 2),
                'revenue': round(revenue, 2),
                'cost': round(cost, 2),
                'profit': round(revenue - cost, 2),
                'date': date,
                'region': region,
                'sales_rep': sales_rep,
                'customer_segment': np.random.choice(['Enterprise', 'SMB', 'Consumer'], p=[0.2, 0.3, 0.5])
            })
        
        return pd.DataFrame(data)
    
    def generate_customer_data(self, size: int = 500) -> pd.DataFrame:
        """Generate realistic customer data."""
        np.random.seed(self.seed + 1)
        
        # Name components
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Emily', 
                      'James', 'Jessica', 'William', 'Ashley', 'Richard', 'Amanda', 'Thomas']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
                     'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson']
        
        regions = ['North', 'South', 'East', 'West', 'Central']
        segments = ['Enterprise', 'SMB', 'Consumer']
        
        data = []
        for i in range(size):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            
            # Generate correlated demographic data
            age = np.random.normal(40, 15)
            age = max(18, min(80, int(age)))
            
            # Income correlated with age and segment
            segment = np.random.choice(segments, p=[0.15, 0.35, 0.5])
            base_income = {'Enterprise': 80000, 'SMB': 50000, 'Consumer': 35000}[segment]
            income = base_income * (0.7 + 0.6 * np.random.random()) * (age / 40)
            
            # Signup date with growth pattern
            start_date = datetime(2018, 1, 1)
            # More recent signups (growth pattern)
            days_offset = int(np.random.exponential(365)) % 2190  # Max ~6 years
            signup_date = start_date + timedelta(days=days_offset)
            
            data.append({
                'customer_id': f'CUST_{i+1:05d}',
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f'{first_name} {last_name}',
                'email': f'{first_name.lower()}.{last_name.lower()}@example.com',
                'age': age,
                'income': round(income, 0),
                'region': np.random.choice(regions),
                'segment': segment,
                'signup_date': signup_date,
                'is_active': np.random.choice([True, False], p=[0.85, 0.15]),
                'lifetime_value': round(income * 0.1 * np.random.random(), 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_product_data(self, size: int = 100) -> pd.DataFrame:
        """Generate product catalog data."""
        np.random.seed(self.seed + 2)
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Generic']
        
        data = []
        for i in range(size):
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            
            # Generate product attributes
            base_price = {
                'Electronics': 300, 'Clothing': 40, 'Home & Garden': 80,
                'Sports': 60, 'Books': 15
            }[category]
            
            price = base_price * (0.5 + np.random.random())
            cost = price * (0.4 + 0.3 * np.random.random())
            
            # Rating correlated with brand and price
            brand_quality = {'BrandA': 4.5, 'BrandB': 4.2, 'BrandC': 3.8, 
                           'BrandD': 3.5, 'BrandE': 3.2, 'Generic': 2.8}
            base_rating = brand_quality[brand]
            rating = min(5.0, max(1.0, base_rating + np.random.normal(0, 0.3)))
            
            data.append({
                'product_id': f'PROD_{i+1:04d}',
                'product_name': f'{brand} {category} Item {i+1}',
                'category': category,
                'brand': brand,
                'price': round(price, 2),
                'cost': round(cost, 2),
                'margin': round(((price - cost) / price) * 100, 1),
                'rating': round(rating, 1),
                'review_count': np.random.poisson(50) + 5,
                'in_stock': np.random.choice([True, False], p=[0.9, 0.1]),
                'stock_quantity': np.random.poisson(100) if np.random.random() > 0.1 else 0,
                'launch_date': datetime(2018, 1, 1) + timedelta(days=np.random.randint(0, 2190))
            })
        
        return pd.DataFrame(data)
    
    def generate_time_series_data(self, 
                                 start_date: datetime, 
                                 end_date: datetime, 
                                 frequency: str = 'D',
                                 metrics: List[str] = None) -> pd.DataFrame:
        """Generate time series data with trends and seasonality."""
        if metrics is None:
            metrics = ['revenue', 'orders', 'customers']
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        data = {'date': date_range}
        
        for metric in metrics:
            # Generate base trend
            trend = np.linspace(100, 200, len(date_range))
            
            # Add seasonality (yearly and weekly patterns)
            yearly_season = 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)
            weekly_season = 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / 7)
            
            # Add noise
            noise = np.random.normal(0, 5, len(date_range))
            
            # Combine components
            values = trend + yearly_season + weekly_season + noise
            values = np.maximum(values, 0)  # Ensure non-negative
            
            # Scale based on metric type
            if metric == 'revenue':
                values *= 1000
            elif metric == 'orders':
                values = np.round(values).astype(int)
            elif metric == 'customers':
                values = np.round(values * 0.3).astype(int)
            
            data[metric] = values
        
        return pd.DataFrame(data)
    
    def generate_hierarchical_data(self, levels: List[str], size: int = 1000) -> pd.DataFrame:
        """Generate hierarchical data (e.g., organization, department, team)."""
        np.random.seed(self.seed + 3)
        
        # Generate hierarchy structure
        hierarchy = {}
        for i, level in enumerate(levels):
            if i == 0:
                # Top level
                hierarchy[level] = [f'{level}_{j:02d}' for j in range(1, 6)]
            else:
                # Child levels
                parent_level = levels[i-1]
                hierarchy[level] = []
                for parent in hierarchy[parent_level]:
                    children = [f'{parent}_{level}_{k:02d}' for k in range(1, 4)]
                    hierarchy[level].extend(children)
        
        # Generate data with hierarchical relationships
        data = []
        for i in range(size):
            record = {'id': i + 1}
            
            # Select random path through hierarchy
            for j, level in enumerate(levels):
                if j == 0:
                    record[level] = np.random.choice(hierarchy[level])
                else:
                    parent_level = levels[j-1]
                    parent_value = record[parent_level]
                    # Find children of this parent
                    children = [item for item in hierarchy[level] if item.startswith(parent_value)]
                    record[level] = np.random.choice(children) if children else f'{parent_value}_default'
            
            # Add metrics that roll up through hierarchy
            record['value'] = np.random.exponential(100)
            record['count'] = np.random.poisson(10) + 1
            
            data.append(record)
        
        return pd.DataFrame(data)


class TestDataFixtures:
    """Pre-generated test data fixtures for consistent testing."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize fixtures with optional data directory."""
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.generator = SyntheticDataGenerator()
    
    def get_small_sales_dataset(self) -> pd.DataFrame:
        """Get small sales dataset for unit tests."""
        return self.generator.generate_sales_data(size=100)
    
    def get_medium_sales_dataset(self) -> pd.DataFrame:
        """Get medium sales dataset for integration tests."""
        return self.generator.generate_sales_data(size=1000)
    
    def get_large_sales_dataset(self) -> pd.DataFrame:
        """Get large sales dataset for performance tests."""
        return self.generator.generate_sales_data(size=10000)
    
    def get_customer_dataset(self, size: str = "medium") -> pd.DataFrame:
        """Get customer dataset of specified size."""
        sizes = {"small": 50, "medium": 500, "large": 2000}
        return self.generator.generate_customer_data(size=sizes[size])
    
    def get_product_dataset(self, size: str = "medium") -> pd.DataFrame:
        """Get product dataset of specified size."""
        sizes = {"small": 20, "medium": 100, "large": 500}
        return self.generator.generate_product_data(size=sizes[size])
    
    def get_time_series_dataset(self, 
                               days: int = 365, 
                               frequency: str = 'D') -> pd.DataFrame:
        """Get time series dataset."""
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=days)
        return self.generator.generate_time_series_data(start_date, end_date, frequency)
    
    def get_hierarchical_dataset(self, size: str = "medium") -> pd.DataFrame:
        """Get hierarchical organizational data."""
        sizes = {"small": 100, "medium": 1000, "large": 5000}
        levels = ['region', 'country', 'state', 'city']
        return self.generator.generate_hierarchical_data(levels, size=sizes[size])
    
    def get_multi_table_dataset(self) -> Dict[str, pd.DataFrame]:
        """Get related multi-table dataset for join testing."""
        # Generate related datasets
        customers = self.get_customer_dataset("medium")
        products = self.get_product_dataset("medium")
        sales = self.generator.generate_sales_data(size=2000)
        
        # Ensure referential integrity
        customer_ids = customers['customer_id'].tolist()
        product_names = products['product_name'].tolist()
        
        # Update sales to reference existing customers and products
        sales['customer_id'] = np.random.choice(customer_ids, size=len(sales))
        sales['product'] = np.random.choice(product_names, size=len(sales))
        
        return {
            'customers': customers,
            'products': products,
            'sales': sales
        }
    
    def save_datasets_to_files(self, formats: List[str] = None) -> Dict[str, Path]:
        """Save all datasets to files for external testing."""
        if formats is None:
            formats = ['csv', 'parquet']
        
        datasets = {
            'small_sales': self.get_small_sales_dataset(),
            'medium_sales': self.get_medium_sales_dataset(),
            'customers': self.get_customer_dataset(),
            'products': self.get_product_dataset(),
            'time_series': self.get_time_series_dataset(),
            'hierarchical': self.get_hierarchical_dataset()
        }
        
        saved_files = {}
        
        for name, df in datasets.items():
            for fmt in formats:
                if fmt == 'csv':
                    file_path = self.data_dir / f"{name}.csv"
                    df.to_csv(file_path, index=False)
                elif fmt == 'parquet':
                    file_path = self.data_dir / f"{name}.parquet"
                    df.to_parquet(file_path, index=False)
                elif fmt == 'json':
                    file_path = self.data_dir / f"{name}.json"
                    df.to_json(file_path, orient='records', date_format='iso')
                
                saved_files[f"{name}.{fmt}"] = file_path
        
        return saved_files
    
    def generate_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce data quality issues for testing error handling."""
        df_with_issues = df.copy()
        
        # Introduce missing values
        missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        missing_columns = np.random.choice(df.columns, size=min(3, len(df.columns)), replace=False)
        
        for idx in missing_indices:
            for col in missing_columns:
                if np.random.random() < 0.3:
                    df_with_issues.loc[idx, col] = None
        
        # Introduce outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if np.random.random() < 0.5:
                outlier_indices = np.random.choice(df.index, size=max(1, int(len(df) * 0.01)), replace=False)
                for idx in outlier_indices:
                    df_with_issues.loc[idx, col] = df[col].mean() + 10 * df[col].std()
        
        # Introduce duplicate rows
        if len(df) > 10:
            duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
            duplicates = df_with_issues.loc[duplicate_indices].copy()
            df_with_issues = pd.concat([df_with_issues, duplicates], ignore_index=True)
        
        return df_with_issues


# Global fixtures instance
test_fixtures = TestDataFixtures()


def get_test_datasets() -> Dict[str, pd.DataFrame]:
    """Get all standard test datasets."""
    return {
        'small_sales': test_fixtures.get_small_sales_dataset(),
        'medium_sales': test_fixtures.get_medium_sales_dataset(),
        'customers': test_fixtures.get_customer_dataset(),
        'products': test_fixtures.get_product_dataset(),
        'time_series': test_fixtures.get_time_series_dataset(),
        'hierarchical': test_fixtures.get_hierarchical_dataset()
    }


def get_edge_case_datasets() -> Dict[str, pd.DataFrame]:
    """Get datasets with edge cases for robust testing."""
    generator = SyntheticDataGenerator()
    
    return {
        'empty_dataset': pd.DataFrame(),
        'single_row': generator.generate_sales_data(size=1),
        'single_column': pd.DataFrame({'value': range(100)}),
        'all_nulls': pd.DataFrame({'col1': [None] * 10, 'col2': [None] * 10}),
        'mixed_types': pd.DataFrame({
            'int_col': range(10),
            'float_col': [i + 0.5 for i in range(10)],
            'str_col': [f'item_{i}' for i in range(10)],
            'bool_col': [i % 2 == 0 for i in range(10)],
            'date_col': pd.date_range('2024-01-01', periods=10)
        }),
        'with_quality_issues': test_fixtures.generate_data_quality_issues(
            generator.generate_sales_data(size=200)
        )
    }