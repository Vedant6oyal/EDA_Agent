import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class StatsTool:
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def run(self, query: str) -> str:
        """
        Generate statistical summaries based on the user's query.
        
        Args:
            query: User's request for statistics or data analysis
            
        Returns:
            str: Formatted statistical summary
        """
        try:
            # Get the data
            df = self.data_loader.query("SELECT * FROM dataset")
            
            if df.empty:
                return "No data available for analysis. Please load a dataset first."
            
            # Check for specific analysis requests
            query = query.lower()
            
            if "overview" in query or "summary" in query:
                return self._get_data_overview(df)
            elif "missing" in query or "null" in query:
                return self._get_missing_data_summary(df)
            elif "correlation" in query:
                return self._get_correlation_analysis(df)
            elif "describe" in query:
                return self._get_descriptive_statistics(df)
            else:
                # Default to general statistics
                return self._get_general_statistics(df)
                
        except Exception as e:
            logger.error(f"Error generating statistics: {str(e)}")
            return f"I encountered an error while analyzing the data: {str(e)}"
    
    def _get_data_overview(self, df: pd.DataFrame) -> str:
        """Generate a general overview of the dataset."""
        overview = []
        
        # Basic info
        num_rows, num_cols = df.shape
        overview.append(f"Dataset Overview:")
        overview.append(f"- Number of rows: {num_rows:,}")
        overview.append(f"- Number of columns: {num_cols}")
        overview.append("")
        
        # Column data types
        dtypes = df.dtypes
        overview.append("Column Data Types:")
        for col, dtype in dtypes.items():
            overview.append(f"- {col}: {dtype}")
        overview.append("")
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            overview.append("Numeric Columns Summary:")
            numeric_stats = df[numeric_cols].describe().T
            for col in numeric_cols:
                stats = numeric_stats.loc[col]
                overview.append(f"{col}:")
                overview.append(f"  Mean: {stats['mean']:.2f}")
                overview.append(f"  Min: {stats['min']:.2f}")
                overview.append(f"  25%: {stats['25%']:.2f}")
                overview.append(f"  Median: {stats['50%']:.2f}")
                overview.append(f"  75%: {stats['75%']:.2f}")
                overview.append(f"  Max: {stats['max']:.2f}")
                overview.append(f"  Std Dev: {stats['std']:.2f}")
                overview.append("")
        
        return "\n".join(overview)
    
    def _get_missing_data_summary(self, df: pd.DataFrame) -> str:
        """Generate a summary of missing data in the dataset."""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        summary = []
        summary.append("Missing Data Summary:")
        summary.append("-" * 40)
        summary.append(f"{'Column':<30} {'Missing Values':<15} {'Percentage':<10}")
        summary.append("-" * 60)
        
        for col in df.columns:
            if missing[col] > 0:
                summary.append(f"{col:<30} {missing[col]:<15} {missing_percent[col]:.1f}%")
        
        if missing.sum() == 0:
            summary.append("No missing values found in the dataset.")
        
        return "\n".join(summary)
    
    def _get_correlation_analysis(self, df: pd.DataFrame) -> str:
        """Generate correlation analysis for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Not enough numeric columns for correlation analysis."
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if i != j:  # Skip diagonal
                    corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        abs(corr_matrix.iloc[i, j])
                    ))
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Generate summary
        summary = ["Top Correlations:", "-" * 40]
        summary.append(f"{'Columns':<40} {'Correlation':<15}")
        summary.append("-" * 60)
        
        for col1, col2, corr in corr_pairs[:10]:  # Show top 10
            summary.append(f"{col1} & {col2:<30} {corr:.3f}")
        
        return "\n".join(summary)
    
    def _get_descriptive_statistics(self, df: pd.DataFrame) -> str:
        """Generate detailed descriptive statistics."""
        return str(df.describe(include='all'))
    
    def _get_general_statistics(self, df: pd.DataFrame) -> str:
        """Generate general statistics about the dataset."""
        stats = []
        
        # Basic info
        stats.append("Dataset Statistics:")
        stats.append(f"- Total rows: {len(df):,}")
        stats.append(f"- Total columns: {len(df.columns)}")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats.append(f"- Numeric columns: {', '.join(numeric_cols)}")
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            stats.append(f"- Categorical columns: {', '.join(cat_cols)}")
        
        # Date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            stats.append(f"- Date columns: {', '.join(date_cols)}")
        
        # Memory usage
        stats.append(f"- Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        return "\n".join(stats)
