from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        """Initialize with the dataset."""
        self.data = data.copy()  # Work on a copy to avoid modifying the original data
        self.scaler = MinMaxScaler()
        # self.encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid dummy variable trap
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')  

    def handle_duplicates(self):
        """Remove duplicate rows."""
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        print(f"Removed {before - after} duplicate rows.")

    def handle_missing_values(self):
        """Fill missing values using forward fill method."""
        missing_before = self.data.isnull().sum().sum()
        # self.data = self.data.fillna(method='ffill')  # Forward fill
        self.data = self.data.ffill()  # Correct way
        missing_after = self.data.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values.")

    def check_outliers(self):
        """Detect outliers using the IQR method."""
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers[col] = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()

        print("Outlier count per column:", outliers)

    def check_data_types(self):
        """Check and display data types."""
        print("\nData Types:\n", self.data.dtypes)

    def Normalized_data(self):
        """Normalize numerical features and encode categorical features."""
        # Identify numerical and categorical columns
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        # Identify date column
        date_col = self.data.select_dtypes(include=['datetime']).columns

        # Scale numerical features
        scaled_numeric = pd.DataFrame(self.scaler.fit_transform(self.data[numeric_cols]), 
                                    columns=numeric_cols, index=self.data.index)

        # Encode categorical features if they exist
        if len(categorical_cols) > 0:
            encoded_categorical = pd.DataFrame(self.encoder.fit_transform(self.data[categorical_cols]),
                                            columns=self.encoder.get_feature_names_out(categorical_cols),
                                            index=self.data.index)
            # Combine scaled numerical and encoded categorical features
            processed_data = pd.concat([scaled_numeric, encoded_categorical], axis=1)
        else:
            processed_data = scaled_numeric  # If no categorical features, return only scaled numeric data
        
        # If date column exists, ensure it's retained in the final dataframe
        if len(date_col) > 0:
            processed_data[date_col[0]] = self.data[date_col[0]]
        
        # Ensure that the final DataFrame retains the original index
        processed_data.index = self.data.index

        return processed_data
