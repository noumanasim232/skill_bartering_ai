import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path='data/user_skills.csv'):
        self.data_path = data_path
        self.data = None
        self.user_mapping = {}
        self.skill_mapping = {}

    def generate_sample_data(self, num_users=100, num_skills=50, sparsity=0.1):
        """Generate sample user-skill data and save to CSV."""
        users = [f"user_{i}" for i in range(num_users)]
        skills = [f"skill_{i}" for i in range(num_skills)]
        
        data = []
        for user in users:
            for skill in skills:
                if np.random.random() < sparsity:
                    data.append({
                        'user_id': user,
                        'skill_id': skill,
                        'proficiency': np.random.randint(1, 6)  # 1-5 proficiency level
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        print(f"Sample data generated and saved to {self.data_path}")

    def load_data(self):
        # Load data from CSV
        self.data = pd.read_csv(self.data_path)
        
        # Create user and skill mappings
        self.user_mapping = {user: idx for idx, user in enumerate(self.data['user_id'].unique())}
        self.skill_mapping = {skill: idx for idx, skill in enumerate(self.data['skill_id'].unique())}
        
        # Convert to matrix format
        user_skill_matrix = np.zeros((len(self.user_mapping), len(self.skill_mapping)))
        for _, row in self.data.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            skill_idx = self.skill_mapping[row['skill_id']]
            user_skill_matrix[user_idx, skill_idx] = row['proficiency']
        
        return user_skill_matrix

    def get_train_test_split(self, test_size=0.2):
        user_skill_matrix = self.load_data()
        return train_test_split(user_skill_matrix, test_size=test_size, random_state=42)

    def get_skill_names(self):
        return {v: k for k, v in self.skill_mapping.items()}

    def get_user_names(self):
        return {v: k for k, v in self.user_mapping.items()}

# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Generate sample data if the CSV doesn't exist
    import os
    if not os.path.exists(loader.data_path):
        loader.generate_sample_data()
    
    train_data, test_data = loader.get_train_test_split()
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    print(f"Number of skills: {len(loader.get_skill_names())}")
    print(f"Number of users: {len(loader.get_user_names())}")