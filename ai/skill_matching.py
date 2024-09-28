import tensorflow as tf
import numpy as np
from ai.data.data_loader import DataLoader

class SkillMatcher:
    def __init__(self):
        self.skill_embeddings = {}
        self.data_loader = DataLoader()
        self.skill_names = self.data_loader.get_skill_names()

    def train(self):
        """Train the skill matcher using the user-skill matrix."""
        user_skill_matrix = self.data_loader.load_data()
        
        # Use SVD to get skill embeddings
        _, _, V = np.linalg.svd(user_skill_matrix, full_matrices=False)
        embedding_size = min(50, V.shape[0])  # Use top 50 components or less
        
        for skill_idx, skill_name in self.skill_names.items():
            self.skill_embeddings[skill_name] = V[skill_idx, :embedding_size]

    def find_matches(self, query_embedding, top_k=5):
        """Find the top k matching skills for a given query embedding."""
        query_embedding = tf.constant(query_embedding, dtype=tf.float32)
        
        similarities = {}
        for skill, embedding in self.skill_embeddings.items():
            similarity = tf.tensordot(query_embedding, embedding, axes=1)
            similarities[skill] = similarity.numpy()
        
        # Sort similarities in descending order
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_matches[:top_k]

# Example usage
if __name__ == "__main__":
    matcher = SkillMatcher()
    matcher.train()
    
    # Use the first skill's embedding as a query for demonstration
    query_skill = list(matcher.skill_embeddings.keys())[0]
    query_embedding = matcher.skill_embeddings[query_skill]
    
    matches = matcher.find_matches(query_embedding)
    
    print(f"Top matches for {query_skill}:")
    for skill, score in matches:
        print(f"{skill}: {score:.4f}")