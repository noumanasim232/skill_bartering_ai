import tensorflow as tf
import numpy as np

class SkillMatcher:
    def __init__(self):
        self.skill_embeddings = {}

    def add_skill(self, skill_name, embedding):
        """Add a skill and its embedding to the matcher."""
        self.skill_embeddings[skill_name] = tf.constant(embedding, dtype=tf.float32)

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
    
    # Add some example skills (in a real scenario, these would be learned embeddings)
    matcher.add_skill("Python Programming", [1.0, 0.8, 0.6])
    matcher.add_skill("Data Analysis", [0.8, 1.0, 0.7])
    matcher.add_skill("Machine Learning", [0.9, 0.9, 1.0])
    
    # Find matches for a query
    query = [0.9, 0.8, 0.9]  # This could represent a user's skill profile
    matches = matcher.find_matches(query)
    
    print("Top matches:")
    for skill, score in matches:
        print(f"{skill}: {score:.4f}")