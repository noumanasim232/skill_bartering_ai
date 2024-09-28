import tensorflow as tf
import numpy as np
from ai.data.data_loader import DataLoader

class SkillRecommender:
    def __init__(self, embedding_size=50):
        self.data_loader = DataLoader()
        self.user_skill_matrix = self.data_loader.load_data()
        self.num_users, self.num_skills = self.user_skill_matrix.shape
        self.embedding_size = embedding_size

        # Initialize user and skill embeddings
        self.user_embeddings = tf.Variable(tf.random.normal([self.num_users, embedding_size]))
        self.skill_embeddings = tf.Variable(tf.random.normal([self.num_skills, embedding_size]))

    def train(self, epochs=100, learning_rate=0.01):
        optimizer = tf.optimizers.Adam(learning_rate)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Predict ratings
                predictions = tf.matmul(self.user_embeddings, self.skill_embeddings, transpose_b=True)
                
                # Calculate loss (Mean Squared Error)
                mask = tf.cast(self.user_skill_matrix > 0, tf.float32)
                loss = tf.reduce_sum(mask * tf.square(self.user_skill_matrix - predictions)) / tf.reduce_sum(mask)

            # Compute gradients and update embeddings
            gradients = tape.gradient(loss, [self.user_embeddings, self.skill_embeddings])
            optimizer.apply_gradients(zip(gradients, [self.user_embeddings, self.skill_embeddings]))

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")

    def recommend_skills(self, user_id, top_k=5):
        user_embedding = tf.gather(self.user_embeddings, user_id)
        skill_scores = tf.matmul([user_embedding], self.skill_embeddings, transpose_b=True)
        
        # Get top k skill indices
        _, top_indices = tf.nn.top_k(skill_scores[0], k=top_k)
        
        # Convert indices to skill names
        skill_names = self.data_loader.get_skill_names()
        recommended_skills = [skill_names[idx.numpy()] for idx in top_indices]
        
        return recommended_skills

# Example usage
if __name__ == "__main__":
    recommender = SkillRecommender()
    recommender.train(epochs=50)
    
    # Get recommendations for the first user
    user_names = recommender.data_loader.get_user_names()
    first_user = list(user_names.keys())[0]
    first_user_name = user_names[first_user]
    
    recommended_skills = recommender.recommend_skills(first_user)
    print(f"Recommended skills for {first_user_name}:")
    for skill in recommended_skills:
        print(f"- {skill}")