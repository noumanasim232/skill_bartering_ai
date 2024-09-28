import tensorflow as tf
import numpy as np

class SkillRecommender:
    def __init__(self, num_users, num_skills, embedding_size=50):
        self.num_users = num_users
        self.num_skills = num_skills
        self.embedding_size = embedding_size

        # Initialize user and skill embeddings
        self.user_embeddings = tf.Variable(tf.random.normal([num_users, embedding_size]))
        self.skill_embeddings = tf.Variable(tf.random.normal([num_skills, embedding_size]))

    def train(self, user_skill_matrix, epochs=100, learning_rate=0.01):
        optimizer = tf.optimizers.Adam(learning_rate)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Predict ratings
                predictions = tf.matmul(self.user_embeddings, self.skill_embeddings, transpose_b=True)
                
                # Calculate loss (Mean Squared Error)
                loss = tf.reduce_mean(tf.square(user_skill_matrix - predictions))

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
        
        return top_indices.numpy()

# Example usage
if __name__ == "__main__":
    num_users = 100
    num_skills = 50
    
    # Create a dummy user-skill interaction matrix
    user_skill_matrix = np.random.randint(0, 2, size=(num_users, num_skills))
    
    recommender = SkillRecommender(num_users, num_skills)
    recommender.train(user_skill_matrix, epochs=50)
    
    # Get recommendations for user 0
    recommended_skills = recommender.recommend_skills(0)
    print("Recommended skill indices for user 0:", recommended_skills)