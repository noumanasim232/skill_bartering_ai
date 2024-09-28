from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello_world():
       return jsonify({"message": "Welcome to Skill Bartering AI!"})

@app.route('/api/skills', methods=['GET'])
def get_skills():
       # This is a placeholder. In a real app, you'd fetch this from a database.
       skills = [
           {"id": 1, "name": "Python Programming"},
           {"id": 2, "name": "Data Analysis"},
           {"id": 3, "name": "Web Design"}
       ]
       return jsonify(skills)

if __name__ == '__main__':
       app.run(debug=True)