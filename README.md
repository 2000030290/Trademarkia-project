# Trademarkia-project
AI Engineer Task -  Class Recommendation System

Creating an AI model to recommend trademark classes based on goods and services entered by the user. The code utilizes the USPTO ID manual dataset and exposes a REST API for integration with flask framework.

Now, the /recommend route will accept both GET and POST requests. GET requests will return recommendations, while POST requests will trigger the class recommendation based on the provided goods and services description.

Remember to save the code in a Python file and run it using python trademarkia-recommendation.py. Once the Flask server is running, you can try making a POST request to http://localhost:5000/recommend with the goods and services description in the request body as JSON to receive the recommended trademark classes.
