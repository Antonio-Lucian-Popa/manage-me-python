import requests
import json

def findUserById(userId):
    # API endpoint URL and parameters
    url = "http://localhost:8080/api/v1/users/findById/" + userId

    # Make the API request
    response = requests.get(url)

    # Parse the JSON response
    data = json.loads(response.text)

    # Extract the temperature in Celsius from the response
    #is_user_not_expired = data["userNotExpired"]

    return data
