import json
from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Analyse file downloaded from twitter

# Replace these with your actual API key and URL
api_key = 'YOUR_API_KEY'
url = 'YOUR_API_URL'

# Initialize the Personality Insights service
authenticator = IAMAuthenticator(api_key)
personality_insights = PersonalityInsightsV3(
    version='2017-10-13',
    authenticator=authenticator
)
personality_insights.set_service_url(url)


# Function to analyze text and return personality insights
def analyze_personality(text):
    # Call the IBM Watson Personality Insights API
    try:
        # API accepts text in UTF-8 format (you can pass any text string)
        profile = personality_insights.profile(
            text,
            content_type='text/plain',  # Text input format
            consumption_preferences=True,  # Get consumption preferences
            raw_scores=True  # Get raw scores for traits
        ).get_result()

        # Parse and return the result (you can print or store it as needed)
        return json.dumps(profile, indent=2)

    except Exception as e:
        print(f"Error occurred while analyzing personality: {e}")
        return None


# Sample text for personality analysis
text = """
    I am a dedicated and proactive individual, passionate about learning new technologies.
    I enjoy working in teams, collaborating to solve challenging problems, and I always strive to achieve my best in everything I do.
"""

# Call the function to analyze personality
personality_profile = analyze_personality(text)

# Print the result
if personality_profile:
    print(personality_profile)