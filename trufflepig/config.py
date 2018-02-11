import os

NODE_URL = os.environ.get('NODE_URL', 'https://api.steemit.com')

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))