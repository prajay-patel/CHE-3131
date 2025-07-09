import json
import base64
from IPython.display import display, HTML
from jupyterquiz import display_quiz

def collapsible_display_quiz(
    encoded_json_path='quiz_encoded.json',
    dict_key=None
):
    """
    Loads base64-encoded quiz data from a JSON file and displays it using JupyterQuiz.
    Supports decoding either a list of questions or a dictionary containing the list.
    The code is wrapped in a collapsible HTML block to optionally hide it in notebooks.

    Parameters:
        encoded_json_path (str): Path to the base64-encoded quiz JSON file.
        dict_key (str or None): If the decoded data is a dict, this key is used to extract the list of quiz questions.
        open_by_default (bool): Whether the collapsible block is open by default.
    """

    # Decode the quiz
    with open(encoded_json_path, 'r') as f:
        data = json.load(f)

    quiz_b64 = data["data"]
    quiz_json = base64.b64decode(quiz_b64).decode("utf-8")
    decoded = json.loads(quiz_json)

    # Extract quiz data (list of questions) depending on structure
    if dict_key:
        if dict_key not in decoded:
            raise KeyError(f"Key '{dict_key}' not found in decoded JSON.")
        quiz_data = decoded[dict_key]
    else:
        quiz_data = decoded

    # Display the quiz
    display_quiz(quiz_data)

    # End collapsible block
    display(HTML("</details>"))