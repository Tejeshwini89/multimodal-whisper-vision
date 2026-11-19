from multimodal_whisper.pipeline import detect_intent


def test_question_with_question_mark():
    assert detect_intent("What color is the car?") == "question"


def test_question_without_question_mark():
    assert detect_intent("How many people are in the picture") == "question"


def test_describe_request():
    assert detect_intent("Please describe the image") == "describe"


def test_describe_what_do_you_see():
    assert detect_intent("What do you see in this photo?") == "describe"


def test_action_request():
    assert detect_intent("Count the number of dogs") == "question"


def test_statement():
    assert detect_intent("Nice weather today") == "statement"


def test_empty_text():
    assert detect_intent("") == "statement"


def test_substring_does_not_trigger_describe():
    assert detect_intent("The cat is sitting") == "statement"
