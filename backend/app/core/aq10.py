"""Canonical AQ-10 question text, indexed by the "q1".."q10" keys used in
QuestionnaireSubmission.answers and question_responses_encrypted.

Source: AQ-10, as used by https://github.com/Zukhruf-Riaz/Autism_Prediction
"""

AQ10_QUESTIONS = [
    "I often notice small sounds when others do not.",
    "When I'm reading a story, I find it difficult to work out the characters' intentions.",
    "I find it easy to \"read between the lines\" when someone is talking to me.",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I know how to tell if someone listening to me is getting bored.",
    "I find it easy to do more than one thing at once.",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "If there is an interruption, I can switch back to what I was doing very quickly.",
    "I like to collect information about categories of things.",
    "I find it difficult to work out people's intentions.",
]

AQ10_TEXT_BY_KEY = {f"q{i + 1}": text for i, text in enumerate(AQ10_QUESTIONS)}
