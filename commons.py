def tokenize_category(category):
    return f"{category}____"

def detokenize_category(token):
    return token.split("____", 1)[0] if "____" in token else token

def remove_token(token):
    return token.split("____", 1)[1] if "____" in token else token
