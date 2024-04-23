import matplotlib.pyplot as plt

def tokenize_category(category):
    return f"{category}____"

def detokenize_category(token):
    return token.split("____", 1)[0] if "____" in token else token

def remove_token(token):
    return token.replace("____", "") if "____" in token else token


def show_image(img, cmap=None, title=None):
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    plt.show()