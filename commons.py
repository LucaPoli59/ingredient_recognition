import matplotlib.pyplot as plt

def tokenize_category(category):
    return f"____{category}"

def detokenize_category(token):
    return token.rsplit("____", 1)[1] if "____" in token else token

def remove_token(token):
    return token.replace("____", "") if "____" in token else token


def show_image(img, cmap=None, title=None):
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    plt.show()