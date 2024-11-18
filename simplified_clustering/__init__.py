import spacy


def get_uploaded_corpus():
    return spacy.load("ru_core_news_sm", disable=["ner", "parser"])


def main():
    nlp = get_uploaded_corpus()


if __name__ == "__main__":
    main()
