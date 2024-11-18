import spacy
from spacy.lang.ru.examples import sentences


def main():
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(sentences[0])
    print(doc.text)
    for token in doc:
        print(token.text, token.pos_, token.dep_)


if __name__ == "__main__":
    main()
