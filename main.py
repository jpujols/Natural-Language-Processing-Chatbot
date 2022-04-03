import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('averaged_perception_tagger')
nltk.download('wordnet')
nltk.download('punkt')

text = 'Originally, vegetables were collected from the wild by hunters-gatheres. Vegetables are all plants. Vegetables can be eaten either raw or cooked.'
question = 'What are plants?'

lemmatizer = WordNetLemmatizer()

def lemma_me(sent):
  sentence_tokens = nltkl.word_tokenize(sent.lower())
  pos_tags = nltk.pos_tag(sentence_tokens)

  sentence_lemmas = []
  for token, pos_tag in zip(sentence_tokens, pos_tags):
    if pos_tags[1][0].lower() in ['n', 'v', 'a', 'r']:
      lemma = lemmatizer.lemmatize(token, pos_tag[1[0]].lower())
      return sentence_lemmas

sentence_tokens = nltk.sent_tokenize(text)
sentence_tokens.append(question)

tv = TfidVectorizer(tokenizer=lemma_me)
tf = tv.fit_transform(sentence_tokens)
values = cosine_similarity(tf[-1], tf)
index = values.argsort()[0][-2]
values_flat = values.flatten()
values_flat.sort()
coeff = values_flat[-2]
if coeff > 0.3:
  print(sentence_tokens[index])

