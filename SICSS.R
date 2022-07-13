library(quanteda)
library(quanteda.textplots)

texts <- read.csv('attacks.csv', stringsAsFactors=F)
texts$attack <- factor(texts$attack)
corpus <- corpus(texts, text_field="text") # create a corpus
corpus
summary(corpus)

?tokens
toks <- tokens(corpus, remove_punct = TRUE, remove_url=TRUE, verbose=TRUE)
toks

dfm <- dfm(toks, verbose=TRUE)
dfm

toks_stem <- tokens_wordstem(toks)
dfm_stem <- dfm(toks_stem, tolower=TRUE)
dfm_stem
example <- tolower(texts$text[5])
tokens(example)
tokens_wordstem(tokens(example))
