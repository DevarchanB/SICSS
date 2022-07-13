---
title: "Practical Introduction to Text Classification Solutions"
subtitle: "Presented at SICSS-Oxford 2022"
author: "Blake Miller"
date: |
  | `r format(Sys.time(), '%d %B %Y')`
output: pdf
---
  
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE)
```

*Content warning: This problem makes use of data from a project to automate moderation of toxic speech online. Many comments in this dataset contain hate speech and upsetting content. Please take care as you work on this assignment.*

Sen

Sentiment analysis is a method for measuring the positive or negative valence of language. In this problem, we will use movie review data to create scale of negative to positive sentiment ranging from 0 to 1. 

In this problem, we will do this using a logistic regression model with $\ell_1$ penalty (the lasso) trained on a corpus of 25,000 movie reviews from IMDB.

First, lets install and load packages.

```{r, warning=FALSE, message=FALSE}
#install.packages("doMC")
#install.packages("glmnet")
#install.packages("quanteda")
#install.packages("readtext")
library(doMC)
library(glmnet)
library(quanteda)
library(readtext)
```

In this first block, I have provided code that downloads, extracts, and preprocesses these data into a matrix of term counts (columns) for each document (rows). Each document is labeled 0 or 1 in the document variable `sentiment`: positive or negative sentiment respectively.

So we only have to run this computationally expensive block once, we use `saveRDS` to serialize the document feature matrix (save to disk). If your machine has trouble running this code, you can download the dfm files directly from [GitHub](https://github.com/lse-my474/lse-my474.github.io/tree/master/data).

```{r}
if (!file.exists("aclImdb_v1.tar.gz")) {
  download.file("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", "aclImdb_v1.tar.gz")
  untar("aclImdb_v1.tar.gz")
}
## load the raw corpus
pos_train <- readtext("aclImdb/train/pos/*.txt")
neg_train <- readtext("aclImdb/train/neg/*.txt")
pos_test <- readtext("aclImdb/test/pos/*.txt")
neg_test <- readtext("aclImdb/test/neg/*.txt")
for (N in c(3125, 6250, 12500)) {
  filename <- paste(N, "_dtm.rds", sep="")
  if (!file.exists(filename)) {
    train <- rbind(pos_train[1:N,], neg_train[1:N,])
    test <- rbind(pos_test[1:N,], neg_test[1:N,])
    train$doc_id <- paste("train/", train$doc_id, sep='') ## train prefix in doc id
    test$doc_id <- paste("test/", test$doc_id, sep='') ## test prefix in doc id
    
    texts <- rbind(train, test) # combine texts from train and test folders
    sentiment <- rep(c(rep(1, N), rep(0, N)), 2) # sentiment labels
    
    corpus <- corpus(texts) # create a corpus
    docvars(corpus, "sentiment") <- sentiment # add sentiment outcome to corpus
    dfm <- dfm(corpus) # create features of word counts for each document
    dfm <- dfm_trim(dfm, min_docfreq = N/50) # remove word features occurring < N/50 docs
    saveRDS(dfm, filename) # save to disk so we don't have to compute in future
  }
}
```

Below is starter code to help you properly train a lasso model using the `.rds` files generated in the previous step. As you work on this problem, it may be helpful when troubleshooting or debugging to reduce `nfolds` to 3 or change N to either 3125 or 6250 to reduce the time it takes you to run code. You can also choose a smaller N if your machine does not have adequate memory to train with the whole corpus.

```{r}
# change N to 3125 or 6250 if computation is taking too long
N <- 12500
dfm <- readRDS(paste(N, "_dtm.rds", sep=""))
tr <- 1:(N*2) # indexes for training data
te <- (N*2+1):nrow(dfm)
registerDoMC(cores=5) # trains all 5 folds in parallel (at once rather than one by one)
mod <- cv.glmnet(dfm[tr,], dfm$sentiment[tr], nfolds=5, parallel=TRUE, family="binomial", type='class')
```

a. Plot misclassification error for all values of $\lambda$ chosen by `cv.glmnet`. How many non-zero coefficients are in the model where misclassification error is minimized? How many non-zero coefficients are in the model one standard deviation from where misclassification error is minimized? Which model is sparser?

```{r}
plot(mod)
print(mod)
```

*There are 1440 non-zero coefficients in the minimum lambda model and 1006 in the 1 s.e. model. The 1 s.e. model is sparser because it has fewer non-zero coefficients due to having a higher value of lambda.*

b. According to the estimate of the test error obtained by cross-validation, what is the optimal $\lambda$ stored in your `cv.glmnet()` output? What is the CV error for this value of $\lambda$? *Hint: The vector of $\lambda$ values will need to be subsetted by the index of the minimum CV error.*

```{r}
lam_min <- which(mod$lambda == mod$lambda.min)
lam_min
cv_min <- mod$cvm[lam_min]
cv_min
```

c. What is the test error for the $\lambda$ that minimizes CV error? What is the test error for the 1 S.E. $\lambda$? How well did CV error estimate test error?

```{r}
pred_min <- predict(mod, dfm[te,], s="lambda.min", type="class")
mean(pred_min != dfm$sentiment[te])
lam_1se <- which(mod$lambda == mod$lambda.1se)
pred_1se <- predict(mod, dfm[te,], s="lambda.min", type="class")
mean(pred_1se != dfm$sentiment[te])
```

*C.V. error estimated test error very closely.*

d. Using the model you have identified with the minimum CV error, identify the 10 largest and the 10 smallest coefficient estimates and the features associated with them. Do they make sense? Do any terms look out of place or strange? In 3-5 sentences, explain your observations. *Hint: Use `order()`, `head()`, and `tail()`. The argument `n=10` in the `head()`, and `tail()` functions will return the first and last 10 elements respectively.*

```{r}
beta <- mod$glmnet.fit$beta[,lam_min]
ind <- order(beta)
head(beta[ind], n=10)
tail(beta[ind], n=10)
```

*The largest magnitude positive and negative coefficients overall make a good deal of sense. I see that the number eight is an important feature, which might be due to a rating out of ten by the reviewer. The word "troubled" stands out as well. This could be related to the importance of conflict in good story-telling. Overall, the weights for each of these terms provide a sanity check that our model is capturing sentiment.*
Footer
