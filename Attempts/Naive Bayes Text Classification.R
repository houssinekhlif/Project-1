library(quanteda)
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(gmodels)
library(caret)

#import our data
amazon_reviews <- read.table("C:/Users/21650/Downloads/sentiment+labelled+sentences/sentiment labelled sentences/amazon_cells_labelled.txt", header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE)
yelp_reviews<-read.table("C:/Users/21650/Downloads/sentiment+labelled+sentences/sentiment labelled sentences/yelp_labelled.txt", header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE) 
imdb_review<-read.table("C:/Users/21650/Downloads/sentiment+labelled+sentences/sentiment labelled sentences/imdb_labelled.txt", header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE) 
all_reviews<-rbind(amazon_reviews,yelp_reviews,imdb_review)

#label each column and factor label
colnames(all_reviews) <- c("text", "label")
all_reviews$label <- factor(all_reviews$label, levels = c(0, 1), labels = c("negative", "positive"))
str(all_reviews$label)
table(all_reviews$label)

#transform table into corpus 
review_corpus <- VCorpus(VectorSource(all_reviews$text))
print(review_corpus)
inspect(review_corpus[1:4])
as.character(review_corpus[[3]])


#Preprocessing
review_corpus_clean <- tm_map(review_corpus, FUN = content_transformer(tolower))
as.character(review_corpus_clean[[3]])

review_corpus_clean <- tm_map(review_corpus_clean, removeNumbers)

review_corpus_clean <- tm_map(review_corpus_clean, removeWords, stopwords())

review_corpus_clean <- tm_map(review_corpus_clean, content_transformer(function(x) gsub("[[:punct:]]+", " ", x)))
as.character((review_corpus_clean[[4]]))
as.character(review_corpus[[4]])

review_corpus_clean <- tm_map(review_corpus_clean, stemDocument)

review_corpus_clean <- tm_map(review_corpus_clean, stripWhitespace)


#transform corpus into DTM
review_dtm<-DocumentTermMatrix(review_corpus_clean)

#Train Test splitting
set.seed(123)
train_index <- createDataPartition(all_reviews$label, p = 0.75, list = FALSE)

review_dtm_train <- review_dtm[train_index, ]
review_dtm_test <- review_dtm[-train_index, ]

review_train_labels <- all_reviews$label[train_index]
review_test_labels <- all_reviews$label[-train_index]


# Check balance
prop.table(table(review_train_labels))
prop.table(table(review_test_labels))

#wordcloud visual
wordcloud(review_corpus_clean, max.words = 50, random.order = FALSE)
positive <- subset(all_reviews, label == "positive")
negative<- subset(all_reviews, label == "negative")
wordcloud(positive$text, max.words = 30)

#find frequent words
review_freq_words <- findFreqTerms(review_dtm_train, 5)
str(review_freq_words)
#train and test only frequent words
review_dtm_freq_train <- review_dtm_train[ , review_freq_words]
review_dtm_freq_test <- review_dtm_test[ , review_freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

review_train <- apply(review_dtm_freq_train, MARGIN = 2, convert_counts)
review_test <- apply(review_dtm_freq_test, MARGIN = 2, convert_counts)


#Train model
review_classifier <- naiveBayes(review_train, review_train_labels)

#predict using our model
review_test_pred <- predict(review_classifier, review_test, laplace=T)

#asses model
library(gmodels)
CrossTable(review_test_pred, review_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn= c('predicted', 'actual'))

library(caret)
confusionMatrix(review_test_pred, review_test_labels)


predict_new_text <- function(input_text, dtm_terms, classifier) {
  library(tm)
  library(SnowballC)
  
  # Create a single-document corpus
  input_corpus <- VCorpus(VectorSource(input_text))
  
  # Preprocess (same as training)
  input_corpus <- tm_map(input_corpus, content_transformer(tolower))
  input_corpus <- tm_map(input_corpus, removeNumbers)
  input_corpus <- tm_map(input_corpus, removeWords, stopwords())
  input_corpus <- tm_map(input_corpus, content_transformer(function(x) gsub("[[:punct:]]+", " ", x)))
  input_corpus <- tm_map(input_corpus, stemDocument)
  input_corpus <- tm_map(input_corpus, stripWhitespace)
  
  # Create DTM with the same terms (dictionary)
  input_dtm <- DocumentTermMatrix(input_corpus, control = list(dictionary = dtm_terms))
  
  # Convert to matrix and then factor
  input_matrix <- as.matrix(input_dtm)
  
  # If no terms are matched, create a zero vector for all terms
  if(nrow(input_matrix) == 0) {
    input_matrix <- matrix(0, nrow = 1, ncol = length(dtm_terms))
    colnames(input_matrix) <- dtm_terms
  }
  
  # Convert counts to factors ("Yes"/"No")
  convert_counts <- function(x) {
    ifelse(x > 0, "Yes", "No")
  }
  
  input_df <- as.data.frame(t(apply(input_matrix, 1, convert_counts)))
  
  # Predict — note: this returns a factor of length 1
  prediction <- predict(classifier, input_df)
  
  return(prediction)
}


# 'review_freq_words' is the vector of terms used for training
new_text <- "your product is bad"
predict_new_text(new_text, review_freq_words, review_classifier)

