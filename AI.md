Understanding the public opinion on Artificial Intelligence
================
Fabio Taddei Dalla Torre
15/7/2020

# I. Introduction

Artificial intelligence (AI) refers to systems that display intelligent
behavior by analyzing their environment and taking actions – with some
degree of autonomy – to achieve specific goals. This is the definition
of artificial intelligence given by the European Commission in 2018 [1].

Many are the possible advantages of this technology one recent study,
for example, related to the COVID-19 pandemic, has proved that models
built using machine learning shown high sensitivity and high specificity
in detecting COVID-19 by analyzing chest CT exams [2]. Moreover must be
said that Artificial Intelligence is already well present in everybody
daily life. Every smartphone has some type of assistant for example
“Siri” or “Google Assistant” that displays intelligence like behavior.

Although this theme leads inevitably to some problem. “Ex Machina”,
“Trascendence or”Io Robot“, are only three movie titles where movie
industry has shown a World where human life is threatened by self
conscious machine engineered by human themselves. Those are irrelevant
but in 2014 Stephen Hawking told in an interview by the BBC:”The
development of full artificial intelligence could spell the end of the
human race.", a clear warning by one of the most intelligent people of
this century.

This work is composed by five main sections. The second one deals with
the formulation of the research question and will provide a brief
literature review on previous studies. Section three will describe the
methodology used in order to answer to the research question, how data
are collected, cleaned and the techniques used. Section four will
concern the analysis of the data, it contains the codes and comments
about the process and the results. The final section gives a summary of
the results, also presenting the main advantages and disadvantages of
the techniques and processes used.

# II. Previous studies and research question

From the invention of the World Wide Web until today, this concept has
changed and evolved. Among the many implementations that this technology
has allowed, the development of social media platforms capable of fast
and free communication on a global scale has definitely changed our
lives. It is estimated that, in 2020, the global social penetration
reach has reached 49%, with a peak considering Asia and North America
that respectively scored a rate of 71% and 69%. Northern Europe fallow
with a slightly lower 67% [3]. From those statistics is easy to
understand the gold mine that social media represent from an opinion
mining point of view.

The aim of this work is to:

-   Gain a general understanding about the opinion of people, what they
    think about artificial intelligence. Therefore it aims to understand
    which are the feeling and the sentiment of people, are they excited
    or do they fear this technology?
-   Understand whether different conversation topics about this subject
    are present. Maybe there would be a group of people that promote AI
    and so they will talk about certain subject while others will be
    more skeptical writing about something else.
-   Investigate what are the most common words in the negative sentiment
    tweets and in the positive in order to understand if there is a
    potential correlation between these terms and the emotions.

This analysis is carried out by using Tweets, firstly because it
guarantees open data despite other platforms and even because of the
large amount of available content. In 2012 it was calculated that
registered users produced a total of 340 million tweets per day [4].
People do not use Twitter only for sharing personal content but also for
expressing their opinion. Because of this behavior by analyzing this
tweets, first with more qualitative technique and then with a sentiment
analysis is possible to understand the perception of users to a certain
topic [5].

From business and marketing purpose to politics, opinion mining is
becoming more and more important in order to understand people thoughts,
hence for building better strategies [6]. A pivotal factor in opinion
mining is sentiment analysis, in this case the technique tries to
understand the sentiment of the writer by judging the document. The
object of the analysis is to assign for each document (in this case each
tweet) a predefined sentiment (positive, negative…), this classification
is based on machine learning and lexicon based approaches [7]. One point
that needs attention when working with Twitter data is the difference
between standard texts used in building sentiment analysis algorithms
and tweets. First of all because of the length of those, 140 character
maximum, and second because of their nature tweets tend to have
misspelling and slang with a relative high rate. [8]

Many studies have been conducted using sentiment analysis and topic
modelling on tweets. Indeed, in [9] S.Yang and H.Zhang have used
sentiment analysis and LDA topic modelling in order to understand the
potential of these techniques on twitter data. S. Yang et al have proven
that LDA allows to easily analyze huge number of tweets and to identify
hidden topics. The sentiment analysis was carried on using the
**syuzhet** package and the results have proved that this is a powerful
tool for understanding and showing the sentiments and emotions present
on tweets. A. Alamsyah [10] et al, have shown that using LDA and
sentiment analysis leads to the capability of extracting meaningful
insights about a specific topic using large scale datasets. Compared
with traditional method they have stated that those techniques allow
better in-time processing. Understand people reaction and sentiment
about the COVID epidemic was done by another research. Also in this case
the analysis was carried out by performing a sentiment analysis, with
the syuzhet package, and an LDA topic modelling. [11]

# III. Methodology

In order to accomplish my task I use the Twitter standard API with the
package **twitteR**. I request 7000 tweets by the hashtag *\#AI*.

Cleaning the tweets is the first step of the analysis. This process is
done by working on a database created with the requested tweets. At the
end of this procedure I get two different text corpus and a cleaned
data-set with three column: one containing the text, one the creation
date and one with the unique tweet ID. I have decided to create a
data-set for further analysis and for consistency of the results, indeed
all the code can be runned by taking the information from the
*df.ai.csv*.

Furthermore, only English tweets are gathered with the exclusion of
retweet. This because, from one side the sentiment classifier works only
with English vocabulary and on the other side because high frequency
retweets could distort the percentages of word occurrence.

As a consequent step I proceed by plotting a word cloud in order to give
a visual representation of the words that occur the most among all the
tweets. To strengthen this I decided to plot a bar-plot with words that
have been registered more than 150 times. This because the bar-plot
allows to have a closer idea of the importance of a word.

The previous analysis gives a general idea of the most important words
but it gives no insight on how these terms are used together. For this
exploration we can use a Word Network, in this case words are linked
together according to their presence with respect to one another.

The next steps regard Topic Modelling. The aim of this technique is to
present the text as a set of topics and so to find different topics
among the tweets. Once topics are found the tweets are labeled according
to that, for further analysis. This procedure is done by using LDA
technique. LDA topic modelling is based on a probabilistic model
approach and in particular it is based on the assumption that documents
are a mixture of topic and that each topics is composed by a certain
probability distribution of words.

The last step regard a sentiment analysis of tweets. This is done by
using the package **syuzhet** because it allows to label tweets not only
as positive or negative but also with a range of other emotions (anger,
anticipation, disgust, fear, joy, negative, positive, sadness, surprise,
trust). The algorithm works by the NRC emotion-Lexicon that is a list of
English words associated with emotions and sentiments. At first the
result is displayed cumulatively in order to have a general overview of
people’s sentiment and then, after normalizing the results, a grouped
bar chart is be produced that allows the comparison of the sentiment
according to the topic. Moreover, understanding what makes a tweet
negative or positive is done by computing a binomial sentiment analysis
and by displaying the most occurrenced words for negative and positive
sentiment.

# VI. Analysis

``` r
#Library
library(twitteR)
library(wordcloud)
library(topicmodels)
library(dplyr)
library(stringr)
library(tm)
library(ggplot2)
library(tidytext)
library(tidyverse)
library(igraph)
library(ggraph)
library(widyr)
library(quanteda)
library(syuzhet)
library(data.table)
```

### Twitter access and authentication

``` r
# key and token
consumer_key <- XXXXXX
consumer_secret <- XXXXXX
access_token <- XXXXXX
access_secret <- XXXXXX
```

Getting twitter access

``` r
# connecting to Twitter
setup_twitter_oauth(consumer_key,
                    consumer_secret,
                    access_token,
                    access_secret)
```

Getting tweets

``` r
# requiring the tweets
tweets <- searchTwitter("#AI -filter:retweets", n=7000, lang = "en") # requiring tweets
length(tweets)
```

### Data claning

``` r
# data preparation
df.ai <- bind_rows(lapply(tweets, as.data.frame))
df.ai$text <- gsub("((?:\\b\\W*@\\w+)+)", "", df.ai$text) # Remove usernames
df.ai$text <- gsub("http.+ |http.+$", "", df.ai$text)     # removing links
df.ai <- df.ai[!duplicated(df.ai[, 'id']),]               # removing duplicates
df.ai$text <- gsub("[^[:graph:]]", " ", df.ai$text)       # removing graphical
df.ai$text <- gsub("[[:punct:]]", " ", df.ai$text)        # remoove punctuation
df.ai$text <- gsub("^ ", "", df.ai$text)                  # remove spaces at the beginning
df.ai$text <- gsub(" $", "", df.ai$text)                  # remove spaces at the end
df.ai$text <- tolower(df.ai$text)                         # transform the whole text to lower
df.ai$text <- removeWords(df.ai$text, stopwords('en'))    # removing stopwords
df.ai$text <- gsub("[ |\t]{2,}", " ", df.ai$text)         # removing tabs
df.ai$text <- gsub(" +", " ", df.ai$text)                 # removing spaces
df.ai$text <- iconv(df.ai$text, to = "ASCII", sub = " ")  # converting to ascii
df.ai$text <- gsub('[0-9]+', "", df.ai$text)              # removing numbers
df.ai$text <- removeWords(df.ai$text, c("9", "10", "18","19", "5", "0", "46", "1", "2", "’s", "-", "amp")) # removing certain char
```

At the end of this cleaning process the prepared dataset is saved

``` r
#write.csv(df.ai,file=paste("df.ai.csv"))
```

For consistency and reproducibility of the results the following
manipulation will be done by loading and using the saved dataset

``` r
# reading and furhter preparation
df.ai <- read.csv("df.ai.csv", header = T) # reading 
df.ai <- df.ai[,c(2,6,9)] # keeping only useful column

#df.ai <- df.ai[,c(1,5,8)] # keeping only useful column
#View(df.ai)

tweets_text_network <- Corpus(VectorSource(df.ai$text)) # creating a text corpus used of the network

df.ai$text <- removeWords(df.ai$text, c("ai", "artificial intelligence", "artificialintelligence", "intelligence")) # removing some other particualar words
df.ai$text <- stripWhitespace(df.ai$text)                       # removing white spaces
df.ai$text <- gsub(" +", " ", df.ai$text)                       # removing spaces
   
df.ai$text <- gsub("\\s+", " ", str_trim(df.ai$text))
df.ai <- df.ai[!(df.ai$text == " "), ]  # removing tweet with only spaces
df.ai <- df.ai[!(df.ai$text == ""), ] # removing empty tweet
df.ai <- df.ai[nchar(as.character(df.ai$text)) > 3, ]

#dim(df.ai) # left with
#View(df.ai)
#write.csv(df.ai,file=paste("df.ai.csv"))

tweets_text_corpus <- Corpus(VectorSource(df.ai$text)) # creating another text corpus for the word cloud
```

### Understanding the word significance

Creating a wordcloud in order to visualize the most important words

``` r
wordcloud(tweets_text_corpus, min.freq = 7, max.words = 200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))#, scale= c(3,0.25))
```

![](AI_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

From the Word cloud it can be seen that the most used words are the ones
related with the technological world, machine, big-data, technology and
machine learning. Nevertheless are present words that concern the
business area like business, marketing or industry.

Displaying a bar-chart of the words that have an occurrency greater than
150 times.

``` r
dtm <- DocumentTermMatrix(tweets_text_corpus)

freq = colSums(as.matrix(dtm)) # let's see the most frequent words 
ord.df <- data.frame(words = names(freq), count = freq)
ord.df <- ord.df[order(-ord.df$count),]
ord.df <- ord.df[which(ord.df$count >= 150),]
ord.df$words <- factor(ord.df$words, levels = ord.df$words)
ord.df = ord.df[-3,]
#View(ord.df)
```

``` r
ggplot(data = ord.df, aes(x = words, y = count)) + 
  geom_col(stat = 'identity', width = 0.7, alpha = 0.7, fill = "#009999") + coord_flip() +
    labs(title="Barplot of the most used words", y="Occurence", x="Words") +
      theme_classic() + theme(panel.border = element_rect(colour = "black", fill=NA))
```

    ## Warning: Ignoring unknown parameters: stat

![](AI_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

This plot allows us to have a better idea of the occurrence of the
words. As already said the most important ones are related to technology
but we can see that both business and COVID-19 have more than 200
occurrences.

### Understanding the correlation among the words

Creating the network text

``` r
corp_net <- corpus(tweets_text_network)
tokns <- tokens(corp_net)
tokns<- tokens_remove(tokns, pattern = stopwords('en'))
tokns <- tokens_select(tokns,c("t","s",'"',"'", '-'), selection = "remove")
fcmat <- fcm(tokns, context = "document", tri = FALSE)
feat <- names(topfeatures(fcmat, 30))
fcm_select(fcmat, pattern = feat) %>% textplot_network(min_freq = 0.7)
```

![](AI_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

Because the selection of the Tweets was made by searching for tweets
with the hashtag *\#AI* this term is in the center and it is the one
that connected with all the others. Thanks to this visualization we can
notice how the most used and linked words (the ones with the marked blue
lines) are the ones that are most related to the technology where others
are less mentioned and less related together.

### Topic Modelling

For what I have stated before I will try a two topic modelling.

``` r
#Set parameters for Gibbs sampling
seed <-list(2003,5,63,100001,765)

#Run LDA using Gibbs sampling
ldaOut <-LDA(dtm,2, method="Gibbs", control=list(nstart=5, seed = seed, best=T, burnin = 4000, iter = 2000, thin=500))
```

``` r
terms = as.matrix(terms(ldaOut,20))
terms
```

    ##       Topic 1     Topic 2          
    ##  [1,] "can"       "data"           
    ##  [2,] "business"  "learning"       
    ##  [3,] "covid"     "new"            
    ##  [4,] "help"      "machinelearning"
    ##  [5,] "future"    "iot"            
    ##  [6,] "market"    "machine"        
    ##  [7,] "industry"  "via"            
    ##  [8,] "digital"   "technology"     
    ##  [9,] "use"       "read"           
    ## [10,] "time"      "using"          
    ## [11,] "gpt"       "tech"           
    ## [12,] "top"       "bigdata"        
    ## [13,] "like"      "world"          
    ## [14,] "work"      "learn"          
    ## [15,] "global"    "analytics"      
    ## [16,] "human"     "now"            
    ## [17,] "robotics"  "best"           
    ## [18,] "robots"    "need"           
    ## [19,] "see"       "join"           
    ## [20,] "companies" "today"

From the top 20 words per topic it can be stated that topic one is more
Technical oriented presenting words like: data, machine learning, big
data, analytic and so one. The second topic result to be more general
with words like: market, future, global, industry, healthcare, digital.
This distinction may suggest that there is a distinction in knowledge
about this subject between the people that wrote the tweets.

Creating a column in the dataframe with the topic number

``` r
ldaOut.topics <- as.matrix(topics(ldaOut))
df.ai$topics <- ldaOut.topics
df.ai$topics <- as.factor(df.ai$topics)
#View(df.ai)
```

Plotting pie for different topics

``` r
 ggplot(df.ai, aes(x="", y = topics, fill = topics)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0) +
  theme_minimal() +
  theme(
  axis.title.y = element_blank(),
  panel.border = element_blank(),
  panel.grid=element_blank(),
  axis.ticks = element_blank(),
  plot.title=element_text(size=14, face="bold")
  ) + theme(axis.text.x=element_blank())
```

![](AI_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

This chart shows that the majority of the tweets are about topic two and
so there are more “soft” knowledge tweets than deep knowledge ones.

### Sentiment analysis

Performing the sentiment analysis on the tweets and adding the result to
the dataset

``` r
sentiment <- data.frame(get_nrc_sentiment(df.ai$text))

total <-  data.frame(sum = colSums(sentiment, na.rm = TRUE)) #create a dataset that contin the sum of the sentyment 

df.ai <- cbind(df.ai, sentiment)

# creating a dataset with the same of the sentiment according to the topic
sent.topic <- data.frame(topic_1 = colSums(df.ai[which(df.ai$topics == "1"), 5:14]))
sent.topic <- cbind(sent.topic, data.frame(topic_2 = colSums(df.ai[which(df.ai$topics == "2"), 5:14])))

# scaling in order to allow comparison
sent.topic$topic_1 <- abs(scale(sent.topic$topic_1))
sent.topic$topic_2 <- abs(scale(sent.topic$topic_2))

sent.topic_2 <- data.frame(Sentiment = rep(row.names(sent.topic), 2),
                           Topic = c( rep("1", length(sent.topic$topic_1)), rep("2", length(sent.topic$topic_2)) ),
                           Value = c(sent.topic$topic_1, sent.topic$topic_2))
#View(sent.topic_2)
```

Plotting

``` r
ggplot (data = total,aes(x = rownames(total), y  = sum, fill = rownames(total))) + 
 geom_bar(stat="identity", width=0.5) +
  theme_classic() +
   theme(legend.title = element_blank(),
        legend.position = "none") +
    labs(y = "Total Value", x= "Sentiments",title= "Sentiment value for the tweets") 
```

![](AI_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

The bar-plot shows that the positive sentiment is the one with the
highest total value by far and in general people tend to have a positive
view of the technology (the other two following sentiment are trust and
anticipation). This result could have been empathized by the fact that
many machine learning algorithms have been introduced due to COVID-19.
Indeed this result is supported by the high frequency in which the word
“covid” is detected. Moreover the plot shows that the negative (negative
and fear) sentiment are way less detected.

It is Interesting to look if there are differences in the sentiments
according to the topic.

``` r
ggplot(sent.topic_2, aes(x = Sentiment, y = Value , fill = Topic)) + 
 theme_classic() +
  geom_bar(stat="identity", position = "dodge") + 
   scale_fill_brewer(palette = "Set1") +
    labs(title= "Sentiment value for the tweets according to their topic") 
```

![](AI_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

This chart shows that there is not much difference in the sentiment of
tweets depending on the topic classification made before. Hence can be
stated that there ate not two clear fronts, pro and con AI.

Lastly I will classify the tweets according to a binary sentiment and
then display the most occurrenced words depending on negative or
positive tweets.

``` r
td <- tidy(dtm) 

sentiments <- td %>% 
   inner_join(get_sentiments("bing"), by = c(term = "word")) 

sentiments <- sentiments %>% 
              count(sentiment, term, wt = count) %>% 
              ungroup() %>% filter(n >= 5) %>% 
              mutate(n = ifelse(sentiment == "negative", -n, n)) %>% 
              mutate(term = reorder(term, n))

sentiments <- sentiments[sentiments$n >= 60 | sentiments$n <= -20,]

ggplot(sentiments, aes(term, n, fill = sentiment)) + 
  geom_bar(stat = "identity") +
  theme_classic() +
  labs(title= "Principal word for the contribution to the sentiment", y = "Words", x = "Contribution")  + coord_flip()
```

![](AI_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

The plot shows the most used words that characterize negative and
positive labeled tweets. As can be seen, regarding the positive ones,
most occurrence words recall an idea of beneficial and confidential
growth. On the other hand negative words underline a fear of something
that could be a problem, a risk.

``` r
head(df.ai[df.ai$text %like% "cloud", 1])
```

    ## [1] "ibm cloud forum highlights hybrid cloud post pandemic businesses data storage asean business"  
    ## [2] "machine learning market may set new growth story aibrain amazon anki cloudminds"               
    ## [3] "machine learning market may set new growth story aibrain amazon anki cloudminds jewi"          
    ## [4] "companies relying analytics cloud computervision blockchain ml driven solutions new norm"      
    ## [5] "former soundcloud founders launch e bike subscription service backed blueyard via mike butcher"
    ## [6] "digital world everything sensing connected intelligent application g cloud"

By looking at the tweets that contain the therm “cloud” it can be seen
that this word appears in negative labeled tweets because many of them
concern cyber-security.

``` r
head(df.ai[df.ai$text %like% "drones", 1])
```

    ## [1] "one advanced bird like robots ever made robotics drones uav biotech"                  
    ## [2] "machinelearning helps robot swarms coordinate case drones ml drone robotics robots mi"
    ## [3] "drones battlefield syrian regime troops captured black hornet nano spy drone"         
    ## [4] "amazing advanced drones controlled student s minds via robotics drone robots"         
    ## [5] "researchers improved system even spotting koalas bushfire areas using drones"         
    ## [6] "government agencies coming forward providing avenues drones help"

By checking the tweets that present the words “drone” can be seen that
only a couple of them talk about military drones and so that is why it
could have a negative connotation. The others seems to have a positive
sentiment so this could be a case of missclassificaiton.

# V. Conclusion, Advantages and Disadvantages

Resuming the research questions:

**Are there different conversation topics?**

Thanks to this analysis it can be stated that tweets on AI go from
technical to business tweets covering a wide range of topics.This can be
seem by the word-cloud. Element that shows technical words, more
business-related ones and even more colloquial ones. Nevertheless it has
been shown that, even if borders are not so marked, a topic modelling
can be done showing a more technical topic and a less one. As already
said this could be a result of the presence of two different classes of
users, one more competent than the other.

**What people think about this technology, what are the feelings about
the topic?**

One pivotal aspect of the analysis was the sentimental one.
Categorization shows an overall positive feeling with the presence of
important value for trust and anticipation. Moreover the categorization
in the two different topics seems not to have an influence on the
sentiment of tweets.

**Which are the words that have the most influence on the sentiment of
users?**

For completing the sentiment analysis I have displayed the most counted
words for positive and negative tweets. This analysis has shown that
both sentiment are characterized by the words that are facing the
future. This could be a sign of the relative young age of the discipline
and the huge steps made in the recent years leading to an undefined
future perspective.

Moreover an aspect that needs to be take into account is the high
presence of the term “covid”. In the past few month it was widely
broadcasted by the news the importance that artificial intelligence
algorithms can play in the battle against this pandemic. This aspect
could have helped developing the overall positive sentiments found in
the analysis.

The different techniques used allowed a fast overview of the topic
giving easy and fast readable results. Moreover the previous steps can
be scaled using larger amount. Furthermore the previous methods allow to
perform a good analysis even with only free standards Twitter API, where
for example using a survey would have been by far more expensive.

On the down side the previous analysis has not taken into account any
information about the users. With other methodologies, for example by
analyzing the results of a survey, more detailed insight could have been
obtained. For example by asking personal information such as the level
of instruction, age, working experience, living area, perhaps a
correlation would have been found between these parameters and the
opinion on AI leading to a more detailed analysis. Nevertheless, in 2019
a group of researchers have conducted a survey on 2 791 adult U.S.
present on the microblogging platform. This research has highlighted
that there exists significance difference between U.S. overall
population and the sample’s one. Twitter users are younger, on average
they are seven years younger. They are 11% more likely to have, at
least, one bachelor’s degree and an income that is 9% higher than the
one of the total populations. Twitter users are more likely to be
Democrats and to have different opinion regarding migrants, gender, and
race. The last aspect concerns the activity of the users, the 10% of
them produce the 80% of the total amount of contents. [12] Given these
hypotheses it is difficult to generalize the results obtained to the
whole population.

Moreover the methodology used has not taken into account any type of
historical data, hence, it is not possible to evaluate the any kind of
trend in the opinion of people. Furthermore it would have been
interesting to analyse historical data in correlation with particular
global phenomena and so test the change in the sentiment according to
other facts. For example the perception on this theme may have been
changed before and after the Cambridge Analytica scandal and it would
have been interesting to understand whether this change of perception
has been maintained or not.

Another disadvantage of the technique used is for example the potential
misclassification of words for the sentiment analysis, as shown for the
“drone” word. This could have lead to a not so accurate analysis of the
tweets and so to a not complete correct perception of them.

In conclusion it can be stated that the overall opinion about artificial
intelligence is positive showing, significant expectations for the
future, but always with an eye for possible risks.

[1] For the definition of the concept of Artificial Intelligence:
European Commission, Brussels, 25-04-2018
<http://www.governo.it/sites/new.governo.it/files/CommunicationArtificialIntelligence.pdf>

[2] Lin Li, Lixin Qin, Zeguo Xu, Youbing Yin, Xin Wang, Bin Kong, Junjie
Bai, Yi Lu, Zhenghan Fang, Qi Song, Kunlin Cao, Daliang Liu, Guisheng
Wang, Qizhong Xu, Xisheng Fang, Shiqin Zhang, Juan Xia, Jun Xia,
*Artificial Intelligence Distinguishes COVID-19 from Community Acquired
Pneumonia on Chest CT*, Radiology, 2020, (doi :
<https://doi.org/10.1148/radiol.2020200905>)

[3] J.Clement, *Social media - Statistics & Facts*, Statista, 2020 ,
available at: <https://www.statista.com/topics/1164/social-networks/>

[4] T. K. Das, D. P. Acharjya and M. R. Patra, *Opinion mining about a
product by analyzing public tweets in Twitter*, 2014 International
Conference on Computer Communication and Informatics, Coimbatore, 2014,
pp. 1-4 (doi: 10.1109/ICCCI.2014.6921727.)

[5] B. Gokulakrishnan, P. Priyanthan, T. Ragavan, N. Prasath and A.
Perera, *Opinion mining and sentiment analysis on a Twitter data
stream*, International Conference on Advances in ICT for Emerging
Regions (ICTer2012), Colombo, 2012, pp. 182-188 (doi:
10.1109/ICTer.2012.6423033)

[6] R. K. Bakshi, N. Kaur, R. Kaur and G. Kaur, *Opinion mining and
sentiment analysis*, 2016 3rd International Conference on Computing for
Sustainable Global Development (INDIACom), New Delhi, 2016, pp. 452-455.

[7] B.Sampath Kumar, D.Bhanu Sree Reddy, *An Analysis on Opinion Mining:
Techniques and Tools*, Indian Journal of Research (2016)

[8] A.Go, R. Bhayani, *Twitter Sentiment Classification using Distant
Supervision*,Stanford: CS224N Project Report , 2009

[9] S.Yang, H.Zhang, *Text Mining of Twitter Data Using an LDA Topic
Model and Sentiment Analysis*, World Academy of Science, Engineering and
Technology International Journal of Computer and Information Engineering
Vol:12, No:7, 2018

[10] A. Alamsyah, W. Rizkika, D. D. A. Nugroho, F. Renaldi and S.
Saadah, *Dynamic Large Scale Data on Twitter Using Sentiment Analysis
and Topic Modeling*, 2018 6th International Conference on Information
and Communication Technology (ICoICT), Bandung, 2018, pp. 254-258, doi:
10.1109/ICoICT.2018.8528776.

[11] R. J. Medford, S. N. Saleh, A. Sumarsono, T. M. Perl, C. U.
Lehmann, *An “Infodemic”: Leveraging High-Volume Twitter Data to
Understand Public Sentiment for the COVID-19 Outbreak*, 2020 (doi:
<https://doi.org/10.1101/2020.04.03.20052936>) (Preprints and
early-stage research may not have been peer reviewed yet.)

[12] S. Wojcik, A. Hughes, *Sizing Up Twitter Users*, Pew Research
Center, 2019 Twitter Users”
