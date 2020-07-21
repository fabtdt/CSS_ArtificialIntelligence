# CSS_ArtificialIntelligence

The file AI.Rmd is a R markdown document that contains the code and the text of the work. The code can be runned without the need of connecting to the Twitter API.
All the computation are performed using the previous created dataset.
The code can be runned with new data by connecting it to the Twitter API, to do this you must:
   
1. create or use an already existing developer Twitter account
2. insert the credential into the cell "key and toke" by substituting the value to the "XXXXX"
3. from the cells: "key and token", "connecting to Twitter", "requiring the tweets", "data preparation", delete "eval = FALSE" from {r eval = FALSE} in order to have
only {r}
4. from the cell "reading and further preparation" comment the lines 138 and 139: <br>
*df.ai <- read.csv("df.ai.csv", header = T) # reading* <br>
*df.ai <- df.ai[,c(2,6,9)] # keeping only useful column* <br>
and uncomment the line 141: *#df.ai <- df.ai[,c(1,5,8)] # keeping only useful column*
        
 The file AI.html is a html knit of the previous file and it represent the complete and final version of the work.
 **In order to be displayed this file need to be downladed.**
 
 The file df.ai.csv is the dataset in csv format created at the beginning and it is the one from which the tweet are taken the analysis in the work.
