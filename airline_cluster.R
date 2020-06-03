################
## Clustering ##
################

library(caret)

### PART 1 : exploring and scaling the data

# after setting the current directory, loading the airline data 
airline <- read.csv("data/airline.csv")
str(airline)

airline
# Balance: Number of miles eligible for award travel
# BonusTrans: Number of non-flight bonus transactions in the past 12 months
# BonusMiles: Number of miles earned from those transactions
# FlightTrans: Number of flight transactions
# FlightMiles: Number of miles earned from those transactions
# DaysSinceEnroll: Tenure in the program (days)

# Need to preprocess the data to treat each column equally to compute the clusters

# preprocessing steps:
# First, *center* the data (substract the mean to each column)
# => mean becomes 0 for each column
# Then, *scale* the data, by dividing by the standard deviation
# => std becomes 1 for each column

#step 1: create the pre-processor using preProcess
pp <- preProcess(airline, method=c("center", "scale"))   
# normalization for each col: (X_i-mean)/std
class(pp)
pp
pp$mean


sd(airline$Balance/sd(airline$Balance))

#step 2: apply it to the dataset
airline.scaled <- predict(pp, airline)

# Sanity check
colMeans(airline)
colMeans(airline.scaled)# mean is (approximately) 0 for all columns
apply(airline.scaled,2,sd)# standard deviation is 1 for all columns 

# (apply() applies function given as third argument to matrix given as first argument. 
# 2 means apply sd() to cols. 1 would apply it row-wise, col(1,2) both to row and column.)
head(airline.scaled)
# What does a negative value represent?

### PART 2 : Clustering: K-Means

# k-means has a random start (where the centroids are initially randomly located)
# set the seed to have the same result
set.seed(144)

# The kmeans function creates the clusters
# set the number of k=8
km <- kmeans(airline.scaled, centers = 8, iter.max=100) 
# centers randomly selected from rows of airline.scaled

class(km) # class: kmeans
names(km)

# cluster centroids. Store this result
km.centroids <- km$centers
km.centroids
# cluster for each point. Store this result.
km.clusters <- km$cluster
km.clusters
# the sum of the squared distances of each observation from its cluster centroid => cluster dissimilarity
km$tot.withinss  # cluster dissimilarity: 8289.099

# the number of observations in each cluster
km.size <- km$size
km.size # 893 1124  504  212 1107   69   76   14

km$centers
# denormalization for each col: 
# X_i=(x*std)+mean

x_i=(km$centers*
       
km_center <- km$centers
x_i <- (km_center[,1]*sd(km_center[,1]))+mean(km_center[,1])

sd(airline$Balance)
airline


# Scree plot for k-means
# For k means, we literally try many value of k and look at their dissimilarity.
# here we test all k from 1 to 100
k.data <- data.frame(k = 1:100)
k.data$SS <- sapply(k.data$k, function(k) {
  kmeans(airline.scaled, iter.max=100, k)$tot.withinss
})

# Plot the scree plot.
plot(k.data$k, k.data$SS, type="l")
plot(k.data$k, k.data$SS, type="l", xlim=c(0,40))
axis(side = 1, at = 1:10)


### PART 3 : Hierarchical Clustering
# Compute all-pair euclidian distances between the observations
d <- dist(airline.scaled)    # method = "euclidean"
class(d)

# Creates the Hierarchical clustering
hclust.mod <- hclust(d, method="ward.D2")
# The "method=ward.D2" indicates the criterion to select the pair of clusters to be merged at each iteration

# Now, plot the hierarchy structure (dendrogram)
# labels=F (false) not to print text for each of the 3999 observations
plot(hclust.mod, labels=F, ylab="Dissimilarity", xlab = "", sub = "")

# To select a "good" k value, pick something that defines the corner / pivot in the L (knee)
# the next line puts this data in the right form to be plotted
hc.dissim <- data.frame(k = seq_along(hclust.mod$height),   # index: 1,2,...,length(hclust.mod$height)
                        dissimilarity = rev(hclust.mod$height)) # reverse elements
head(hc.dissim)

# Scree plot
plot(hc.dissim$k, hc.dissim$dissimilarity, type="l")
# Let's zoom on the smallest k values:
plot(hc.dissim$k, hc.dissim$dissimilarity, type="l", xlim=c(0,40))
axis(side = 1, at = 1:10)

# Improvement in dissimilarity for increasing number of clusters
hc.dissim.dif = head(hc.dissim,-1)-tail(hc.dissim,-1)
head(hc.dissim.dif,10)

# now that we have k=7, construct the clusters
h.clusters <- cutree(hclust.mod, 7)
h.clusters

# The *centroid* for a cluster is the mean value of all points in the cluster: 
aggregate(airline.scaled, by=list(h.clusters), mean) # Compute centroids

# *size* of each cluster
table(h.clusters) # 1242  833  746  723  271   63  121

# many zeros mean clusters from kmeans and hierarchical "match up"
table(h.clusters, km.clusters)

###Part 4: Potential Marketing

