library(RColorBrewer)
library(ggplot2)
library(flexclust)
library(caret)
library(cluster) 
library(dplyr)

##### Airline Data

### Importation of data and normalization

# Data import
airline <- read.csv("AirlinesCluster.csv")

# Data display
head(airline, 16)
tail(airline, 5)
colMeans(airline)
apply(airline, 2, sd)

# Normalization
pp <- preProcess(airline, method=c("center", "scale"))
airline.scaled <- predict(pp, airline)

# Display of nromalized data
head(round(airline.scaled, 2), 16)
tail(round(airline.scaled, 2), 5)

### k-means

## k-means with k=8

set.seed(144)
# Running the k means algorithm
mod <- kmeans(airline.scaled, iter.max=100, 8)
cluster.assignment.kmeans <- mod$cluster

# Summary of results
t(round(mod$centers, 2))
round(t(mod$centers) * pp$std + pp$mean)
table(mod$cluster)

## Scree plot: k means for all values of k

# Running the k means algorithm for each k
dat <- data.frame(k = 1:100)
dat$SS <- sapply(dat$k, function(k) {
  set.seed(144)
  kmeans(airline.scaled, iter.max=100, k)$tot.withinss
})

# Plotting the results
print(ggplot(dat, aes(x=k, y=SS)) +
        geom_line(lwd=2) +
        theme_bw() +
        xlab("Number of Clusters (k)") +
        ylab("Within-Cluster Sum of Squares") +
        ylim(0, 25000) +
        theme(axis.title=element_text(size=18), axis.text=element_text(size=18)))


### Hierarchical clustering

## Running the hierarchical clustering algorithm

d <- dist(airline.scaled)
mod.hclust <- hclust(d, method="ward.D2")

## Dendrogram


dissimilarity.all <- c(75,50,32,25,20,15)

plot(mod.hclust, labels=F, xlab=NA, ylab="Dissimilarity", sub=NA, main=NA)


plot(mod.hclust, labels=F, xlab=NA, ylab="Dissimilarity", sub=NA, main=NA)


## Scree plot

# Varying the number of clusters
dat.hc.airline <- data.frame(nclust = seq_along(mod.hclust$height),
                             dissimilarity = rev(mod.hclust$height))

# Plotting the results
print(ggplot(dat.hc.airline, aes(x=nclust, y=dissimilarity)) +
        geom_line(lwd=2) +
        theme_bw() +
        xlab("Number of Clusters") +
        ylab("Dissimilarity") +
        xlim(0, 100) +
        theme(axis.title=element_text(size=18), axis.text=element_text(size=18)))


## Analyze hirearchical clustering results with 7 clusters

# Assignment of data points to clusters
assignments <- cutree(mod.hclust, 7)

# Display clusters
round(sapply(split(airline.scaled, assignments), colMeans), 2)
table(assignments)

##### Automobile data

### Importation of data and normalization

# Data import
data(auto)

# Data wrangling
auto.cutdown <- auto[,c("ch_driving_properties", "ch_interior", "ch_technology", "ch_comfort", "ch_reliability",
                        "ch_handling", "ch_power", "ch_consumption", "ch_sporty", "ch_safety")]
for (k in names(auto.cutdown)) {
  auto.cutdown[,k] <- as.numeric(auto.cutdown[,k])
}
names(auto.cutdown) <- substr(names(auto.cutdown), 4, nchar(names(auto.cutdown)))

# Display
head(auto.cutdown, 16)
tail(auto.cutdown, 5)

### Hierarchical clustering

# Running the hierarchical clustering algorithm
d <- dist(auto.cutdown)
mod.hclust <- hclust(d, method="ward.D2")

# Plotting the dendogram

plot(mod.hclust, labels=F, xlab=NA, ylab="Dissimilarity", sub=NA, main=NA)

plot(mod.hclust, labels=F, xlab=NA, ylab="Dissimilarity", sub=NA, main=NA)


# Computing the scree plot
dat.hc.auto <- data.frame(nclust = seq_along(mod.hclust$height),
                          dissimilarity = rev(mod.hclust$height))

# Scree plot

print(ggplot(dat.hc.auto, aes(x=nclust, y=dissimilarity)) +
        geom_line(lwd=2) +
        theme_bw() +
        xlab("Number of Clusters") +
        ylab("Dissimilarity") +
        xlim(0, 100) +
        theme(axis.title=element_text(size=18), axis.text=element_text(size=18)))

# Assignment of data points to clusters
assignments <- cutree(mod.hclust, 7)

# Display clusters
round(sapply(split(auto.cutdown, assignments), colMeans), 2)
table(assignments)
