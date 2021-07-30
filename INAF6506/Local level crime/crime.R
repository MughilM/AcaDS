## DSPP predicting local-level crime


library(geosphere)
library(ggplot2)
library(RColorBrewer) 
library(sp)
library(maptools)
library(maps)
library(data.table)




# setwd("")

#-#-#-#-#-#-#-#-#-#
# 1. Chicago data
#-#-#-#-#-#-#-#-#-#

# Chicago crime data
chicago = read.csv("Chicago/chicago_crimes_2018.csv", as.is=T)
chicago$date = as.Date(as.POSIXct(chicago$Date, format= "%m/%d/%Y %H:%M:%OS"))

# Chicago map
chicago_shape_files = readShapePoly("Chicago/Boundaries - Police Beats (current)/geo_export_fb1cc8b1-500e-49cd-a492-47f82d2121a5.shp")

## Aggregate to date and Community area level
chicago_aggregated = aggregate(chicago$ID, by=list(chicago$date, chicago$Community.Area), FUN=length)
colnames(chicago_aggregated) = c("date", "Community.Area", "num_crimes")
chicago_aggregated$num_days = chicago_aggregated$date - as.Date("2018-01-01")
chicago_aggregated$week = as.numeric(floor(chicago_aggregated$num_days/7))
# Lag number of crimes for each location
chicago_aggregated = data.table(chicago_aggregated)
chicago_aggregated[, num_crimes_lagged:=c(NA, num_crimes[-.N]), by=Community.Area]
base::save(chicago_aggregated, file="Chicago/crimes_aggregated.rdata")

# Example: aggregate arsons 
chicago_arson = chicago[chicago$Primary.Type=="ARSON",]
chicago_aggregated_arson = aggregate(chicago_arson$ID, by=list(chicago_arson$date, chicago_arson$Community.Area), FUN=length)
colnames(chicago_aggregated_arson) = c("date", "Community.Area", "num_crimes")
chicago_aggregated_arson$num_days = chicago_aggregated_arson$date - as.Date("2018-01-01")
chicago_aggregated_arson$week = as.numeric(floor(chicago_aggregated_arson$num_days/7))
# Lag number of crimes for each location
chicago_aggregated_arson = data.table(chicago_aggregated_arson)
chicago_aggregated_arson[, num_crimes_lagged:=c(NA, num_crimes[-.N]), by=Community.Area]
base::save(chicago_aggregated_arson, file="Chicago/crimes_aggregated_arson.rdata")



#-#-#-#-#
# 2. UK 
#-#-#-#-#

# UK crime data
base::load("UK/police_incidents_df.rdata")
# UK map
base::load("UK/uk_areas.rdata")

## Aggregate to the month and police area 
uk_aggregated = aggregate(police_incidents$Crime.ID, by=list(police_incidents$Month, police_incidents$Reported.by), FUN=length)
colnames(uk_aggregated) = c("month", "police_area", "num_crimes")
uk_aggregated$date = paste0(as.character(uk_aggregated$month), "-01")
uk_aggregated$num_days = as.Date(uk_aggregated$date) - as.Date("2014-10-01")
uk_aggregated$month = as.numeric(floor(uk_aggregated$num_days/30))
# Lag number of crimes for each location
uk_aggregated = data.table(uk_aggregated)
uk_aggregated[, num_crimes_lagged:=c(NA, num_crimes[-.N]), by=police_area]
base::save(uk_aggregated, file="UK/uk_aggregated.rdata")

# Example: aggregate Anti-social behaviour
uk_anti_social = police_incidents[police_incidents$Crime.type=="Anti-social behaviour",]
uk_aggregated_anti_social = aggregate(uk_anti_social$Crime.ID, by=list(uk_anti_social$Month, uk_anti_social$Reported.by), FUN=length)
colnames(uk_aggregated_anti_social) = c("month", "police_area", "num_crimes")
uk_aggregated_anti_social$date = paste0(as.character(uk_aggregated_anti_social$month), "-01")
uk_aggregated_anti_social$num_days = as.Date(uk_aggregated_anti_social$date) - as.Date("2014-10-01")
uk_aggregated_anti_social$month = as.numeric(floor(uk_aggregated_anti_social$num_days/30))
# Lag number of crimes for each location
uk_aggregated_anti_social = data.table(uk_aggregated_anti_social)
uk_aggregated_anti_social[, num_crimes_lagged:=c(NA, num_crimes[-.N]), by=police_area]
base::save(uk_aggregated_anti_social, file="UK/uk_aggregated_anti_social.rdata")


#-#-#-#-#-#-#
# 3. Germany
#-#-#-#-#-#-#

## Germany hate crime data
devtools::install_github("davben/arvig")
library(arvig)
data("arvig")
arvig = as.data.frame(arvig)
# Germany map
germany_shape_files = readShapePoly("Germany/VG250_1Jan2011_WGS84/VG250_Bundeslaender.shp")


## Aggregate to date and Community level
germany_aggregated = aggregate(arvig$date, by=list(arvig$date, arvig$community_id), FUN=length)
colnames(germany_aggregated) = c("date", "community_id", "num_crimes")
germany_aggregated$num_days = germany_aggregated$date - as.Date("2014-01-01")
germany_aggregated$week = as.numeric(floor(germany_aggregated$num_days/7))
# Lag number of crimes for each location
germany_aggregated = data.table(germany_aggregated)
germany_aggregated[, num_crimes_lagged:=c(NA, num_crimes[-.N]), by=community_id]
base::save(germany_aggregated, file="Germany/germany_aggregated.rdata")

# Example: aggregate arsons 
germany_arson = arvig[arvig$category_en=="arson",]
germany_aggregated_arson = aggregate(germany_arson$date, by=list(germany_arson$date, germany_arson$community_id), FUN=length)
colnames(germany_aggregated_arson) = c("date", "community_id", "num_crimes")
germany_aggregated_arson$num_days = germany_aggregated_arson$date - as.Date("2014-01-01")
germany_aggregated_arson$week = as.numeric(floor(germany_aggregated_arson$num_days/7))
# Lag number of crimes for each location
germany_aggregated_arson = data.table(germany_aggregated_arson)
germany_aggregated_arson[, num_crimes_lagged:=c(NA, num_crimes[-.N]), by=community_id]
base::save(germany_aggregated_arson, file="Germany/germany_aggregated_arson.rdata")


## Useful functions

# 1. Plot a map

# Chicago
ggplot() +
  geom_path(data = chicago_shape_files, aes(x = long, y = lat, group = group)) +
  geom_point(aes(x = Longitude, y = Latitude,  colour = "red"), data = chicago, size = 0.5)

# UK
ggplot() +
  geom_path(data = uk_shape_files, aes(x = long, y = lat, group = group)) +
  geom_point(aes(x = Longitude, y = Latitude,  colour = "red"), data = police_incidents, size = 0.5)

# Germany
ggplot() +
  geom_path(data = germany_shape_files, aes(x = long, y = lat, group = group)) +
  geom_point(aes(x = longitude, y = latitude,  colour = "red"), data = arvig, size = 0.5)

# 2. Plot a heatmap (using Germany as an example)

ggplot() +
  geom_path(data = germany_shape_files, aes(x = long, y = lat, group = group)) +
  stat_density2d(
  aes(x = longitude, y = latitude, fill = ..level.., alpha = 0.25),
  size = 0.01, bins = 30, data = arvig,
  geom = "polygon"
)

# 3. Compute distance between points

# (In this example, we compute the distance between each hate crime in Germany
# and the location of the federal office of migration)
arvig$distance_to_federal_office_of_migration =  distm(arvig[,c("longitude", "latitude")], c(11.098083, 49.430583), fun = distHaversine)

