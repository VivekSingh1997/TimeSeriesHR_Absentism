# This code is based on the ideas and pseudo code in Compression of Time Series
# by Extracting Major Extrema by Eugene Fink and harith Suman Gahdiu
# Link is https://pdfs.semanticscholar.org/96e2/2e86768c9569e1185927670de7a63fa5f9e0.pdf
# The code calculates all minimums and maximums in a time series and categorizes
# then a strict, left, right and flat.
# The code then calculates the most important minimums and maximums based on a distance
# functions.
# This can be used for peak values.
# It can also be used to find similar patterns in other data time series data sets.
# This code is at the crew level.

Rbin <- file.path(R.home(), "bin", "R");

.libPaths('E:/HR_Analytics_R_program/R-3.4.1/library')
.libPaths() 

#.libPaths('E:/R_library_3_4_1_latest')
#.libPaths()

library("RPostgreSQL")
library("DBI")
#remove.packages("dplyr")
#install.packages('dplyr', dependencies = TRUE)
library("dplyr")
#remove.packages("ggplot2")
#install.packages('ggplot2', dependencies = TRUE)
library(ggplot2)
library(reshape2)
#library(vcd)
#library(fitdistrplus)
library(tseries)
#library(forecast)
#library(dynlm)
#library(xts)
library(tidyr)
# library(tsoutliers)
# library(RSNNS)
# library(nnfor)
# library(EMD)
# library(Rssa)
library(lubridate)
library(zoo)
# library(rugarch)
# library(zipcode)
# library(cumstats)
# library(rpart)
# library(rpart.plot)
#library(Hmisc)
#library(tidyverse)
#library(stats)
#library(KFAS)
#library(Rtsne)
#library(hot.deck)
# library(ggfortify)
# library(factoextra)
# library(cluster)
# library(dbscan)

print("Packages Loaded Sucessfully")

drv <- dbDriver("PostgreSQL")
host = 'shbdmdwp001.servers.chrysler.com'
port = 5432
user = 'datasci' 
database = 'odshawq' 
password = 'datasci_01'

print("Executing WriteDB Function")

write_to_db <- function(df_name, table_name, grant_statement,increments){
  
  #Removing invalid characters 
  
  df_name <- df_name[, !(names(df_name) %in% c("X"))]
  # 
  # #Removing Special Characters from columns
  # 
  # for ( colname in names(df_name))
  # {
  #   index <-grep(colname, colnames(df_name))
  #   names(df_name)[index] <- colname <-gsub("[^0-9A-Za-z///' ]","_" , colname ,ignore.case = TRUE)
  # }
  # 
  # # Column Names 
  # 
  df_names<-colnames(df_name)
  # 
  # 
  
  # Datatypes of Dataframe
  
  df_dtypers<- sapply(df_name,class)
  
  
  #Create the Table Name with Schema Name:: 
  
  
  #Specify Column to be distributed as Ex::  "BY (column_name)"
  
  distributed_by<- "RANDOMLY;"    
  drop_table<-"drop table if exists"
  create_sql<- paste(drop_table,table_name,";","CREATE TABLE", table_name ,"(")
  distbuted_sql<- paste(") DISTRIBUTED ", distributed_by)
  
  ################ Datatypes Mapping to datatypes of Green Plum
  
  l <- c("character" = "text", "factor" = "text", "Date" ="date", "numeric" = "float", "integer" = "int" )
  
  
  
  for( col in df_names)
    
  {   
    if ( df_dtypers[[col]] %in% names(l))
    {
      
      df_dtypers[[col]]<-l[[df_dtypers[[col]]]]
    }
    
    else
    { 
      df_dtypers[[col]]<-"text"
      
    }
  }
  
  
  
  #Zipping the Dataframe names and Datatypes
  
  zipped<-as.list(paste(df_names, df_dtypers))
  concat<-NULL
  x<-1
  for (colname in zipped ){
    if(length(zipped) > x )
    { x<-x+1 
    concat<-paste(concat,colname, ",") }
    else {
      concat<-paste(concat,colname)
    }}
  
  
  # SQL Query for creating the table, If in case of ERROR: Please check for datatypes and replace with required datatypes 
  # using gsub("String to be Replace", "With String", string) 
  
  sql_create<-paste(create_sql,concat,distbuted_sql )
  
  #print(sql_create)               #Check SQL Statement
  
  
  # SQL Query for creating the table, If in case of ERROR: Please check for datatypes and replace with required datatypes 
  # using gsub("String to be Replace", "With String", string) 
  
  #sql_create<-paste(create_sql,concat,distbuted_sql )
  
  print(sql_create)               #Check SQL Statement
  
  
  # Run the sql command to create table ::
  
  print(paste0( "Table ==> ",table_name," Table Created !!", dbGetQuery(con, sql_create)))
  
  #dbGetQuery(con, "drop table if exists lab_datasci.Test; create table lab_datasci.Test ( a text) ")
  
  
  #increments<-500
  
  end_row<- 0
  
  for ( start_row in seq(1,nrow(df_name), increments))
    
  { if ( nrow(df_name) - start_row > increments)
  {
    end_row<-start_row + increments-1
    valuePairs<-data.frame(df_name[start_row:end_row,])
    values<-paste0(apply(valuePairs, 1, function(x) paste0("(", paste0(lapply(strsplit(as.character(x), ",",fixed=TRUE), function(z) ( paste0("$xzwmnp$",z ,"$xzwmnp$") )), collapse = ", "), ")")), collapse = ", ")
    #values<-paste0(apply(valuePairs, 1, function(x) paste0("('", paste0(lapply(strsplit(x, ",",fixed=TRUE), function(z) (gsub("'","''",z)) ) , collapse = "', '"), "')")), collapse = ", ")
    insert<-paste0("INSERT INTO ",table_name," VALUES ",values,";")
    dbGetQuery(con, insert)
  }
    else 
    { 
      start_row<- end_row + 1
      end_row<-nrow(df_name)
      valuePairs<-data.frame(df_name[start_row:end_row,])
      #col_names<-colnames(valuePairs)
      values<-paste0(apply(valuePairs, 1, function(x) paste0("(", paste0(lapply(strsplit(as.character(x), ",",fixed=TRUE), function(z) (paste0("$xzwmnp$",z,"$xzwmnp$") ) ), collapse = ", "), ")")), collapse = ", ")
      insert<-paste0("INSERT INTO ",table_name," VALUES ",values,";")
      dbGetQuery(con, insert)
      break
      
    }
  }
  
  
  #Check for Creation
  
  print(paste0("Count of Created Table:: ",dbGetQuery(con, paste0("SELECT COUNT(*) FROM ",table_name))))
  
  
  
  # 
  
  qry_output = sprintf(grant_statement, table_name, table_name, table_name, table_name)
  dbGetQuery(con,qry_output) 
  
}
# End if write to db function
grant_statement <- "GRANT ALL ON TABLE %s TO hawqadmin;
GRANT ALL ON TABLE %s TO datasci;
GRANT SELECT ON TABLE %s TO hawq_read_only;
GRANT SELECT ON TABLE %s TO hrba;"

####################################
####################################
# find min function

find_min <- function(i, subset, n){
  left = i
  while (i < n & subset$actual[i] >= subset$actual[i + 1]){
    i <- i + 1
    if (subset$actual[left] > subset$actual[i]) {left <- i}
  }
  if (i < n) {
    extrema <- list(left, i, "min", i + 1)
    return(extrema)
  }else{
    extrema <- list(left, i, "end", i + 1)
    return(extrema)
  }
  
}
#########################################
#############################################

# find max function

find_max <- function(i, subset, n){
  left = i
  while (i < n & subset$actual[i] <= subset$actual[i + 1]){
    i <- i + 1
    if (subset$actual[left] < subset$actual[i]) {left <- i}
  }
  
  if (i < n) {
    extrema <- list(left, i, "max", i + 1)
    return(extrema)
  }else{
    extrema <- list(left, i, "end", i + 1)
    return(extrema)
  }
}

find_first <- function(i,  subset, n, R){
  i <- 1
  leftMin <- 1
  rightMin <- 1
  leftMax <- 1
  rightMax <- 1
  while (i < n & 
         abs(subset$actual[i + 1] - subset$actual[leftMax]) < R & 
         abs(subset$actual[i + 1] - subset$actual[leftMin]) < R ) {
    i <- i + 1
    if(subset$actual[leftMin] > subset$actual[i]){leftMin <- i}
    if(subset$actual[rightMin] >= subset$actual[i]){rightMin <- i}
    if(subset$actual[leftMax] < subset$actual[i] ) {leftMax <- i}
    if(subset$actual[rightMax] <= subset$actual[i]) {rightMax <- i}
  }
  i <- i + 1
  if(i < n & subset$actual[i] > subset$actual[1]){
    if (leftMin == 1){
      for(t in leftMin + 1:rightMin){
        if(subset$actual[t] == subset$actual[leftMin]){
          leftMin <- t
        }
      }
    }
    extrema <- list(leftMin, rightMin, "imp_min", i )}
  if(i < n & subset$actual[i] < subset$actual[1]){
    if (leftMax == 1){
      for(t in leftMax + 1:rightMax){
        if(subset$actual[t] == subset$actual[leftMax]){
          leftMax <- t
        }
      }
    }
    extrema <- list(leftMax, rightMax , "imp_max", i )}
  return(extrema)
}

find_imp_min <- function(i, subset, n, dist_threshold){
  left = i
  right = i
  while (i < n & (subset$actual[i + 1] < subset$actual[left] | 
                  abs(subset$actual[i + 1] - subset$actual[left]) < dist_threshold )){
    i <- i + 1
    if (subset$actual[left] > subset$actual[i]) {left <- i}
    if (subset$actual[right] >= subset$actual[i]) {right <- i}
  }
  extrema <- list(left, right, "imp_min", i + 1)
  return(extrema)
}

find_imp_max <- function(i, subset, n, dist_threshold){
  left = i
  right = i
  while (i < n & (subset$actual[i + 1] > subset$actual[left] | 
                  abs(subset$actual[i + 1] - subset$actual[left]) < dist_threshold )){
    i <- i + 1
    if (subset$actual[left] < subset$actual[i]) {left <- i}
    if (subset$actual[right] <= subset$actual[i]) {right <- i}
  }
  extrema <- list(left, right, "imp_max", i + 1)
  return(extrema)
}

################################################################
################################################################

#### Input Arguments from command line #########################

args = commandArgs(trailingOnly=TRUE)


plants <- c("jnap", "shap", "tac", "wtap", "bvp")
# fill in the latest table date here


#tbl_date <- "20200202"
#plant <- "bvp"
#start_date <- "2018-01-01"
#test_flag <- "test"

###############################################################
print("Storing Input Arguments")

tbl_date <- args[1]

plant <- args[2]

start_date <-args[3]

test_flag <-args[4]

Rbin <- file.path(R.home(), "bin", "R");


###########################################################################
###########################################################################
###########################################################################
print("Connecting to Database")

con <- dbConnect(drv, dbname=database,host=host,port=port,user=user,password=password )

if (test_flag=="test"){

  tbl_name <-paste("lab_datasci.abs",test_flag,plant,"model_seq",tbl_date,"tbl", sep="_")
  } else {
  tbl_name<-paste("lab_datasci.abs",plant,"model_seq",tbl_date,"tbl", sep="_")
}


print(paste0("Reading table :: ", tbl_name))

################################################################################
#
# Data injestion
#
###############################################################################
qry <- paste0("select * from ", tbl_name, " where workdate > ","'" , start_date , "'")

qry_role<-paste0("set role to datasci;")

dbGetQuery(con, qry_role)

model_data <- dbGetQuery(con, qry)
########################################################
# Aggregate unplanned absences by crew

actual_data <- model_data %>% group_by(workdate, crew) %>% summarise(actual = sum(absences_unplanned)) %>%
  dplyr::select(workdate, crew, actual) %>% arrange(workdate)



######################################################
#Close Connection
#####################################################
print("Closing Connection")
lapply(dbListConnections(drv = dbDriver("PostgreSQL")), function(x) {dbDisconnect(conn = x)})
#dbDisconnect(con)
print("Closed Connection")

#####################################################

print(paste0("Running Compression for ", toupper(plant)))


#########################################
#########################################
# Code for finding all extrema by crew 

# find the unique crew 


loop_df <- actual_data %>% group_by(crew) %>% summarise(cnt = n())
plant_df <- data.frame()

for (j in 1:nrow(loop_df)){
  extrema_df <- data.frame()
  extrema_imp_df <- data.frame()
  
  subset <- actual_data %>% 
    filter(crew == loop_df$crew[j]) %>%
    arrange(workdate) 
  if (nrow(subset) > 90) {
    crew <- loop_df$crew[j]
    # first row is the end point
    i <- 2
    n <- length(subset$actual)
    while (i < n & subset$actual[i] == subset$actual[1]){
      i = i + 1
    }
    # find the first minimum
    if ((i < n) & (subset$actual[i] < subset$actual[1])){
      extrema <- find_min(i, subset,n)
      i <- unlist(extrema[2])
      left <- unlist(extrema[1])
      type <- unlist(extrema[3])
      row<-data.frame(plant, crew,  left, i, type)
      extrema_df <- rbind(extrema_df, row) 
      i <- unlist(extrema[4])
    }else{
      extrema <- find_max(i, subset,n)
      i <- unlist(extrema[2])
      left <- unlist(extrema[1])
      type <- unlist(extrema[3])
      row<-data.frame(plant, crew,   left, i, type)
      extrema_df <- rbind(extrema_df, row) 
      i <- unlist(extrema[4])
    }
    
    
    while (i < n){
      #print(c(i, n, type))
      if (type == "min"){
        extrema <- find_max(i, subset,n)
        i <- unlist(extrema[2])
        left <- unlist(extrema[1])
        type <- unlist(extrema[3])
        row<-data.frame(plant,crew,   left, i, type)
        extrema_df <- rbind(extrema_df, row) 
        i <- unlist(extrema[4])
      }else{
        extrema <- find_min(i, subset,n)
        left <- unlist(extrema[1])
        i <- unlist(extrema[2])
        type <- unlist(extrema[3])
        row<-data.frame(plant, crew,   left, i, type)
        extrema_df <- rbind(extrema_df, row) 
        i <- unlist(extrema[4])
        
      }
      
      
    }
    ######################################################
    ####################################################
    # find the important min and max points
    i <- 1
    n <- length(subset$actual)
    # need to calculate a threshold. Currently 1/4 max - min
    dist_threshold <- ceiling(.25 * (max(subset$actual) - min(subset$actual)))
    # find first important extrema
    imp_ext <- find_first(i, subset, n, dist_threshold)
    left <- unlist(imp_ext[1])
    right <- unlist(imp_ext[2])
    type <- unlist(imp_ext[3])
    i <- unlist(imp_ext[4])
    row<-data.frame(plant, crew,  left, right, type)
    extrema_imp_df <- rbind(extrema_imp_df, row) 
    # find the first important minimum
    if ((i < n) & (subset$actual[i] < subset$actual[1])){
      imp_ext <- find_imp_min(i, subset,n, dist_threshold)
      left <- unlist(imp_ext[1])
      right <- unlist(imp_ext[2])
      type <- unlist(imp_ext[3])
      i <- unlist(imp_ext[4])
      row<-data.frame(plant, crew,   left, right, type)
      extrema_imp_df <- rbind(extrema_imp_df, row) 
    }
    
    while (i < n){
      #print(c(i, n, type))
      if (type == "imp_min"){
        imp_ext <- find_imp_max(i, subset,n, dist_threshold)
        left <- unlist(imp_ext[1])
        right <- unlist(imp_ext[2])
        type <- unlist(imp_ext[3])
        i <- unlist(imp_ext[4])
        row<-data.frame(plant, crew,   left, right, type)
        extrema_imp_df <- rbind(extrema_imp_df, row) 
      }else{
        imp_ext <- find_imp_min(i, subset,n, dist_threshold)
        left <- unlist(imp_ext[1])
        right <- unlist(imp_ext[2])
        type <- unlist(imp_ext[3])
        i <- unlist(imp_ext[4])
        row<-data.frame(plant, crew,   left, right, type)
        extrema_imp_df <- rbind(extrema_imp_df, row) 
        
      }
      
    }
    #######################################################################
    ########################################################################
    # update the plant data with extrema and important extrema
    sub <- subset 
    sub$plant <- rep(plant, nrow(sub))
    sub$extrema = rep("not extrema", nrow(sub)) 
    sub$imp_extrema = rep("not important extrema", nrow(sub))
    # put "end" in the first and last row of the sub dataframe
    number <- nrow(sub)
    sub$extrema[1] <- "end"
    sub$extrema[number] <- "end"
    sub$imp_extrema[1] <- "end"
    sub$imp_extrema[number] <- "end"
    index <- 1
    for (k in 1:nrow(extrema_df)){
      #print(k)
      left <- extrema_df$left[k]
      if (index < left){
        index <- left
      }
      if(extrema_df$left[k] == extrema_df$i[k]){
        #print("equal")
        sub$extrema[index] <- as.character(extrema_df$type[k])
        index <- index + 1
      }else{
        #print("not equal")
        for (m in extrema_df$left[k]:extrema_df$i[k]){
          #print(m)
          #print(as.character(extrema_df$type[k]))
          sub$extrema[index] <- as.character(extrema_df$type[k])
          index <- index + 1
          
        }
      }
    }
    #########################################################################
    ########################################################################
    ind_imp <- 1
    for (n in 1:nrow(extrema_imp_df) ){
      left <- extrema_imp_df$left[n]
      if(ind_imp < left){
        ind_imp <- left
      }
      if(extrema_imp_df$left[n] == extrema_imp_df$right[n]){
        sub$imp_extrema[ind_imp] <- as.character(extrema_imp_df$type[n])
        ind_imp <- ind_imp + 1
      }else{
        point <- sub$actual[ind_imp]
        for (t in extrema_imp_df$left[n]:extrema_imp_df$right[n]){
          if (sub$actual[t] == point){
            sub$imp_extrema[ind_imp] <- as.character(extrema_imp_df$type[n])
            ind_imp <- ind_imp + 1 
          }else{
            ind_imp <- ind_imp + 1
          }
          
          
        }
        
      }
    }
  }
  
  plant_df <- rbind.data.frame(plant_df,sub)
  ###################################################################
  ##################################################################
  
} # end of J loop




increments <- 500
con <- dbConnect(drv, dbname=database,host=host,port=port,user=user,password=password )
qry_role<-paste0("set role to datasci;")

dbGetQuery(con, qry_role)

#tbl_name_date <- paste0("lab_datasci.abs_",plant,"_compression_crew_tbl_",tbl_date)

if (test_flag=="test"){

  tbl_name_date <-paste("lab_datasci.abs",test_flag,plant,"compression_crew",tbl_date,"tbl", sep="_")
  
  } else {
  
  tbl_name_date<-paste("lab_datasci.abs",plant,"compression_crew",tbl_date,"tbl", sep="_")
}


print("Writing Compression Data to Database")
write_to_db(plant_df, tbl_name_date , grant_statement, increments)
######################################################
#Close Connection
#####################################################
print("Closing Connection")
lapply(dbListConnections(drv = dbDriver("PostgreSQL")), function(x) {dbDisconnect(conn = x)})
#dbDisconnect(con)
print("Closed Connection")
print(paste0("Compression for ",toupper(plant), " Completed "))

