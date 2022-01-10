#============================================
# Forecasting New Covid-19 Cases in Canada
#============================================


# Data Source: https://github.com/owid/covid-19-data/tree/master/public/data


#library (readr)
#vaccinationfile="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
#Covid19casesfile="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
#Variantfile="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/variants/covid-variants.csv"

#vaccination.data<-read_csv(url(vaccinationfile))
#Covid19cases.data<-read_csv(url(Covid19casesfile))
#Variant.data<-read_csv(url(Variantfile))

# install.packages("dplyr")
# Use the pipe %>% function in "dplyr" package
library(dplyr)

#Question 1 - throwing raw data to the model
model.raw.CAN <- auto.arima( CANNewcase, stepwise=FALSE, seasonal=TRUE)
model.raw.CAN
fit.raw.CAN <- arima(CANNewcase, order=c(2,1,3))
autoplot( forecast(fit.raw.CAN,15)) 

fc.CAN.raw<-forecast(fit.raw.CAN,15) #interested to see the fc mean
fc.CAN.raw$mean

#it's flat, clearly its not a good prediction, hence important to go through log, aic ..etc

#Question 2 
#load the data
Cases.data <- read.csv("/Users/jacquelinemak/MMA /MMA 867 Predictive Modelling/Assignment 3/owid-covid-data_v2.csv")

unique(Cases.data$iso_code)

#canada.vaccination.data = vaccination.data %>% filter(vaccination.data$iso_code == "CAN")
Canada.cases.data = Cases.data %>% filter(Cases.data$iso_code == "CAN")

#ts = time series
plot.ts(Canada.cases.data$new_cases) #number of new canada cases
plot.ts(Canada.cases.data$stringency_index) #measures how strict the public restriction is, as number of cases go up, more strict 
#plot.ts(US.vaccination.data$total_vaccinations)
plot.ts(Canada.cases.data$total_vaccinations)

# Preprocessing 
CANNewcase<- Canada.cases.data$new_cases
CANNewcase<-tail(CANNewcase,-100) # Remove the first 100 days when the case numbers are small
CANNewcase<-CANNewcase[-498] #Remove data on September 14th as I need to predict cases from Sept 14-28

#change to time series object, however there is no time stamp on it, so i need to tell it, first number corresponds to year 2020, and 122nd day of 2020
CANNewcase <- ts(CANNewcase, frequency=365, start=c(2020, 126)) #add frequency which is # of days, May 5th, 2020 is the start

plot.ts(CANNewcase, xlab="Date", ylab="CAN New cases")
#this object becomes time series object

#the magnitude of fluctuation, more volatile recently which would suggest things are a little different than before
#could be due to the variance or reporting reasons
#this time series does not have constant variance, hence not homosketacity (homo = constant variance), as cases go up, variances tend to be bigger
## of cases go to, variances tend to go up, u can see in the middle, this is a problem in time series analysis, it can only work well when you have constant variation
#if u apply arima model to this time series, it will not make great forecasting cases 
#varianes not stable - what should we do? simple way is to log it to remove heteroskatacity or use boxcox

# We notice two properties:
# (1)  the variance increases with mean (heteroscedasticity)
# (2)  the data has cyclic pattern(seasonality)  
# We cannot directly apply the ARIMA model when either of the above behaviors shows up! 

#-----------(1) Stabilizing the Variance---------------
# To stabilized the variance, we need to transform the original data using Box-Cox Transform, 
# also known as Power transform. It also makes the data more Gaussian (more normal)   
CANNewcase.lambda <- BoxCox.lambda(CANNewcase)  # The transform is parameterized by lambda
# lambda is usually chosen between -2 and 2
CANNewcase.BoxCox<-BoxCox(CANNewcase, CANNewcase.lambda)

# Check the transformed data and compare it with log transform (skip this step in lecture)
par(mfrow=c(1,2))
plot.ts(CANNewcase.BoxCox, xlab="Date", ylab="CAN New cases")
plot.ts(log(CANNewcase), xlab="Date", ylab="CAN New cases")
# if lambda is close to zero, BoxCox is essentially doing log transform.

logCANNewcase<-log(CANNewcase) # We use log transform for simplicity 
plot.ts(logCANNewcase, xlab="Date", ylab="log CAN New cases")
#although you still can see more variability starting April of this year, but overall the variance is not changing with respect to the level of y, seems to be stable
#how to deal with after April. cut time series into two parts and use most recent part to predict
#all of the previous info seems to obsolete, justify this bc the dynamic of the pandemic has changed
#ppl are interested the level of cases, not the fluctuation 
#more variability  recently maybe the reporting data in CAN works more accurately after April 2021
#log is an important preprocessed if we have unstable variances otherwise get wrong predictions
#the seasonality is still here, bad for arima, bc arima cannot handle seasonlity, it will mess up the prediction

#-----------(2) Remove Seasonality through Seasonal Differencing ---------------
# To check the period of cyclic pattern, use the autocorrelation function 
#how often the pattern is repeating
#we can use acf function, a very powerful tool in time series, and tell us what is the period
Acf(diff(logCANNewcase,1),lag.max =25) 

#AIC measures how good the fit is, and how simple the model is
# We see spikes at p=7,14,21.. What does it suggest?  multiply of 7, big spikes poping at fixed time intervals, aka seasonlity and the period is 7
# Here we first perform regular differencing "diff(logCANNewcase,1)" to make the series more stationary, so its seasonality becomes easier to detect in the Acf plot
#big spike at lag 7, 14, 21; the pattern is weeks, multiple of 7
#when u see big spike at fixed time interval, there is seasonlity in the time series data, the period is simply 7
#make sure to specify max lag
#look at daily changes hence the diff function used
#main function of acf is to figure out the period, and we can see the period here is 7, so we need to remove it before doing time series analysis

# We now remove the seasonality using seasonal differencing
logCANNewcase.deSeasonality <- diff(logCANNewcase,7) # period is 7 because of the weekly pattern, the difference between today's number and last thursday's number 
plot.ts(logCANNewcase.deSeasonality, xlab="Date", ylab="log CAN New Case after removing trend and seasonality") #theres no weekly pattern
Acf(logCANNewcase.deSeasonality,lag.max =25) 
#as you can see the big spikes are gone, seasonality has been remvoved, now i can see this dataset, i have logged the data and deseasoned it

# What does "logCANNewcase.deSeasonality" mean? log the data and deseason the data 

#-------------Automatic ARIMA Modeling -------------------
# To begin, we use an automated algorithm to find a good model. However, there is no guarantee that it is the best model. So we treat it as a starting point. 
model.auto.CAN <- auto.arima( logCANNewcase.deSeasonality, stepwise=FALSE, seasonal= FALSE) #Fit using the Hyndman-Khandakar algorithm (Hyndman & Khandakar, 2008)
model.auto.CAN
#p, d, q <- this function automatically gives u these 
#p is the order of the AR term - it refers to the number of lags of Y to be used as predictors. 
#d is the number of differencing required to make the time series stationary
#q is the order of the MA term -  It refers to the number of lagged forecast errors that should go into the ARIMA Model.

#suggests 0 zero which means there is no intercept, MA is 5, 0 is autoregressive terms 
#if you want to predict tmrw's case, multiply today case by =-0.1202, yesterday case by 0.5117 and then add them up to make prediction for tomorrow 
#second row is the standard dev - no need to pay attention to these numbers
#AIC - measure of how good your model is, the lower the better 
#AICc - u want this measure to be low 
#we use it to make comparison of the different models and decide which one to choose
#AIC heavily depends on transformation. for example, we used log transformation
#if you dont use log transform, your AIC will be very different 

# It suggests a ARIMA(0,0,5) model with zero mean
checkresiduals(model.auto.CAN)  # Check the quality of fit. Residuals should: 
# (1) not have any significant autocorrelation
# (2) follow normal distribution
# (3) have stable variance over time
#the most important thing is check ACF plot, use to examine the residuals, we should NOT see many sticks coming out, which means the residuals are not significantly correlated with each other
#if we see a lot of sticks, then we neeed to worry bc predicton error of 2 days ago is related to the prediction today 
#which means the model did not improve, u should go back, and change moving average 
#rule of thumb - lets say 150 sticks, we dont want to see more than 5% sticks sticking out, which is 7.5 sticking out
# we have only 5 sticks sticking out, so that means its oK
#check the residuals are normal over time, this one looks good 
#arima suggested 0, 0, 5 (MA)

#we can now fit the model, autoarima suggested 0, 0 , 5
# We can use the auto selected model to make forecasting; the arima function can build the model
fit.yourself.CAN <- Arima(logCANNewcase, order=c(0,0,5), seasonal=list(order=c(0,1,0),period=7)) # The seasonal differencing with period=7 is equivalent to "seasonal=list(order=c(0,1,0),period=7)"
#log y variable and this time series has seasonlity and the period is 7
fit.yourself.CAN #the coefficients are same as before
autoplot( forecast(fit.yourself.CAN,15) ) #we can forecast the # of cases in the future; 15 days in advance (sept 13-28)
#this autoplot can predict those detailed fluctuations, 15 days, you can see it predicted lower number on weekends and higher in the weekdays, but overall
#the model suggests the # of cases will be stable - the more recent weeks will be slowing down - getting plateau
#time series doesnt know the schools are opening now, all it uses are more recent data to predict the future, which is also a limitation
#all time series know is use more recent data to predict future which is also a limitation
#those shaded part are marginal error - we can say that the marginal error is getting bigger as we move further to the future bc it is expected

# Plot the forecasting in the original scale (to see the numbers)
fc.CAN<-forecast(fit.yourself.CAN,15) #interested to see the fc mean 

fc.CAN$mean #most predicted value for tomorrow is 8.795205 -> exp(12.44472) = 6602.51 cases 

fc.CAN$x <- exp(fc.CAN$x)
fc.CAN$mean <- exp(fc.CAN$mean)
fc.CAN$mean
fc.CAN$lower <- exp(fc.CAN$lower)
fc.CAN$upper <- exp(fc.CAN$upper) #upper bound of marginal error
autoplot(fc.CAN) #this is the actual prediction curve
#this is the prediction for the actual number of cases, actualy pandemic curve, as we move further, the marginal error gets bigger
#first log, then differencing 

#Question 3 -------------Improving the Automatically selected Model -------------------
# Note that the auto.arima function may not always find the model with the lowest AIC/AICc/BIC. 
# To improve the existing model, we may explore other models manually to see if one could yield a lower criterion than the existing model. For example: 
fit.alternative1.CAN <- Arima(logCANNewcase, order=c(7,0,9), seasonal=list(order=c(0,1,0),period=7)) #Increased MA from 5 to 9 to see if i get a lower AIC
fit.alternative1.CAN #this AIC is lower, so better than previous model
checkresiduals(fit.alternative1.CAN)
fc1.CAN<-forecast(fit.alternative1.CAN,15)
#use this model because it has a lower AIC

fc1.CAN$x <- exp(fc1.CAN$x)
fc1.CAN$mean <- exp(fc1.CAN$mean)
fc1.CAN$mean
fc1.CAN$lower <- exp(fc1.CAN$lower)
fc1.CAN$upper <- exp(fc1.CAN$upper)
autoplot(fc1.CAN)


#Question 4 - 

#============================================
# ARIMA with Covariates ("dynamic regression")
#============================================

#on top arima model, you can add time series x which represents how strict the restriction measure is, xt (restriction level today) + xt-1(restriction level yesterday)

# Lagged predictors. Test 0, 1, 2 or 3 lags.

#Remove the first 100 rows,
Canada.cases.data1 <- tail(Canada.cases.data,-100) #remove first 100 rows as there are many N/A
Canada.cases.data1 <- Canada.cases.data1[-498,] #remove row for Sept 14th data


#46 stringency
#35 total vaccination
#8 new deaths

stringency<- cbind(Canada.cases.data1[,46],
                      c(NA,Canada.cases.data1[1:496,46]),
                      c(NA,NA,Canada.cases.data1[1:495,46]),
                      c(NA,NA,NA,Canada.cases.data1[1:494,46]))
colnames(stringency) <- paste("AdLag",0:3,sep="")
stringency


# Choose optimal lag length for stringency index based on AIC
# Restrict data so models use same fitting period
fit1 <- auto.arima(logCANNewcase[4:497], xreg=stringency[4:497,1], d=0) #without lag
fit2 <- auto.arima(logCANNewcase[4:497], xreg=stringency[4:497,1:2], d=0) 
fit3 <- auto.arima(logCANNewcase[4:497], xreg=stringency[4:497,1:3], d=0)#2 lags, predicting tomorrow's quote, im using today'tv spending and yesterday's spending
fit4 <- auto.arima(logCANNewcase[4:497], xreg=stringency[4:497,1:4], d=0) #the most 4 days 
#since we don't know which is the right lag, therefore calculate aic and see which one is lower

# Compute Akaike Information Criteria
AIC(fit1) #lowest AIC (443.8593), without lag
AIC(fit2) 
AIC(fit3) 
AIC(fit4) 

#How many lags would predict the lowest AIC
# Compute Bayesian Information Criteria
BIC(fit1) #strong evidence that first model is good
BIC(fit2) 
BIC(fit3)
BIC(fit4) 

#good habit to check BIC, found that we have to use no lag
#Best fit (as per AIC and BIC) is with all data, so the final model becomes (using no lag)
fit <- auto.arima(logCANNewcase, xreg=stringency[,1], d=0) # d is the order of first-differencing
fit
#3, 0 , 0 - > to predict tmrw's quote, use up to t-3
#yt (tmrw's cases) = 10.3320 + 0.3394yt-1 + 0.2272yt-2 + 0.4073yt-3 + errort - 0.0399xt (tmrw) 
#tmrw's cases (t) = depends on today's cases (t-1), yesterday's cases (t-2), and the day before yesterday (t-3)
#also depends on tmrw's stringency index (t)
#if you want tmrw's cases, you need to know stringency index for tmrw
#xt is (stringecy infex) is something you can test


# forecast covid cases with stringency index = 60
#stringency index = 60 for 15 days

fc60 <- forecast(fit, xreg=rep(60,15), h=15)
plot(fc60, main="Forecast cases with stringency index set to 60", ylab="Log CAN New Cases")

# forecast covid cases with stringency index = 10
#stringency index = 10 for 15 days

fc5 <- forecast(fit, xreg=rep(5,15), h=15)
plot(fc5, main="Forecast cases with stringency index set to 5", ylab="Log CAN New Cases")











