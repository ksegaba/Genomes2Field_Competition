# Author: Peipei Wang
# Modified by: Kenia Segura Abá
library(rrBLUP)
library(data.table)
set.seed(42)
args = commandArgs(trailingOnly=TRUE)
X_file <- args[1] # your genetic matrix, e.g., geno.csv
Y_file <- args[2] # your phenotypic matrix, e.g., pheno.csv
feat_file <- args[3] # selected features or "all" for all the markers in the genetic matrix
trait <- args[4] # the column name of your target trait, or "all" for all the traits in the pheno matrix
test_file <- args[5] # file with individuals in test set
cv <- as.numeric(args[6]) # the fold number of the cross-validation scheme
number <- as.numeric(args[7]) # how many times your want to repeat the cross-validation scheme
cvs_file <- args[8] # the CVs file
save_name <- args[9]


Y <- read.csv(Y_file, row.names=1) 
Test <- scan(test_file, what='character')

# if file is larger than 10Mb, using fread to read the file
if(file.size(X_file) > 10*1024*1024){
	# Subset X if feat_file is not all
	if (feat_file != 'all'){
		print('Pulling features to use...')
		FEAT <- scan(feat_file, what='character')
		X <- fread(X_file,select=c('ID',FEAT))
		fwrite(X,paste(feat_file, '_geno.csv',sep=''),sep = ",",quote=FALSE) # 01/22/2022 Kenia: If feat_file is contains the path, then put first
		X <- read.csv(paste(feat_file, '_geno.csv',sep=''), row.names=1) # 01/22/2022 Kenia: If feat_file is contains the path, then put first
	} else{
		X <- as.matrix(fread(X_file),rownames=1)
		}
}else{
	X <- read.csv(X_file, row.names=1) 
	# Subset X if feat_file is not all
	if (feat_file != 'all'){
		print('Pulling features to use...')
		FEAT <- scan(feat_file, what='character')
		X <- X[FEAT]
	}
}


cvs <- read.csv(cvs_file, row.names=1)
cvs_all <- merge(Y,cvs,by="row.names",all.x=TRUE)
rownames(cvs_all) <- cvs_all$Row.names
cvs_all <- cvs_all[,(dim(Y)[2]+2):ncol(cvs_all)]
cvs_all[is.na(cvs_all)] = 0

# make sure X and Y have the same order of rows as cvs_all
X <- X[rownames(cvs_all),]
Y <- Y[rownames(cvs_all),]


if (trait == 'all') {
  print('Modeling all traits')
} else {
  Y <- Y[trait]
}

# 07/26/2022 Kenia: Added coefficient of determination (R^2) function
r2_score <- function(preds, actual) {
	# This function is comparable to sklearn's r2_score function
	# It computes the coefficient of determination (R^2)
	rss <- sum((preds - actual) ^ 2) # residual sum of squares
	tss <- sum((actual - mean(actual)) ^ 2) # total sum of squares
	return(1 - (rss/tss)) # return R^2 value
}

PCC_cv <- c() # 01/19/2022 Kenia: PCC to output
PCC_test <- c() # 01/19/2022 Kenia: PCC to output
R2_cv <- c()
R2_test <- c()
Predict_validation <- c()
Predict_test <- c()
for(i in 1:length(Y)){
	print(names(Y)[i])
	corr_CV <- c() # 01/19/2022 Kenia: PCC across CV folds for validation and predicted label
	corr_test <- c() # 01/19/2022 Kenia: PCC across CV folds for test set and pred label
	Accuracy_test <- c()
	Accuracy_CV <- c()
	Coef <- c()
	Error <- c() # 12/12/2021 Kenia: Model residual error term (Ve) per trait
	Beta <- c() # 12/12/2021 Kenia: Model fixed effects (β) per trait
	pred_val <- c()
	pred_test <- c()
	for(k in 1:number){
		print(k)
		tst = cvs_all[,k]
		Coeff <- c()
		Errors <- c() # 12/12/2021 Kenia: Model residual error term (Ve) per j-cv repetition
		Betas <- c() # 12/12/2021 Kenia: Model fixed effects (β) per j-cv repetition
		y_test <- c()
		yhat <- data.frame(cbind(Y, yhat = 0))
		yhat$yhat <- as.numeric(yhat$yhat)
		row.names(yhat) <- row.names(Y)
		for(j in 1:cv){
			validation <- which(tst==j)
			training <- which(tst!=j & tst!=0)
			test <- which(tst==0)
			yNA <- Y[,i]
			yNA[validation] <- NA # Mask yields for validation set
			yNA[test] <- NA # Mask yields for test set
			# Build rrBLUP model and save yhat for the masked values
			# predict marker effects
			coeff <- mixed.solve(y=Y[training,i], Z=X[training,], K=NULL, SE=FALSE, return.Hinv=FALSE)
			Coeff <- rbind(Coeff,coeff$u)
			Errors <- rbind(Errors, coeff$Ve) # 12/12/2021 Kenia: Model residual error term (Ve) per cv fold
			Betas <- rbind(Betas, coeff$beta) # 12/12/2021 Kenia: Model fixed effects term (β) per cv fold
			effect_size <- as.matrix(coeff$u)
			# predict breeding 
			#rrblup <- mixed.solve(y=yNA, K=A.mat(X))
			#yhat$yhat[validation] <- rrblup$u[validation]
			yhat$yhat[validation] <- (as.matrix(X[validation,]) %*% effect_size)[,1] + c(coeff$beta)
			#yhat$yhat[test] <- rrblup$u[test]
			yhat$yhat[test] <- (as.matrix(X[test,]) %*% effect_size)[,1] + c(coeff$beta)
			y_test <- cbind(y_test,yhat$yhat)
			}
		corr_cv <- cor(yhat[which(tst!=0),i], yhat$yhat[which(tst!=0)]) # 01/19/2022 Kenia: Added PCC value column
		corr_CV <- c(corr_CV, corr_cv)
		#Accuracy_CV <- c(Accuracy_CV,corr_cv^2) # PCC value of cross-validation
		Accuracy_CV <- c(Accuracy_CV, r2_score(yhat[which(tst!=0),i], yhat$yhat[which(tst!=0)])) # 07/26/2022 Kenia: Added coefficient of determination (R^2) function
		y_test <- cbind(y_test,rowMeans(y_test))
		corr_Test <- cor(yhat[which(tst==0),i], y_test[which(tst==0),ncol(y_test)]) # 01/19/2022 Kenia: Added PCC value column
		corr_test <- c(corr_test, corr_Test)
		#Accuracy_test <- c(Accuracy_test,corr_Test^2) # PCC value of test set
		Accuracy_test <- c(Accuracy_test, r2_score(yhat[which(tst==0),i], y_test[which(tst==0),ncol(y_test)])) # 07/26/2022 Kenia: Added coefficient of determination (R^2) function
		Coef <- rbind(Coef,colMeans(Coeff))
		Error <- rbind(Error, colMeans(Errors)) # 01/19/2022 Kenia: Model average residual errors across folds
		Beta <- rbind(Beta, colMeans(Betas)) # 01/19/2022 Kenia: Model average fixed effects across folds
		pred_val <- cbind(pred_val,yhat$yhat[which(tst!=0)])
		pred_test <- cbind(pred_test,y_test[which(tst==0),ncol(y_test)])
		}
	PCC_cv <- cbind(PCC_cv, corr_CV)
	PCC_test <- cbind(PCC_test, corr_test)
	R2_cv <- cbind(R2_cv,Accuracy_CV)
	R2_test <- cbind(R2_test,Accuracy_test)
	write.csv(Coef,paste('Coef_',save_name,'_',names(Y)[i],'.csv',sep=''),row.names=FALSE,quote=FALSE) # Coefficients
	write.csv(Error,paste('Error_',save_name,'_',names(Y)[i],'.csv',sep=''),row.names=FALSE,quote=FALSE) # 01/19/2022 Kenia: Save model average residual errors across folds
	write.csv(Beta,paste('Beta_',save_name,'_',names(Y)[i],'.csv',sep=''),row.names=FALSE,quote=FALSE) # 01/19/2022 Kenia: Save model average fixed effects across folds
	colnames(pred_val) <- paste(names(Y)[i],'_',1:number,sep='')
	Predict_validation <- cbind(Predict_validation,pred_val)
	colnames(pred_test) <- paste(names(Y)[i],'_',1:number,sep='')
	Predict_test <- cbind(Predict_test,pred_test)
}

colnames(PCC_cv) <- names(Y)
colnames(PCC_test) <- names(Y)
write.csv(PCC_cv,paste('PCC_cv_results_',save_name,'_',trait,'.csv',sep=''),row.names=FALSE,quote=FALSE)
write.csv(PCC_test,paste('PCC_test_results_',save_name,'_',trait,'.csv',sep=''),row.names=FALSE,quote=FALSE)
colnames(R2_cv) <- names(Y)
colnames(R2_test) <- names(Y)
write.csv(R2_cv,paste('R2_cv_results_',save_name,'_',trait,'.csv',sep=''),row.names=FALSE,quote=FALSE)
write.csv(R2_test,paste('R2_test_results_',save_name,'_',trait,'.csv',sep=''),row.names=FALSE,quote=FALSE)
rownames(Predict_validation) <- rownames(X)[which(tst!=0)]
write.csv(Predict_validation,paste('Predict_value_cv_',save_name,'_',trait,'.csv',sep=''),row.names=TRUE,quote=FALSE)
rownames(Predict_test) <- rownames(X)[test]
write.csv(Predict_test,paste('Predict_value_test_',save_name,'_',trait,'.csv',sep=''),row.names=TRUE,quote=FALSE)
