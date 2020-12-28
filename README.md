# Cifar-10-SVM
Working on Cifar-10 Dataset and Finding out its best SVM Model.
1-1 HOG - Combine HOG and color histogram (must be implemented from scratch) on a whole ie., (hog + color hist) - feature descriptor(2)

1-1 PCA - Perform PCA using sklearn on the dataset such that 90% of the total variance is retained - feature descriptor(1)

1-2 PCA TSNE - Visualize the 2D t-SNE plot. with PCA

1-2 TSNE_HOG+Color - Visualize the 2D t-SNE plot. (with HOG and color histogram)

1-3 HOG_rbr10_GridSearchCV scaled - Use GridSearchCV (cv=5) to find the best parameters (C, kernel, γ in case of gaussian
kernel) of SVM using the train set. Report the accuracies (train, test) and the run-times on the best parameters obtained. State your observations (if any) on the obtained best parameters. (with HOG and color histogram)

1-3 PCA rbf c = 10.py - Use GridSearchCV (cv=5) to find the best parameters (C, kernel, γ in case of gaussian
kernel) of SVM using the train set. Report the accuracies (train, test) and the run-times on the best parameters obtained. State your observations (if any) on the obtained best parameters.(with PCA)

1-4 HOG - Develop a new training set by extracting the support vectors from the SVM fitted in (3). Now fit another SVM with the new training set and report the accuracies(train, test). Compare the accuracies from (3) and (4). (with HOG and color histogram)

1-4 PCA - Develop a new training set by extracting the support vectors from the SVM fitted in (3). Now fit another SVM with the new training set and report the accuracies(train, test). Compare the accuracies from (3) and (4). (with PCA)
